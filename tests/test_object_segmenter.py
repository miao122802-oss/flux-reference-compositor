import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image, ImageDraw

import object_segmenter


def rectangle_mask(size, box):
    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rectangle(box, fill=255)
    return mask


class ObjectPromptTests(unittest.TestCase):
    def test_auto_prompt_is_constrained_to_green_edit_mask(self):
        size = (96, 80)
        background = Image.new("RGB", size, (100, 150, 210))
        generated = background.copy()
        ImageDraw.Draw(generated).rectangle((36, 24, 62, 60), fill=(240, 180, 30))
        edit = rectangle_mask(size, (16, 10, 82, 72))

        prompt = object_segmenter.build_auto_prompt(background, generated, edit)
        allowed = np.asarray(edit) >= 128
        difference = np.asarray(prompt.difference_mask) >= 128
        self.assertFalse(np.any(difference & ~allowed))
        for x, y in prompt.positive_points + prompt.negative_points:
            self.assertTrue(allowed[min(79, int(y)), min(95, int(x))])
        self.assertEqual(prompt.box_xyxy, tuple(float(value) for value in edit.getbbox()))

    def test_yellow_box_uses_extreme_pixels_of_irregular_user_mask(self):
        size = (120, 90)
        background = Image.new("RGB", size, (100, 150, 210))
        generated = background.copy()
        ImageDraw.Draw(generated).ellipse((45, 25, 70, 65), fill=(240, 180, 30))
        edit = Image.new("L", size, 0)
        draw = ImageDraw.Draw(edit)
        draw.ellipse((12, 18, 30, 42), fill=255)
        draw.rectangle((82, 55, 105, 78), fill=255)

        prompt = object_segmenter.build_auto_prompt(background, generated, edit)

        self.assertEqual(prompt.box_xyxy, (12.0, 18.0, 106.0, 79.0))

    def test_quality_rejects_roi_edge_and_oversized_candidate(self):
        size = (64, 64)
        background = Image.new("RGB", size, (120, 160, 210))
        generated = background.copy()
        ImageDraw.Draw(generated).rectangle((23, 20, 40, 44), fill=(230, 170, 20))
        edit = rectangle_mask(size, (6, 6, 57, 57))
        prompt = object_segmenter.build_auto_prompt(background, generated, edit)

        good = rectangle_mask(size, (21, 18, 42, 46))
        oversized = edit.copy()
        _good_quality, good_accepted, _ = object_segmenter._candidate_quality(
            good, 0.90, prompt, edit
        )
        _bad_quality, bad_accepted, details = object_segmenter._candidate_quality(
            oversized, 0.99, prompt, edit
        )
        self.assertTrue(good_accepted)
        self.assertFalse(bad_accepted)
        self.assertGreater(details["area_ratio"], 0.90)
        chosen = object_segmenter._choose_prediction(
            {
                "masks": [oversized, good],
                "scores": [0.99, 0.90],
                "logits": [np.zeros((4, 4)), np.ones((4, 4))],
                "kind": "fake",
            },
            prompt,
            edit,
        )
        self.assertTrue(chosen.accepted)
        self.assertEqual(chosen.mask.getbbox(), good.getbbox())


class CompositeTests(unittest.TestCase):
    def test_negative_cleanup_mask_is_binary_and_removes_outer_sam_ring(self):
        mask = rectangle_mask((48, 48), (10, 10, 37, 37))
        cleaned = object_segmenter.adjust_object_mask(mask, mask.size, -2)
        values = set(np.unique(np.asarray(cleaned)).tolist())
        self.assertEqual(values, {0, 255})
        self.assertEqual(cleaned.getpixel((10, 24)), 0)
        self.assertEqual(cleaned.getpixel((12, 24)), 255)
        self.assertEqual(cleaned.getpixel((24, 24)), 255)

    def test_subject_feather_is_inward_and_never_creates_outer_halo(self):
        size = (48, 40)
        original = Image.new("RGB", size, (30, 100, 210))
        generated = Image.new("RGB", size, (235, 210, 80))
        object_mask = rectangle_mask(size, (16, 10, 31, 29))
        result = object_segmenter.composite_object_layers(
            original,
            generated,
            object_mask,
            Image.new("L", size, 255),
            (0, 0, *size),
            object_feather=2.0,
            object_edge_expand=0,
            seam_gaussian_radius=0,
            seam_gaussian_strength=0,
            shadow_enabled=False,
        )
        self.assertEqual(result.object_alpha.getpixel((15, 20)), 0)
        self.assertEqual(result.final.getpixel((15, 20)), (30, 100, 210))
        self.assertGreater(result.object_alpha.getpixel((16, 20)), 0)
        self.assertEqual(result.object_alpha.getpixel((24, 20)), 255)

    def test_negative_mask_adjustment_shrinks_without_moving_subject(self):
        size = (40, 40)
        original = Image.new("RGB", size, (10, 20, 30))
        generated = Image.new("RGB", size, (220, 180, 40))
        object_mask = rectangle_mask(size, (10, 10, 29, 29))
        result = object_segmenter.composite_object_layers(
            original,
            generated,
            object_mask,
            Image.new("L", size, 255),
            (0, 0, *size),
            object_feather=0,
            object_edge_expand=-2,
            seam_gaussian_radius=0,
            seam_gaussian_strength=0,
            shadow_enabled=False,
        )
        self.assertEqual(result.final.getpixel((10, 20)), (10, 20, 30))
        self.assertEqual(result.final.getpixel((12, 20)), (220, 180, 40))
        self.assertEqual(result.final.getpixel((20, 20)), (220, 180, 40))

    def test_manual_sam_mask_keeps_new_color_and_discards_generated_panel(self):
        size = (64, 48)
        original = Image.new("RGB", size, (8, 8, 8))
        generated = Image.new("RGB", size, (135, 135, 135))
        ImageDraw.Draw(generated).rectangle((22, 12, 43, 35), fill=(225, 175, 35))
        object_mask = rectangle_mask(size, (22, 12, 43, 35))
        allowed = Image.new("L", size, 255)

        result = object_segmenter.composite_object_layers(
            original,
            generated,
            object_mask,
            allowed,
            (0, 0, *size),
            object_feather=0,
            object_edge_expand=0,
            seam_gaussian_radius=0,
            seam_gaussian_strength=0,
            shadow_enabled=False,
        )

        self.assertEqual(result.final.getpixel((32, 24)), (225, 175, 35))
        self.assertEqual(result.final.getpixel((10, 10)), (8, 8, 8))

    def test_insert_mode_keeps_every_non_object_pixel_exact(self):
        width, height = 48, 40
        base_array = np.arange(width * height * 3, dtype=np.uint16).reshape(height, width, 3) % 256
        base = Image.fromarray(base_array.astype(np.uint8), "RGB")
        generated = Image.new("RGB", (32, 28), (240, 180, 20))
        object_mask = rectangle_mask((32, 28), (10, 8, 20, 20))
        core_full = rectangle_mask((width, height), (8, 6, 39, 33))
        result = object_segmenter.composite_object_layers(
            base,
            generated,
            object_mask,
            core_full,
            (8, 6, 40, 34),
            object_feather=0,
            shadow_enabled=False,
        )
        final = np.asarray(result.final)
        alpha_full = np.zeros((height, width), dtype=bool)
        alpha_full[6:34, 8:40] = np.asarray(result.object_alpha) > 0
        self.assertTrue(np.array_equal(final[~alpha_full], base_array.astype(np.uint8)[~alpha_full]))

    def test_replace_mode_uses_clean_background_behind_small_object(self):
        size = (40, 40)
        cleaned_background = Image.new("RGB", size, (80, 145, 210))
        generated = Image.new("RGB", size, (230, 175, 25))
        object_mask = rectangle_mask(size, (15, 10, 25, 28))
        core = Image.new("L", size, 255)
        result = object_segmenter.composite_object_layers(
            cleaned_background,
            generated,
            object_mask,
            core,
            (0, 0, 40, 40),
            object_feather=0,
            shadow_enabled=False,
        )
        self.assertEqual(result.final.getpixel((5, 5)), cleaned_background.getpixel((5, 5)))
        self.assertEqual(result.final.getpixel((20, 18)), generated.getpixel((20, 18)))

    def test_shadow_is_multiplicative_and_does_not_copy_generated_texture(self):
        width = height = 80
        yy, xx = np.mgrid[:height, :width]
        base_values = (150 + ((xx + yy) % 25)).astype(np.uint8)
        base_array = np.repeat(base_values[..., None], 3, axis=2)
        base = Image.fromarray(base_array, "RGB")
        generated_array = base_array.copy()
        generated_array[38:66, 18:62] = 70
        # Loud synthetic texture that must not be pasted through the shadow.
        generated_array[42:60:2, 12:68:2] = (10, 240, 10)
        generated = Image.fromarray(generated_array, "RGB")
        object_mask = rectangle_mask((width, height), (28, 18, 51, 48))
        core = Image.new("L", (width, height), 255)
        result = object_segmenter.composite_object_layers(
            base,
            generated,
            object_mask,
            core,
            (0, 0, width, height),
            object_feather=0,
            shadow_enabled=True,
            shadow_radius=20,
            shadow_strength=0.9,
        )
        shadow = np.asarray(result.shadow_mask) > 0
        outside_object = np.asarray(object_mask) == 0
        pixels = np.argwhere(shadow & outside_object)
        self.assertGreater(len(pixels), 0)
        y, x = pixels[len(pixels) // 2]
        final_pixel = np.asarray(result.final)[y, x].astype(int)
        base_pixel = base_array[y, x].astype(int)
        self.assertTrue(np.all(final_pixel <= base_pixel))
        self.assertLessEqual(int(final_pixel.max() - final_pixel.min()), 1)

    def test_gaussian_seam_reduces_low_frequency_boundary_jump(self):
        width = height = 96
        base = Image.new("RGB", (width, height), (80, 145, 215))
        generated = Image.new("RGB", (width, height), (224, 168, 32))
        object_mask = rectangle_mask((width, height), (28, 20, 67, 75))
        core = Image.new("L", (width, height), 255)
        plain = object_segmenter.composite_object_layers(
            base,
            generated,
            object_mask,
            core,
            (0, 0, width, height),
            object_feather=2,
            object_edge_expand=1,
            seam_gaussian_radius=0,
            shadow_enabled=False,
        )
        softened = object_segmenter.composite_object_layers(
            base,
            generated,
            object_mask,
            core,
            (0, 0, width, height),
            object_feather=2,
            object_edge_expand=1,
            seam_gaussian_radius=4,
            seam_gaussian_strength=1,
            shadow_enabled=False,
        )
        plain_array = np.asarray(plain.final, dtype=np.float32)
        soft_array = np.asarray(softened.final, dtype=np.float32)
        plain_jump = float(np.mean(np.abs(plain_array[24:72, 27] - plain_array[24:72, 26])))
        soft_jump = float(np.mean(np.abs(soft_array[24:72, 27] - soft_array[24:72, 26])))
        self.assertLess(soft_jump, plain_jump)
        alpha = np.asarray(softened.object_alpha) > 0
        self.assertTrue(np.array_equal(soft_array[~alpha], np.asarray(base, dtype=np.float32)[~alpha]))


class RecoveryAndRefinementTests(unittest.TestCase):
    def _previous(self):
        size = (48, 48)
        background = Image.new("RGB", size, (100, 150, 210))
        generated = background.copy()
        ImageDraw.Draw(generated).rectangle((16, 12, 32, 35), fill=(230, 170, 20))
        edit = rectangle_mask(size, (4, 4, 43, 43))
        prompt = object_segmenter.build_auto_prompt(background, generated, edit)
        mask = rectangle_mask(size, (14, 10, 34, 37))
        prediction = {
            "masks": [mask],
            "scores": [0.9],
            "logits": [np.ones((256, 256), dtype=np.float32)],
            "kind": "sam2",
        }
        return generated, edit, object_segmenter._choose_prediction(prediction, prompt, edit)

    def test_click_refinement_reuses_logits_and_only_calls_sam(self):
        generated, edit, previous = self._previous()
        captured = {}

        def fake_segment(image, **kwargs):
            captured.update(kwargs)
            return {
                "masks": [rectangle_mask(image.size, (13, 9, 35, 38))],
                "scores": [0.93],
                "logits": [np.full((256, 256), 2, dtype=np.float32)],
                "kind": "sam2",
            }

        with patch("sam_segmenter.segment_with_prompts", side_effect=fake_segment):
            result = object_segmenter.refine_segmentation(
                generated,
                edit,
                previous,
                [[20, 20], [25, 25]],
                [[6, 6]],
                backend="sam2",
                sam2_path="sam2",
                sam1_path="sam1",
            )
        self.assertIs(captured["previous_mask_logits"], previous.logits)
        self.assertEqual(captured["positive_points"], [[20, 20], [25, 25]])
        self.assertIsNotNone(result.mask)

    def test_missing_dino_is_a_safe_lazy_error(self):
        with self.assertRaises(FileNotFoundError):
            object_segmenter.detect_grounding_box(
                Image.new("RGB", (32, 32)),
                Image.new("L", (32, 32), 255),
                "object",
                root=str(Path("missing-dino-root")),
                config=str(Path("missing-dino-config.py")),
                checkpoint=str(Path("missing-dino.pth")),
            )

    def test_sam_failure_returns_recoverable_result(self):
        generated, edit, _previous = self._previous()
        background = Image.new("RGB", generated.size, (100, 150, 210))
        with patch("sam_segmenter.segment_with_prompts", side_effect=RuntimeError("no sam")):
            result = object_segmenter.segment_generated_object(
                background,
                generated,
                edit,
                backend="sam2",
                sam2_path="missing",
                sam1_path="missing",
            )
        self.assertFalse(result.accepted)
        self.assertIsNone(result.mask)
        self.assertIn("initial composite has been preserved", result.message)

    def test_dino_no_detection_preserves_sam_result(self):
        generated, edit, _previous = self._previous()
        background = Image.new("RGB", generated.size, (100, 150, 210))
        oversized_prediction = {
            "masks": [edit.copy()],
            "scores": [0.98],
            "logits": [np.zeros((256, 256), dtype=np.float32)],
            "kind": "sam2",
        }
        with (
            patch("sam_segmenter.segment_with_prompts", return_value=oversized_prediction),
            patch("sam_segmenter.release_segmenter"),
            patch(
                "object_segmenter.detect_grounding_box",
                side_effect=RuntimeError("no detection"),
            ),
        ):
            result = object_segmenter.segment_generated_object(
                background,
                generated,
                edit,
                backend="sam2",
                sam2_path="sam2",
                sam1_path="sam1",
                dino_prompt="yellow character",
            )
        self.assertIsNotNone(result.mask)
        self.assertFalse(result.accepted)
        self.assertIn("DINO fallback failed", result.message)

    def test_coordinate_mapping_and_prompt_undo_history(self):
        self.assertEqual(
            object_segmenter.full_to_roi_point((130, 75), (100, 50, 200, 150)),
            [30.0, 25.0],
        )
        self.assertIsNone(
            object_segmenter.full_to_roi_point((99, 75), (100, 50, 200, 150))
        )
        history = [
            {"point": [20, 21], "label": 1},
            {"point": [8, 9], "label": 0},
        ]
        positive, negative = object_segmenter.merge_prompt_history(
            [[10, 10]], [[2, 2]], history
        )
        self.assertEqual(positive, [[10, 10], [20.0, 21.0]])
        self.assertEqual(negative, [[2, 2], [8.0, 9.0]])
        history.pop()
        positive, negative = object_segmenter.merge_prompt_history([], [], history)
        self.assertEqual(positive, [[20.0, 21.0]])
        self.assertEqual(negative, [])


if __name__ == "__main__":
    unittest.main()
