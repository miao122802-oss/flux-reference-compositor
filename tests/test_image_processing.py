"""CPU-only tests; no FLUX, SAM, CUDA or Gradio dependency is required."""

import unittest
from unittest import mock

import numpy as np
from PIL import Image, ImageDraw

import flux_inpaint
from image_processing import (
    composite_roi,
    composite_roi_with_outward_feather,
    crop_reference_cutout,
    despill_green_boundary,
    despill_green_seam_band,
    draw_user_mask_bbox,
    expand_mask,
    final_background_alpha,
    fill_mask_from_surroundings,
    harmonize_roi_boundary_colors,
    inward_feather_mask,
    mask_bbox_with_context,
    match_reference_background_colors,
    normalize_mask,
    paint_mask_green,
    resize_pair_for_model,
)


class ImageProcessingTests(unittest.TestCase):
    def test_second_stage_never_loads_any_flux_pipeline(self):
        source = Image.new("RGB", (32, 32), (70, 120, 190))
        generated = Image.new("RGB", source.size, (85, 130, 195))
        old_mask = Image.new("L", source.size, 0)
        new_mask = Image.new("L", source.size, 0)
        core = Image.new("L", source.size, 0)
        ImageDraw.Draw(core).rectangle((4, 4, 27, 27), fill=255)
        ImageDraw.Draw(old_mask).rectangle((4, 8, 10, 20), fill=255)
        ImageDraw.Draw(new_mask).rectangle((20, 8, 27, 21), fill=255)

        with mock.patch("flux_inpaint.load_pipeline") as load:
            result = flux_inpaint.edit_background_consistency(
                source,
                generated,
                old_mask,
                new_mask,
                core,
                original_cleanup_expand=1,
                generated_fill_expand=1,
                fill_radius=8,
                color_lock_radius=8,
            )

        load.assert_not_called()
        self.assertFalse(result["second_flux_enabled"])
        self.assertFalse(result["consistency_lora_enabled"])
        # A core-only pixel must remain exact Target in the final composite.
        # Only the old-subject cleanup region receives generated background.
        self.assertEqual(result["transition_alpha"].getpixel((27, 4)), 0)
        self.assertEqual(result["transition_alpha"].getpixel((7, 16)), 255)
        self.assertGreater(result["transition_alpha"].getpixel((2, 16)), 0)
        # Background color correction covers the full ROI and does not end at
        # the old-subject contour; only the generated subject is protected.
        self.assertEqual(result["color_application_alpha"].getpixel((27, 4)), 255)
        self.assertEqual(result["color_application_alpha"].getpixel((24, 12)), 0)

    def test_final_background_ownership_does_not_use_rectangular_core(self):
        source = Image.new("RGB", (80, 60), (70, 120, 190))
        generated = Image.new("RGB", source.size, (120, 150, 205))
        old_mask = Image.new("L", source.size, 0)
        new_mask = Image.new("L", source.size, 0)
        core = Image.new("L", source.size, 0)
        ImageDraw.Draw(core).rectangle((8, 6, 71, 53), fill=255)
        ImageDraw.Draw(old_mask).ellipse((14, 18, 30, 38), fill=255)

        result = flux_inpaint.edit_background_consistency(
            source,
            generated,
            old_mask,
            new_mask,
            core,
            original_cleanup_expand=2,
            generated_fill_expand=0,
            fill_radius=8,
            color_lock_strength=0,
            background_transition_radius=4,
        )
        alpha = result["transition_alpha"]
        composited = source.copy()
        composited.paste(result["consistent_background_roi"], (0, 0), alpha)
        # Far inside the rectangular generation mask, Target is byte-exact.
        self.assertEqual(alpha.getpixel((60, 12)), 0)
        self.assertEqual(composited.getpixel((60, 12)), source.getpixel((60, 12)))
        # The old subject itself is still completely replaced.
        self.assertEqual(alpha.getpixel((22, 28)), 255)
        self.assertEqual(composited.getpixel((22, 28)), generated.getpixel((22, 28)))

    def test_inward_feather_starts_from_zero_at_edit_contour(self):
        mask = Image.new("L", (64, 64), 0)
        ImageDraw.Draw(mask).rectangle((12, 12, 51, 51), fill=255)
        alpha = inward_feather_mask(mask, 4.0)
        # The outer contour must be pure Target, not the ~50/50 blend produced
        # by GaussianBlur(binary_mask).
        self.assertLessEqual(alpha.getpixel((12, 32)), 2)
        self.assertEqual(alpha.getpixel((0, 32)), 0)
        self.assertGreaterEqual(alpha.getpixel((32, 32)), 220)
        # The transition exists only on the inside of the edit region.
        self.assertGreater(alpha.getpixel((18, 32)), alpha.getpixel((12, 32)))

    def test_final_background_alpha_never_reveals_old_subject(self):
        core = Image.new("L", (72, 56), 0)
        old_cleanup = Image.new("L", core.size, 0)
        ImageDraw.Draw(core).rectangle((24, 12, 61, 45), fill=255)
        ImageDraw.Draw(old_cleanup).rectangle((8, 18, 30, 38), fill=255)
        alpha = final_background_alpha(core, old_cleanup, 4.0)
        # The complete user edit region, including its outer contour, remains
        # generated. Target is never mixed back inside the rectangle.
        self.assertEqual(alpha.getpixel((61, 28)), 255)
        self.assertEqual(alpha.getpixel((24, 28)), 255)
        # Every old-subject cleanup pixel, including its outermost contour,
        # remains fully replaced by clean background.
        self.assertEqual(alpha.getpixel((8, 28)), 255)
        self.assertEqual(alpha.getpixel((30, 28)), 255)
        self.assertEqual(alpha.getpixel((18, 18)), 255)
        # Softness exists only outside the union and eventually reaches zero.
        self.assertGreater(alpha.getpixel((7, 28)), 0)
        self.assertGreater(alpha.getpixel((62, 28)), 0)
        self.assertEqual(alpha.getpixel((71, 55)), 0)

    def test_outward_cosine_alpha_has_no_half_value_boundary_jump(self):
        core = Image.new("L", (96, 64), 0)
        ImageDraw.Draw(core).rectangle((24, 16, 63, 47), fill=255)
        alpha = final_background_alpha(core, Image.new("L", core.size, 0), 12)
        samples = [alpha.getpixel((63 + distance, 32)) for distance in range(0, 13)]
        self.assertEqual(samples[0], 255)
        # The first exterior pixel must stay close to opaque, not drop to the
        # ~128 produced by max(binary, GaussianBlur(binary)).
        self.assertGreaterEqual(samples[1], 245)
        self.assertEqual(samples[-1], 0)
        self.assertTrue(all(a >= b for a, b in zip(samples, samples[1:])))

    def test_subject_fill_uses_surroundings_without_leaking_subject_color(self):
        image = Image.new("RGB", (64, 64), (70, 110, 150))
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).ellipse((20, 20, 43, 43), fill=255)
        ImageDraw.Draw(image).ellipse((20, 20, 43, 43), fill=(240, 10, 10))
        filled = fill_mask_from_surroundings(image, mask, radius=12)
        self.assertEqual(filled.getpixel((0, 0)), (70, 110, 150))
        center = filled.getpixel((32, 32))
        self.assertLess(abs(center[0] - 70), 5)
        self.assertLess(abs(center[1] - 110), 5)
        self.assertLess(abs(center[2] - 150), 5)

    def test_consistency_prep_edits_old_subject_but_protects_new_subject(self):
        source = Image.new("RGB", (64, 48), (90, 100, 110))
        generated = Image.new("RGB", source.size, (95, 105, 115))
        old_mask = Image.new("L", source.size, 0)
        new_mask = Image.new("L", source.size, 0)
        core = Image.new("L", source.size, 255)
        ImageDraw.Draw(old_mask).rectangle((8, 12, 20, 30), fill=255)
        ImageDraw.Draw(new_mask).rectangle((38, 10, 54, 32), fill=255)
        ImageDraw.Draw(source).rectangle((8, 12, 20, 30), fill=(230, 20, 20))
        ImageDraw.Draw(source).rectangle((38, 10, 54, 32), fill=(30, 210, 60))
        ImageDraw.Draw(generated).rectangle((8, 12, 20, 30), fill=(40, 90, 220))
        ImageDraw.Draw(generated).rectangle((38, 10, 54, 32), fill=(240, 190, 20))
        prepared = flux_inpaint.prepare_consistency_backgrounds(
            source,
            generated,
            old_mask,
            new_mask,
            core,
            original_cleanup_expand=2,
            generated_fill_expand=2,
            fill_radius=8,
        )
        self.assertEqual(prepared["background_mask"].getpixel((14, 20)), 255)
        self.assertEqual(prepared["background_mask"].getpixel((46, 20)), 0)
        # Final ownership protects the subject, while the consistency model
        # repaints one continuous edit region rather than a subject-shaped
        # patch whose boundary would reveal the old silhouette.
        self.assertEqual(prepared["background_mask"].getpixel((39, 20)), 0)
        self.assertEqual(prepared["model_background_mask"].getpixel((39, 20)), 255)
        self.assertEqual(prepared["model_background_mask"].getpixel((46, 20)), 255)
        self.assertEqual(prepared["model_background_mask"].getpixel((4, 4)), 255)
        self.assertEqual(prepared["model_background_mask"].getpixel((14, 20)), 255)
        self.assertEqual(prepared["visible_repair_mask"].getpixel((14, 20)), 255)
        self.assertEqual(prepared["visible_repair_mask"].getpixel((46, 20)), 0)
        self.assertEqual(prepared["valid_target_background"].getpixel((14, 20)), 0)
        self.assertEqual(prepared["valid_target_background"].getpixel((4, 4)), 255)
        self.assertEqual(prepared["hidden_union"].getpixel((14, 20)), 255)
        self.assertEqual(prepared["hidden_union"].getpixel((46, 20)), 255)
        # The consistency conditioning plate must erase both subject regions.
        # Keeping first-pass pixels in the old-only area can feed a faint old
        # silhouette back into the second diffusion pass.
        self.assertEqual(prepared["generated_plate"].getpixel((14, 20)), (95, 105, 115))
        self.assertEqual(prepared["generated_plate"].getpixel((46, 20)), (95, 105, 115))
        # New-only area does not contain a subject in Target and stays intact.
        self.assertEqual(prepared["source_plate"].getpixel((46, 20)), (30, 210, 60))

    def test_consistency_model_mask_stays_binary_when_resized(self):
        image = Image.new("RGB", (97, 73), (80, 120, 180))
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).ellipse((23, 14, 78, 62), fill=255)
        _resized_image, resized_mask = resize_pair_for_model(
            image, mask, 64, 16, binary_mask=True
        )
        self.assertEqual(set(np.unique(np.asarray(resized_mask)).tolist()), {0, 255})

    def test_old_subject_dilation_survives_outside_original_core(self):
        source = Image.new("RGB", (64, 48), (80, 110, 150))
        generated = Image.new("RGB", source.size, (85, 115, 155))
        old_mask = Image.new("L", source.size, 0)
        new_mask = Image.new("L", source.size, 0)
        core = Image.new("L", source.size, 0)
        ImageDraw.Draw(core).rectangle((10, 10, 30, 36), fill=255)
        # Simulate an old subject whose SAM contour reaches beyond the original
        # edit region and needs a small safety margin for missed edge pixels.
        ImageDraw.Draw(old_mask).rectangle((27, 18, 34, 28), fill=255)
        prepared = flux_inpaint.prepare_consistency_backgrounds(
            source,
            generated,
            old_mask,
            new_mask,
            core,
            original_cleanup_expand=3,
            generated_fill_expand=0,
            fill_radius=8,
        )
        self.assertEqual(core.getpixel((36, 22)), 0)
        self.assertEqual(prepared["original_hidden"].getpixel((36, 22)), 255)
        self.assertEqual(prepared["effective_edit_mask"].getpixel((36, 22)), 255)
        self.assertEqual(prepared["model_background_mask"].getpixel((36, 22)), 255)
        self.assertEqual(prepared["visible_repair_mask"].getpixel((36, 22)), 255)
        self.assertEqual(prepared["background_mask"].getpixel((36, 22)), 255)

    def test_consistency_roi_uses_generated_pixels_under_new_subject(self):
        generated = Image.new("RGB", (48, 36), (230, 170, 30))
        corrected_background = Image.new("RGB", generated.size, (50, 120, 210))
        background_mask = Image.new("L", generated.size, 255)
        ImageDraw.Draw(background_mask).rectangle((16, 10, 31, 27), fill=0)
        merged = flux_inpaint.compose_consistency_roi(
            corrected_background, generated, background_mask
        )
        self.assertEqual(merged.getpixel((24, 18)), (230, 170, 30))
        self.assertEqual(merged.getpixel((4, 4)), (50, 120, 210))
        # The first pixel outside the subject is already 100% corrected
        # background; no hidden Gaussian transition may retain a pale halo.
        self.assertEqual(merged.getpixel((15, 18)), (50, 120, 210))

    def test_background_color_lock_reduces_low_frequency_color_error(self):
        reference = Image.new("RGB", (96, 72), (70, 145, 225))
        generated = Image.new("RGB", reference.size, (115, 100, 205))
        mask = Image.new("L", reference.size, 0)
        ImageDraw.Draw(mask).rectangle((12, 10, 83, 61), fill=255)
        corrected, info = match_reference_background_colors(
            reference, generated, mask, radius=16, strength=0.9, max_delta=80
        )
        self.assertTrue(info["applied"])
        self.assertLess(info["error_after"], info["error_before"])
        self.assertEqual(corrected.getpixel((0, 0)), generated.getpixel((0, 0)))
        before = sum(abs(a - b) for a, b in zip(reference.getpixel((40, 35)), generated.getpixel((40, 35))))
        after = sum(abs(a - b) for a, b in zip(reference.getpixel((40, 35)), corrected.getpixel((40, 35))))
        self.assertLess(after, before)

    def test_color_estimation_plate_is_never_composited(self):
        reference = Image.new("RGB", (48, 40), (100, 100, 100))
        generated = Image.new("RGB", reference.size, (10, 20, 30))
        estimation = Image.new("RGB", reference.size, (80, 80, 80))
        mask = Image.new("L", reference.size, 255)
        corrected, _info = match_reference_background_colors(
            reference,
            generated,
            mask,
            generated_estimation_image=estimation,
            radius=8,
            strength=1.0,
            max_delta=64,
        )
        # Delta is estimated as 100-80=20 but applied to the original
        # first-pass pixel (10,20,30), not to the temporary 80-gray plate.
        self.assertEqual(corrected.getpixel((24, 20)), (30, 40, 50))

    def test_background_color_lock_fades_without_a_hard_mask_line(self):
        reference = Image.new("RGB", (48, 32), (100, 100, 100))
        generated = Image.new("RGB", reference.size, (20, 20, 20))
        soft = Image.new("L", reference.size, 0)
        ImageDraw.Draw(soft).rectangle((0, 0, 15, 31), fill=255)
        ImageDraw.Draw(soft).rectangle((16, 0, 31, 31), fill=128)
        corrected, _info = match_reference_background_colors(
            reference,
            generated,
            soft,
            radius=8,
            strength=1.0,
            max_delta=80,
        )
        self.assertEqual(corrected.getpixel((8, 16)), (100, 100, 100))
        middle = corrected.getpixel((24, 16))[0]
        self.assertGreater(middle, 55)
        self.assertLess(middle, 65)
        self.assertEqual(corrected.getpixel((40, 16)), (20, 20, 20))

    def test_background_color_lock_ignores_invalid_old_subject_pixels(self):
        reference = Image.new("RGB", (96, 72), (80, 140, 210))
        # A vivid old object must not contaminate the extrapolated background
        # color field used inside its removed region.
        ImageDraw.Draw(reference).rectangle((30, 18, 65, 55), fill=(245, 15, 10))
        valid = Image.new("L", reference.size, 255)
        ImageDraw.Draw(valid).rectangle((26, 14, 69, 59), fill=0)
        generated = Image.new("RGB", reference.size, (120, 105, 180))
        apply_mask = Image.new("L", reference.size, 0)
        ImageDraw.Draw(apply_mask).rectangle((30, 18, 65, 55), fill=255)
        corrected, _info = match_reference_background_colors(
            reference,
            generated,
            apply_mask,
            reference_valid_mask=valid,
            radius=12,
            strength=1.0,
            max_delta=100,
        )
        center = corrected.getpixel((48, 36))
        # Correction follows the true blue background, not the excluded red
        # old subject or an artificial fill plate.
        self.assertGreater(center[2], center[0])
        self.assertGreater(center[1], 120)

    def test_fast_large_binary_expansion_matches_expected_square(self):
        mask = Image.new("L", (80, 70), 0)
        ImageDraw.Draw(mask).rectangle((30, 25, 39, 34), fill=255)
        expanded = expand_mask(mask, 12)
        self.assertEqual(expanded.getbbox(), (18, 13, 52, 47))
        self.assertEqual(expanded.getpixel((18, 13)), 255)
        self.assertEqual(expanded.getpixel((17, 13)), 0)

    def test_green_seam_cleanup_removes_line_outside_core_without_touching_center(self):
        source = Image.new("RGB", (96, 72), (105, 88, 70))
        generated = Image.new("RGB", source.size, (108, 91, 73))
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).rectangle((28, 18, 67, 53), fill=255)
        ImageDraw.Draw(generated).line((24, 55, 71, 55), fill=(5, 235, 15), width=2)
        ImageDraw.Draw(generated).rectangle((44, 30, 51, 39), fill=(20, 210, 40))

        cleaned, info = despill_green_seam_band(
            source,
            generated,
            mask,
            width=20,
        )

        self.assertTrue(info["applied"])
        self.assertLess(cleaned.getpixel((40, 55))[1], generated.getpixel((40, 55))[1])
        self.assertEqual(cleaned.getpixel((47, 34)), generated.getpixel((47, 34)))

    def test_seam_cleanup_removes_thin_magenta_complement_inside_core(self):
        source = Image.new("RGB", (96, 72), (105, 88, 70))
        generated = Image.new("RGB", source.size, (108, 91, 73))
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).rectangle((28, 18, 67, 53), fill=255)
        ImageDraw.Draw(generated).line((28, 20, 28, 51), fill=(210, 25, 205), width=2)
        ImageDraw.Draw(generated).rectangle((44, 30, 51, 39), fill=(205, 30, 200))

        cleaned, info = despill_green_seam_band(source, generated, mask, width=20)

        self.assertTrue(info["applied"])
        before = generated.getpixel((28, 34))
        after = cleaned.getpixel((28, 34))
        self.assertGreater(after[1], before[1])
        self.assertLessEqual(abs(sum(after) - sum(before)), 2)
        self.assertEqual(cleaned.getpixel((47, 34)), generated.getpixel((47, 34)))

    def test_outward_feather_keeps_core_sharp_and_softens_only_outer_ring(self):
        source = Image.new("RGB", (80, 60), (20, 40, 180))
        generated = Image.new("RGB", source.size, (230, 50, 30))
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).rectangle((24, 18, 55, 43), fill=255)

        result = composite_roi_with_outward_feather(
            source,
            generated,
            mask,
            (0, 0, 80, 60),
            expand_pixels=12,
            feather_pixels=8,
            hard_protect_pixels=4,
        )

        self.assertEqual(result.getpixel((30, 30)), generated.getpixel((30, 30)))
        self.assertEqual(result.getpixel((5, 5)), source.getpixel((5, 5)))
        ring_pixel = result.getpixel((16, 30))
        self.assertNotEqual(ring_pixel, source.getpixel((16, 30)))
        self.assertNotEqual(ring_pixel, generated.getpixel((16, 30)))

    def test_user_mask_bbox_is_drawn_on_generated_image(self):
        generated = Image.new("RGB", (80, 60), (40, 100, 180))
        mask = Image.new("L", generated.size, 0)
        ImageDraw.Draw(mask).ellipse((12, 8, 55, 47), fill=255)
        preview = draw_user_mask_bbox(generated, mask)
        self.assertEqual(preview.getpixel((12, 8)), (255, 215, 0))
        self.assertEqual(preview.getpixel((56, 48)), (40, 100, 180))

    def test_user_mask_bbox_can_expand_each_side(self):
        generated = Image.new("RGB", (80, 60), (40, 100, 180))
        mask = Image.new("L", generated.size, 0)
        ImageDraw.Draw(mask).rectangle((20, 15, 39, 34), fill=255)
        preview = draw_user_mask_bbox(
            generated,
            mask,
            expand_left=8,
            expand_right=12,
            expand_top=4,
            expand_bottom=10,
        )
        self.assertEqual(preview.getpixel((12, 11)), (255, 215, 0))
        self.assertEqual(preview.getpixel((51, 44)), (255, 215, 0))

    def test_context_bbox_includes_manual_expansion(self):
        mask = Image.new("L", (100, 80), 0)
        ImageDraw.Draw(mask).rectangle((40, 30, 59, 49), fill=255)
        self.assertEqual(
            mask_bbox_with_context(
                mask,
                0,
                0,
                expand_left=10,
                expand_right=20,
                expand_top=5,
                expand_bottom=15,
            ),
            (30, 25, 80, 65),
        )

    def test_binary_mask_and_context_bbox(self):
        mask = Image.new("L", (100, 80), 0)
        ImageDraw.Draw(mask).rectangle((40, 30, 59, 49), fill=200)
        binary = normalize_mask(mask, mask.size)
        self.assertEqual([i for i, count in enumerate(binary.histogram()) if count], [0, 255])
        self.assertEqual(mask_bbox_with_context(binary, 0.5, 0), (30, 20, 70, 60))

    def test_composite_preserves_pixels_outside_mask(self):
        source = Image.new("RGB", (40, 30), (10, 20, 30))
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).rectangle((12, 8, 21, 17), fill=255)
        generated = Image.new("RGB", (25, 22), (200, 100, 50))
        result = composite_roi(source, generated, mask, (5, 3, 30, 25), blend_expand=0)
        self.assertEqual(result.getpixel((11, 8)), source.getpixel((11, 8)))
        self.assertEqual(result.getpixel((22, 17)), source.getpixel((22, 17)))
        self.assertNotEqual(result.getpixel((16, 12)), source.getpixel((16, 12)))

    def test_reference_is_cropped_on_white(self):
        image = Image.new("RGB", (100, 100), (0, 120, 255))
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle((30, 20, 69, 79), fill=255)
        cutout = crop_reference_cutout(image, mask, padding_ratio=0.1, feather_radius=0)
        self.assertLess(cutout.width, image.width)
        self.assertEqual(cutout.getpixel((0, 0)), (255, 255, 255))

    def test_optional_prompt(self):
        default_prompt = flux_inpaint.build_prompt("")
        self.assertIn("reference image", default_prompt)
        self.assertIn("apparent size", default_prompt)
        self.assertIn("pose", default_prompt)
        self.assertIn("action", default_prompt)
        self.assertIn("朝向左侧", flux_inpaint.build_prompt("朝向左侧"))
        self.assertNotIn("green", flux_inpaint.build_prompt("", use_outpaint_lora=True).lower())
        self.assertIn("Replace the original object", default_prompt)
    def test_green_screen_only_replaces_mask(self):
        image = Image.new("RGB", (12, 8), (10, 20, 30))
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).rectangle((3, 2, 7, 5), fill=255)
        result = paint_mask_green(image, mask)
        self.assertEqual(result.getpixel((4, 3)), (0, 255, 0))
        self.assertEqual(result.getpixel((2, 3)), (10, 20, 30))

    def test_lora_scale_activates_named_adapter(self):
        class FakePipeline:
            call = None

            def set_adapters(self, names, adapter_weights=None):
                self.call = (names, adapter_weights)

        pipe = FakePipeline()
        flux_inpaint._set_lora_scale(pipe, 1.1)
        self.assertEqual(pipe.call, ([flux_inpaint.OUTPAINT_LORA_ADAPTER_NAME], [1.1]))

    def test_boundary_color_harmonization_reduces_exposure_error(self):
        source = Image.new("RGB", (120, 90), (90, 165, 240))
        generated = Image.new("RGB", source.size, (42, 92, 155))
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).rectangle((25, 20, 94, 69), fill=255)
        corrected, info = harmonize_roi_boundary_colors(
            source,
            generated,
            mask,
            ring_width=8,
            strength=1.0,
            fade_radius=24,
            interior_strength=0.25,
        )
        self.assertTrue(info["applied"])
        self.assertLess(info["edge_error_after"], info["edge_error_before"])
        self.assertNotEqual(corrected.getpixel((25, 40)), generated.getpixel((25, 40)))

    def test_harmonization_never_changes_pixels_outside_mask(self):
        source = Image.new("RGB", (80, 60), (120, 180, 230))
        generated = Image.new("RGB", source.size, (50, 90, 130))
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).ellipse((20, 12, 59, 51), fill=255)
        corrected, _ = harmonize_roi_boundary_colors(source, generated, mask, ring_width=6)
        self.assertEqual(corrected.getpixel((5, 5)), generated.getpixel((5, 5)))
        self.assertEqual(corrected.getpixel((70, 50)), generated.getpixel((70, 50)))

    def test_spatial_field_keeps_sky_and_wall_corrections_local(self):
        size = (220, 180)
        source = Image.new("RGB", size, (88, 160, 235))
        ImageDraw.Draw(source).rectangle((0, 90, 219, 179), fill=(155, 150, 145))
        generated = Image.new("RGB", size, (62, 118, 190))
        ImageDraw.Draw(generated).rectangle((0, 90, 219, 179), fill=(103, 112, 126))
        mask = Image.new("L", size, 0)
        ImageDraw.Draw(mask).rectangle((35, 22, 184, 157), fill=255)

        corrected, info = harmonize_roi_boundary_colors(
            source,
            generated,
            mask,
            ring_width=12,
            strength=1.0,
            fade_radius=48,
            interior_strength=0.3,
        )

        self.assertEqual(info["sample_strategy"], "spatial_outer_boundary_field")
        for point in ((36, 55), (36, 125)):
            before = sum(abs(a - b) for a, b in zip(source.getpixel(point), generated.getpixel(point)))
            after = sum(abs(a - b) for a, b in zip(source.getpixel(point), corrected.getpixel(point)))
            self.assertLess(after, before)
        self.assertNotEqual(
            tuple(a - b for a, b in zip(corrected.getpixel((36, 55)), generated.getpixel((36, 55)))),
            tuple(a - b for a, b in zip(corrected.getpixel((36, 125)), generated.getpixel((36, 125)))),
        )

    def test_green_despill_is_limited_to_inner_boundary(self):
        source = Image.new("RGB", (100, 80), (80, 150, 225))
        generated = Image.new("RGB", source.size, (80, 150, 225))
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).rectangle((20, 15, 79, 64), fill=255)
        ImageDraw.Draw(generated).line((20, 15, 20, 64), fill=(0, 255, 0), width=3)
        ImageDraw.Draw(generated).ellipse((45, 30, 55, 40), fill=(0, 220, 40))

        corrected, info = despill_green_boundary(source, generated, mask, width=8)

        self.assertTrue(info["applied"])
        before_excess = generated.getpixel((20, 35))[1] - max(
            generated.getpixel((20, 35))[0], generated.getpixel((20, 35))[2]
        )
        after_excess = corrected.getpixel((20, 35))[1] - max(
            corrected.getpixel((20, 35))[0], corrected.getpixel((20, 35))[2]
        )
        self.assertLess(after_excess, before_excess)
        self.assertEqual(corrected.getpixel((50, 35)), generated.getpixel((50, 35)))
        self.assertEqual(corrected.getpixel((5, 5)), generated.getpixel((5, 5)))

    def test_multiband_texture_blend_reduces_cloud_boundary_jump(self):
        width, height = 256, 128
        source = Image.new("RGB", (width, height))
        generated = Image.new("RGB", (width, height))
        source_pixels = source.load()
        generated_pixels = generated.load()
        for y in range(height):
            for x in range(width):
                value = int(round(170 + 25 * np.sin(x / 12) + 12 * np.sin(y / 8)))
                shifted = int(round(170 + 25 * np.sin((x - 4) / 12) + 12 * np.sin(y / 8)))
                source_pixels[x, y] = (value, min(255, value + 8), min(255, value + 18))
                generated_pixels[x, y] = (shifted, min(255, shifted + 8), min(255, shifted + 18))
        mask = Image.new("L", (width, height), 0)
        ImageDraw.Draw(mask).rectangle((64, 16, 191, 111), fill=255)

        simple = composite_roi(
            source,
            generated,
            mask,
            (0, 0, width, height),
            texture_blend_enabled=False,
        )
        multiband = composite_roi(
            source,
            generated,
            mask,
            (0, 0, width, height),
            texture_blend_enabled=True,
            texture_blend_levels=4,
        )
        simple_array = np.asarray(simple, dtype=np.float32)
        multiband_array = np.asarray(multiband, dtype=np.float32)
        source_array = np.asarray(source, dtype=np.float32)
        simple_jump = float(np.mean(np.abs(simple_array[20:108, 64] - simple_array[20:108, 63])))
        multiband_jump = float(np.mean(np.abs(multiband_array[20:108, 64] - multiband_array[20:108, 63])))
        self.assertLess(multiband_jump, simple_jump)
        self.assertEqual(float(np.max(np.abs(multiband_array[:, :64] - source_array[:, :64]))), 0.0)

    def test_similar_background_is_restored_without_edge_feather(self):
        source = Image.new("RGB", (120, 80), (100, 150, 200))
        generated = Image.new("RGB", source.size, (108, 157, 207))
        ImageDraw.Draw(generated).rectangle((48, 30, 71, 53), fill=(230, 40, 30))
        mask = Image.new("L", source.size, 0)
        ImageDraw.Draw(mask).rectangle((30, 20, 89, 59), fill=255)

        restored = composite_roi(
            source,
            generated,
            mask,
            (0, 0, 120, 80),
            texture_blend_enabled=True,
            texture_blend_levels=3,
            background_similarity_threshold=24,
            background_restore_strength=1.0,
        )

        self.assertEqual(restored.getpixel((29, 40)), source.getpixel((29, 40)))
        source_distance = sum(
            abs(a - b) for a, b in zip(restored.getpixel((35, 40)), source.getpixel((35, 40)))
        )
        generated_distance = sum(
            abs(a - b) for a, b in zip(generated.getpixel((35, 40)), source.getpixel((35, 40)))
        )
        self.assertLess(source_distance, generated_distance)
        self.assertGreater(restored.getpixel((60, 40))[0], 180)


if __name__ == "__main__":
    unittest.main()
