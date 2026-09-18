import tempfile
import unittest
from pathlib import Path

from PIL import Image

import ui_preview


class UiPreviewTests(unittest.TestCase):
    def test_preview_size_preserves_aspect_ratio(self):
        self.assertEqual(ui_preview.preview_size((4000, 2000), 1000), (1000, 500))
        self.assertEqual(ui_preview.preview_size((640, 480), 1000), (640, 480))

    def test_click_coordinates_map_back_to_full_resolution(self):
        point = ui_preview.display_point_to_full((999, 499), (4000, 2000), (1000, 500))
        self.assertAlmostEqual(point[0], 3999)
        self.assertAlmostEqual(point[1], 1999)

    def test_color_preview_is_compact_jpeg_and_mask_is_png(self):
        with tempfile.TemporaryDirectory() as directory:
            color_path = ui_preview.save_preview(
                Image.new("RGB", (3000, 1500), (30, 140, 220)),
                Path(directory) / "color",
                max_side=900,
            )
            mask_path = ui_preview.save_preview(
                Image.new("L", (3000, 1500), 255),
                Path(directory) / "mask",
                max_side=900,
                mask=True,
            )
            self.assertTrue(color_path.endswith(".jpg"))
            self.assertTrue(mask_path.endswith(".png"))
            with Image.open(color_path) as color:
                self.assertEqual(color.size, (900, 450))
            with Image.open(mask_path) as mask:
                self.assertEqual(mask.size, (900, 450))

    def test_full_resolution_display_keeps_original_dimensions(self):
        with tempfile.TemporaryDirectory() as directory:
            image = Image.new("RGB", (2400, 1600), (30, 140, 220))
            path = ui_preview.save_preview(
                image,
                Path(directory) / "final_preview",
                max_side=max(image.size),
            )
            with Image.open(path) as displayed:
                self.assertEqual(displayed.size, image.size)


if __name__ == "__main__":
    unittest.main()
