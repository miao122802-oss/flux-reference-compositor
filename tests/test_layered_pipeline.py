import unittest
from types import SimpleNamespace

from PIL import Image

import flux_inpaint


class FakeGenerator:
    created = []

    def __init__(self, device):
        self.device = device
        self.seed = None
        FakeGenerator.created.append(self)

    def manual_seed(self, seed):
        self.seed = seed
        return self


class FakeTorch:
    Generator = FakeGenerator


class FakePipeline:
    def __init__(self):
        self.calls = []

    def __call__(self, **kwargs):
        self.calls.append(kwargs)
        color = (20 * len(self.calls), 0, 0)
        return SimpleNamespace(images=[Image.new("RGB", (32, 32), color)])


class LayeredPipelineTests(unittest.TestCase):
    def setUp(self):
        FakeGenerator.created = []
        self.image = Image.new("RGB", (32, 32))
        self.mask = Image.new("L", (32, 32), 255)
        self.reference = Image.new("RGB", (16, 16), "white")

    def run_layers(self, mode):
        pipe = FakePipeline()
        result = flux_inpaint.run_layered_diffusion(
            pipe,
            FakeTorch,
            model_input=self.image,
            model_mask=self.mask,
            reference_model=self.reference,
            object_prompt="object prompt",
            edit_mode=mode,
            seed=77,
            num_inference_steps=4,
            guidance_scale=1.0,
            strength=1.0,
        )
        return pipe, result

    def test_insert_calls_flux_once(self):
        pipe, result = self.run_layers(flux_inpaint.EDIT_MODE_INSERT)
        self.assertEqual(len(pipe.calls), 1)
        self.assertIn("image_reference", pipe.calls[0])
        self.assertIsNone(result["background"])

    def test_replace_calls_flux_once_with_reference(self):
        pipe, result = self.run_layers(flux_inpaint.EDIT_MODE_REPLACE)
        self.assertEqual(len(pipe.calls), 1)
        self.assertIn("image_reference", pipe.calls[0])
        self.assertIsNone(result["background"])
        self.assertEqual(result["pass_count"], 1)

    def test_replace_uses_requested_seed(self):
        pipe, _result = self.run_layers(flux_inpaint.EDIT_MODE_REPLACE)
        self.assertEqual([call["generator"].seed for call in pipe.calls], [77])


if __name__ == "__main__":
    unittest.main()
