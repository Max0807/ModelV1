from __future__ import annotations

import unittest

import torch

from modelv1.data.augmentation import (
    EyeAppearanceAugmentationConfig,
    PairedEyeAppearanceAugmentation,
)
from modelv1.model import ResNetEyeImageEncoder


def augmentation_config(**overrides: float | int) -> EyeAppearanceAugmentationConfig:
    values: dict[str, float | int] = {
        "photometric_probability": 1.0,
        "brightness_min": 0.9,
        "brightness_max": 0.9,
        "contrast_min": 1.1,
        "contrast_max": 1.1,
        "gamma_min": 1.0,
        "gamma_max": 1.0,
        "blur_probability": 1.0,
        "blur_kernel_size": 3,
        "blur_sigma_min": 0.5,
        "blur_sigma_max": 0.5,
        "noise_probability": 0.0,
        "noise_std_max": 0.0,
        "occlusion_probability": 0.0,
        "occlusion_area_min": 0.01,
        "occlusion_area_max": 0.05,
        "occlusion_aspect_min": 0.5,
        "occlusion_aspect_max": 2.0,
    }
    values.update(overrides)
    return EyeAppearanceAugmentationConfig(**values)  # type: ignore[arg-type]


class PairedEyeAugmentationTests(unittest.TestCase):
    def test_shared_photometric_and_blur_parameters_preserve_pair_identity(self) -> None:
        transform = PairedEyeAppearanceAugmentation(augmentation_config())
        eye = torch.linspace(0.0, 1.0, 3 * 36 * 60).reshape(3, 36, 60)
        left, right = transform(eye, eye.clone())
        self.assertEqual(tuple(left.shape), (3, 36, 60))
        torch.testing.assert_close(left, right)
        self.assertGreaterEqual(float(left.min()), 0.0)
        self.assertLessEqual(float(left.max()), 1.0)
        self.assertFalse(torch.equal(left, eye))

    def test_noise_and_occlusion_are_finite_and_do_not_mutate_input(self) -> None:
        transform = PairedEyeAppearanceAugmentation(
            augmentation_config(
                photometric_probability=0.0,
                blur_probability=0.0,
                noise_probability=1.0,
                noise_std_max=0.02,
                occlusion_probability=1.0,
                occlusion_area_min=0.05,
                occlusion_area_max=0.05,
            )
        )
        eye = torch.full((3, 36, 60), 0.5)
        original = eye.clone()
        left, right = transform(eye, eye)
        torch.testing.assert_close(eye, original)
        self.assertTrue(torch.isfinite(left).all())
        self.assertTrue(torch.isfinite(right).all())
        self.assertTrue(torch.all((left >= 0) & (left <= 1)))
        self.assertTrue(torch.all((right >= 0) & (right <= 1)))


class EyeBackboneFreezeTests(unittest.TestCase):
    def test_full_freeze_then_layer4_unfreeze_keeps_frozen_batchnorm_eval(self) -> None:
        encoder = ResNetEyeImageEncoder(
            backbone="resnet18",
            embedding_dim=8,
            dropout=0.0,
            weights=None,
        )
        encoder.set_backbone_train_from(None)
        encoder.train()
        self.assertTrue(all(not p.requires_grad for p in encoder.backbone.parameters()))
        self.assertFalse(encoder.backbone.bn1.training)
        self.assertFalse(encoder.backbone.layer3[0].bn1.training)

        encoder.set_backbone_train_from("layer4")
        encoder.train()
        self.assertTrue(all(not p.requires_grad for p in encoder.backbone.layer3.parameters()))
        self.assertTrue(all(p.requires_grad for p in encoder.backbone.layer4.parameters()))
        self.assertFalse(encoder.backbone.layer3[0].bn1.training)
        self.assertTrue(encoder.backbone.layer4[0].bn1.training)


if __name__ == "__main__":
    unittest.main()
