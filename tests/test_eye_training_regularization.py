from __future__ import annotations

import unittest

import torch

from modelv1.data.augmentation import (
    EYE_AUGMENTATION_COORDINATE_POLICY,
    FACE_AUGMENTATION_COORDINATE_POLICY,
    EyeAppearanceAugmentationConfig,
    FaceAppearanceAugmentation,
    FaceAppearanceAugmentationConfig,
    PairedEyeAppearanceAugmentation,
    face_augmentation_config_for_policy,
)
from modelv1.model import (
    EYE_FEATURE_MODE_MULTISCALE_FPN24,
    EYE_STEM_MODE_SMALL_3X3_S2_NO_MAXPOOL,
    ResNetEyeImageEncoder,
)
from modelv1.vertical_eye_geometry import ProbabilisticEyeLandmarkHead


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


def face_augmentation_config(
    **overrides: float | int | str,
) -> FaceAppearanceAugmentationConfig:
    values: dict[str, float | int | str] = {
        "photometric_probability": 1.0,
        "brightness_min": 0.9,
        "brightness_max": 0.9,
        "contrast_min": 1.1,
        "contrast_max": 1.1,
        "gamma_min": 1.0,
        "gamma_max": 1.0,
        "blur_probability": 0.0,
        "blur_kernel_size": 3,
        "blur_sigma_min": 0.1,
        "blur_sigma_max": 0.6,
        "noise_probability": 0.0,
        "noise_std_max": 0.0,
    }
    values.update(overrides)
    return FaceAppearanceAugmentationConfig(**values)  # type: ignore[arg-type]


class FaceAppearanceAugmentationTests(unittest.TestCase):
    def test_appearance_changes_preserve_face_shape_and_input(self) -> None:
        transform = FaceAppearanceAugmentation(face_augmentation_config())
        face = torch.linspace(0.0, 1.0, 3 * 160 * 160).reshape(3, 160, 160)
        original = face.clone()

        augmented = transform(face)

        self.assertEqual(tuple(augmented.shape), (3, 160, 160))
        torch.testing.assert_close(face, original)
        self.assertFalse(torch.equal(augmented, face))
        self.assertTrue(torch.isfinite(augmented).all())
        self.assertTrue(torch.all((augmented >= 0) & (augmented <= 1)))

    def test_named_policy_is_mild_and_contains_no_spatial_parameters(self) -> None:
        config = face_augmentation_config_for_policy("appearance_mild_v1")

        self.assertIsNotNone(config)
        assert config is not None
        self.assertEqual(
            config.coordinate_policy,
            FACE_AUGMENTATION_COORDINATE_POLICY,
        )
        self.assertFalse(
            any(
                token in config.__dataclass_fields__
                for token in ("rotation", "flip", "crop", "translation", "scale")
            )
        )

    def test_spatial_coordinate_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "appearance-only"):
            face_augmentation_config(coordinate_policy="rotate_or_flip")
        self.assertTrue(
            FaceAppearanceAugmentation.preserves_spatial_coordinates
        )


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

    def test_keypoint_coordinates_are_bitwise_unchanged(self) -> None:
        transform = PairedEyeAppearanceAugmentation(augmentation_config())
        eye = torch.linspace(0.0, 1.0, 3 * 36 * 60).reshape(3, 36, 60)
        points = torch.rand((2, 15, 2))
        valid = torch.ones((2, 15))

        _, _, augmented_points, augmented_valid = transform.apply_with_keypoints(
            eye,
            eye.clone(),
            points,
            valid,
        )

        self.assertTrue(torch.equal(augmented_points, points))
        self.assertTrue(torch.equal(augmented_valid, valid))

    def test_full_crop_occlusion_invalidates_visible_keypoints(self) -> None:
        transform = PairedEyeAppearanceAugmentation(
            augmentation_config(
                photometric_probability=0.0,
                blur_probability=0.0,
                occlusion_probability=1.0,
                occlusion_area_min=1.0,
                occlusion_area_max=1.0,
                occlusion_aspect_min=60.0 / 36.0,
                occlusion_aspect_max=60.0 / 36.0,
            )
        )
        eye = torch.full((3, 36, 60), 0.5)
        points = torch.full((2, 15, 2), 0.5)
        valid = torch.ones((2, 15))

        _, _, augmented_points, augmented_valid = transform.apply_with_keypoints(
            eye,
            eye.clone(),
            points,
            valid,
        )

        self.assertTrue(torch.equal(augmented_points, points))
        self.assertEqual(int(augmented_valid.count_nonzero()), 0)

    def test_spatial_coordinate_policy_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "appearance-only"):
            augmentation_config(
                coordinate_policy="translate_and_rotate",  # type: ignore[arg-type]
            )
        self.assertEqual(
            PairedEyeAppearanceAugmentation.coordinate_policy,
            EYE_AUGMENTATION_COORDINATE_POLICY,
        )


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


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    "CUDA BF16 is required",
)
class BFloat16InterpolationTests(unittest.TestCase):
    def test_fpn_and_landmark_decoder_support_bf16_autocast(self) -> None:
        encoder = ResNetEyeImageEncoder(
            backbone="resnet18",
            embedding_dim=None,
            dropout=0.0,
            weights=None,
            spatial_pool_size=(7, 12),
            feature_map_stage="layer4",
            feature_map_mode=EYE_FEATURE_MODE_MULTISCALE_FPN24,
            stem_mode=EYE_STEM_MODE_SMALL_3X3_S2_NO_MAXPOOL,
        ).cuda().eval()
        landmark_head = ProbabilisticEyeLandmarkHead(
            output_size=(14, 23),
        ).cuda().eval()
        image = torch.zeros((1, 3, 56, 90), device="cuda")

        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            maps = encoder.forward_feature_maps(image)
            landmarks = landmark_head(maps["layer2"], maps["layer3"])

        self.assertEqual(maps["feature_map"].dtype, torch.bfloat16)
        self.assertEqual(landmarks["logits"].dtype, torch.bfloat16)
        self.assertEqual(tuple(maps["feature_map"].shape[-2:]), (7, 12))
        self.assertEqual(tuple(landmarks["logits"].shape[-2:]), (14, 23))


if __name__ == "__main__":
    unittest.main()
