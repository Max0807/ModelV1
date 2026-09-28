"""Train-only, label-preserving appearance augmentation for image crops."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor
from torch.nn import functional as F


EYE_AUGMENTATION_COORDINATE_POLICY = "normalized_eye_crop_appearance_only_v1"
EYE_AUGMENTATION_POLICY_NONE = "none"
EYE_AUGMENTATION_POLICY_APPEARANCE_MILD_V1 = "appearance_mild_v1"
EYE_AUGMENTATION_POLICIES = (
    EYE_AUGMENTATION_POLICY_NONE,
    EYE_AUGMENTATION_POLICY_APPEARANCE_MILD_V1,
)
FACE_AUGMENTATION_COORDINATE_POLICY = "face_crop_appearance_only_v1"
FACE_AUGMENTATION_POLICY_NONE = "none"
FACE_AUGMENTATION_POLICY_APPEARANCE_MILD_V1 = "appearance_mild_v1"
FACE_AUGMENTATION_POLICIES = (
    FACE_AUGMENTATION_POLICY_NONE,
    FACE_AUGMENTATION_POLICY_APPEARANCE_MILD_V1,
)


@dataclass(frozen=True)
class EyeAppearanceAugmentationConfig:
    """Ranges and probabilities for safe eye-image appearance perturbations."""

    photometric_probability: float
    brightness_min: float
    brightness_max: float
    contrast_min: float
    contrast_max: float
    gamma_min: float
    gamma_max: float
    blur_probability: float
    blur_kernel_size: int
    blur_sigma_min: float
    blur_sigma_max: float
    noise_probability: float
    noise_std_max: float
    occlusion_probability: float
    occlusion_area_min: float
    occlusion_area_max: float
    occlusion_aspect_min: float
    occlusion_aspect_max: float
    coordinate_policy: str = EYE_AUGMENTATION_COORDINATE_POLICY

    def __post_init__(self) -> None:
        for name in (
            "photometric_probability",
            "blur_probability",
            "noise_probability",
            "occlusion_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}.")
        for lower_name, upper_name in (
            ("brightness_min", "brightness_max"),
            ("contrast_min", "contrast_max"),
            ("gamma_min", "gamma_max"),
            ("blur_sigma_min", "blur_sigma_max"),
            ("occlusion_area_min", "occlusion_area_max"),
            ("occlusion_aspect_min", "occlusion_aspect_max"),
        ):
            lower = float(getattr(self, lower_name))
            upper = float(getattr(self, upper_name))
            if lower <= 0 or upper < lower:
                raise ValueError(
                    f"Expected 0 < {lower_name} <= {upper_name}, got "
                    f"{lower} and {upper}."
                )
        if self.blur_kernel_size <= 0 or self.blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be a positive odd integer.")
        if self.noise_std_max < 0:
            raise ValueError("noise_std_max must be non-negative.")
        if self.occlusion_area_max > 1:
            raise ValueError("occlusion_area_max must not exceed 1.")
        if self.coordinate_policy != EYE_AUGMENTATION_COORDINATE_POLICY:
            raise ValueError(
                "Eye augmentation must remain appearance-only when crop-normalized "
                "pseudo labels are enabled; expected coordinate_policy="
                f"{EYE_AUGMENTATION_COORDINATE_POLICY!r}."
            )


@dataclass(frozen=True)
class FaceAppearanceAugmentationConfig:
    """Ranges and probabilities for geometry-preserving face perturbations."""

    photometric_probability: float
    brightness_min: float
    brightness_max: float
    contrast_min: float
    contrast_max: float
    gamma_min: float
    gamma_max: float
    blur_probability: float
    blur_kernel_size: int
    blur_sigma_min: float
    blur_sigma_max: float
    noise_probability: float
    noise_std_max: float
    coordinate_policy: str = FACE_AUGMENTATION_COORDINATE_POLICY

    def __post_init__(self) -> None:
        for name in (
            "photometric_probability",
            "blur_probability",
            "noise_probability",
        ):
            value = float(getattr(self, name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1], got {value}.")
        for lower_name, upper_name in (
            ("brightness_min", "brightness_max"),
            ("contrast_min", "contrast_max"),
            ("gamma_min", "gamma_max"),
            ("blur_sigma_min", "blur_sigma_max"),
        ):
            lower = float(getattr(self, lower_name))
            upper = float(getattr(self, upper_name))
            if lower <= 0 or upper < lower:
                raise ValueError(
                    f"Expected 0 < {lower_name} <= {upper_name}, got "
                    f"{lower} and {upper}."
                )
        if self.blur_kernel_size <= 0 or self.blur_kernel_size % 2 == 0:
            raise ValueError("blur_kernel_size must be a positive odd integer.")
        if self.noise_std_max < 0:
            raise ValueError("noise_std_max must be non-negative.")
        if self.coordinate_policy != FACE_AUGMENTATION_COORDINATE_POLICY:
            raise ValueError(
                "Face augmentation must remain appearance-only; expected "
                "coordinate_policy="
                f"{FACE_AUGMENTATION_COORDINATE_POLICY!r}."
            )


def eye_augmentation_config_for_policy(
    policy: str,
) -> EyeAppearanceAugmentationConfig | None:
    """Resolve one stable named policy into its fixed augmentation contract."""

    name = str(policy).strip().lower()
    if name == EYE_AUGMENTATION_POLICY_NONE:
        return None
    if name != EYE_AUGMENTATION_POLICY_APPEARANCE_MILD_V1:
        raise ValueError(
            f"Unknown eye augmentation policy {policy!r}; expected one of "
            f"{EYE_AUGMENTATION_POLICIES}."
        )
    return EyeAppearanceAugmentationConfig(
        photometric_probability=0.8,
        brightness_min=0.85,
        brightness_max=1.15,
        contrast_min=0.85,
        contrast_max=1.15,
        gamma_min=0.90,
        gamma_max=1.10,
        blur_probability=0.15,
        blur_kernel_size=3,
        blur_sigma_min=0.1,
        blur_sigma_max=0.8,
        noise_probability=0.25,
        noise_std_max=0.02,
        occlusion_probability=0.15,
        occlusion_area_min=0.01,
        occlusion_area_max=0.05,
        occlusion_aspect_min=0.5,
        occlusion_aspect_max=2.0,
    )


def face_augmentation_config_for_policy(
    policy: str,
) -> FaceAppearanceAugmentationConfig | None:
    """Resolve the fixed, geometry-preserving face augmentation policy."""

    name = str(policy).strip().lower()
    if name == FACE_AUGMENTATION_POLICY_NONE:
        return None
    if name != FACE_AUGMENTATION_POLICY_APPEARANCE_MILD_V1:
        raise ValueError(
            f"Unknown face augmentation policy {policy!r}; expected one of "
            f"{FACE_AUGMENTATION_POLICIES}."
        )
    return FaceAppearanceAugmentationConfig(
        photometric_probability=0.8,
        brightness_min=0.90,
        brightness_max=1.10,
        contrast_min=0.90,
        contrast_max=1.10,
        gamma_min=0.95,
        gamma_max=1.05,
        blur_probability=0.10,
        blur_kernel_size=3,
        blur_sigma_min=0.1,
        blur_sigma_max=0.6,
        noise_probability=0.20,
        noise_std_max=0.015,
    )


class PairedEyeAppearanceAugmentation:
    """Apply shared scene changes and independent local corruptions to two eyes.

    Inputs and outputs are RGB ``CHW`` tensors in ``[0, 1]``. Brightness,
    contrast, gamma, and blur use identical sampled parameters for the two eye
    crops. Sensor noise and small mean-filled occlusions are sampled
    independently, as requested for local sensor/visibility variation.
    """

    coordinate_policy = EYE_AUGMENTATION_COORDINATE_POLICY
    preserves_keypoint_coordinates = True

    def __init__(self, config: EyeAppearanceAugmentationConfig) -> None:
        self.config = config

    def __call__(self, left: Tensor, right: Tensor) -> tuple[Tensor, Tensor]:
        left, right, _, _ = self._apply_pair(left, right)
        return left, right

    def apply_with_keypoints(
        self,
        left: Tensor,
        right: Tensor,
        keypoints_xy: Tensor,
        keypoint_valid_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Augment images and invalidate keypoints hidden by random occlusion.

        ``keypoints_xy`` is ``[2, K, 2]`` in normalized eye-crop coordinates,
        ordered anatomical left/right. Appearance operations never change these
        coordinates. Mean-filled occlusion changes only the returned visibility
        mask, using the exact sampled rectangle applied to each eye image.
        """

        points, valid = _validate_keypoints(keypoints_xy, keypoint_valid_mask)
        left, right, left_box, right_box = self._apply_pair(left, right)
        updated_valid = valid.clone()
        for eye_index, (image, box) in enumerate(
            ((left, left_box), (right, right_box))
        ):
            if box is None:
                continue
            hidden = _keypoints_inside_occlusion(
                points[eye_index],
                image_height=int(image.shape[-2]),
                image_width=int(image.shape[-1]),
                box=box,
            )
            updated_valid[eye_index, hidden] = 0.0
        return left, right, points.clone(), updated_valid

    def _apply_pair(
        self,
        left: Tensor,
        right: Tensor,
    ) -> tuple[
        Tensor,
        Tensor,
        tuple[int, int, int, int] | None,
        tuple[int, int, int, int] | None,
    ]:
        left = _validate_eye(left, "left")
        right = _validate_eye(right, "right")
        if left.shape != right.shape:
            raise ValueError(
                "Paired eye crops must have the same shape, got "
                f"{tuple(left.shape)} and {tuple(right.shape)}."
            )

        if _bernoulli(self.config.photometric_probability):
            brightness = _uniform(
                self.config.brightness_min,
                self.config.brightness_max,
            )
            contrast = _uniform(
                self.config.contrast_min,
                self.config.contrast_max,
            )
            gamma = _uniform(self.config.gamma_min, self.config.gamma_max)
            left = _photometric(left, brightness, contrast, gamma)
            right = _photometric(right, brightness, contrast, gamma)

        if _bernoulli(self.config.blur_probability):
            sigma = _uniform(
                self.config.blur_sigma_min,
                self.config.blur_sigma_max,
            )
            left = _gaussian_blur(left, self.config.blur_kernel_size, sigma)
            right = _gaussian_blur(right, self.config.blur_kernel_size, sigma)

        left, left_box = self._independent_local_corruptions(left)
        right, right_box = self._independent_local_corruptions(right)
        return (
            left.clamp(0.0, 1.0),
            right.clamp(0.0, 1.0),
            left_box,
            right_box,
        )

    def _independent_local_corruptions(
        self,
        image: Tensor,
    ) -> tuple[Tensor, tuple[int, int, int, int] | None]:
        if _bernoulli(self.config.noise_probability):
            std = _uniform(0.0, self.config.noise_std_max)
            image = image + torch.randn_like(image) * std
        image = image.clamp(0.0, 1.0)
        occlusion_box = None
        if _bernoulli(self.config.occlusion_probability):
            image, occlusion_box = _mean_occlusion(
                image,
                area_min=self.config.occlusion_area_min,
                area_max=self.config.occlusion_area_max,
                aspect_min=self.config.occlusion_aspect_min,
                aspect_max=self.config.occlusion_aspect_max,
            )
        return image, occlusion_box


class FaceAppearanceAugmentation:
    """Apply mild appearance-only perturbations to one RGB face crop."""

    coordinate_policy = FACE_AUGMENTATION_COORDINATE_POLICY
    preserves_spatial_coordinates = True

    def __init__(self, config: FaceAppearanceAugmentationConfig) -> None:
        self.config = config

    def __call__(self, image: Tensor) -> Tensor:
        image = _validate_rgb_image(image, "face")
        if _bernoulli(self.config.photometric_probability):
            image = _photometric(
                image,
                _uniform(self.config.brightness_min, self.config.brightness_max),
                _uniform(self.config.contrast_min, self.config.contrast_max),
                _uniform(self.config.gamma_min, self.config.gamma_max),
            )
        if _bernoulli(self.config.blur_probability):
            image = _gaussian_blur(
                image,
                self.config.blur_kernel_size,
                _uniform(self.config.blur_sigma_min, self.config.blur_sigma_max),
            )
        if _bernoulli(self.config.noise_probability):
            noise_std = _uniform(0.0, self.config.noise_std_max)
            image = image + torch.randn_like(image) * noise_std
        return image.clamp(0.0, 1.0)


def _validate_eye(image: Tensor, name: str) -> Tensor:
    return _validate_rgb_image(image, f"{name} eye")


def _validate_rgb_image(image: Tensor, name: str) -> Tensor:
    image = torch.as_tensor(image, dtype=torch.float32)
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(f"{name} must be RGB CHW, got {tuple(image.shape)}.")
    if not torch.isfinite(image).all():
        raise ValueError(f"{name} contains non-finite values.")
    if torch.any(image < 0) or torch.any(image > 1):
        raise ValueError(f"{name} must lie in [0, 1] before augmentation.")
    return image


def _validate_keypoints(
    keypoints_xy: Tensor,
    keypoint_valid_mask: Tensor,
) -> tuple[Tensor, Tensor]:
    points = torch.as_tensor(keypoints_xy, dtype=torch.float32)
    valid = torch.as_tensor(keypoint_valid_mask, dtype=torch.float32)
    if points.ndim != 3 or points.shape[0] != 2 or points.shape[-1] != 2:
        raise ValueError(
            "keypoints_xy must have shape [2, K, 2] in anatomical left/right order, "
            f"got {tuple(points.shape)}."
        )
    if valid.shape != points.shape[:2]:
        raise ValueError(
            "keypoint_valid_mask must match keypoints_xy[:2], got "
            f"{tuple(valid.shape)} and {tuple(points.shape)}."
        )
    if not torch.isfinite(points).all() or not torch.isfinite(valid).all():
        raise ValueError("Keypoint inputs must contain only finite values.")
    if torch.any(valid < 0.0) or torch.any(valid > 1.0):
        raise ValueError("keypoint_valid_mask must lie in [0, 1].")
    return points, valid


def _keypoints_inside_occlusion(
    points_xy: Tensor,
    *,
    image_height: int,
    image_width: int,
    box: tuple[int, int, int, int],
) -> Tensor:
    top, left, erase_height, erase_width = box
    x_pixels = points_xy[:, 0] * image_width
    y_pixels = points_xy[:, 1] * image_height
    return (
        (x_pixels >= left)
        & (x_pixels < left + erase_width)
        & (y_pixels >= top)
        & (y_pixels < top + erase_height)
    )


def _bernoulli(probability: float) -> bool:
    return bool(torch.rand(()) < float(probability))


def _uniform(lower: float, upper: float) -> float:
    if upper == lower:
        return float(lower)
    return float(torch.empty(()).uniform_(float(lower), float(upper)))


def _photometric(
    image: Tensor,
    brightness: float,
    contrast: float,
    gamma: float,
) -> Tensor:
    image = image * brightness
    channel_mean = image.mean(dim=(-2, -1), keepdim=True)
    image = (image - channel_mean) * contrast + channel_mean
    return image.clamp(0.0, 1.0).pow(gamma)


def _gaussian_blur(image: Tensor, kernel_size: int, sigma: float) -> Tensor:
    radius = kernel_size // 2
    coordinate = torch.arange(
        -radius,
        radius + 1,
        device=image.device,
        dtype=image.dtype,
    )
    kernel_1d = torch.exp(-0.5 * (coordinate / sigma).square())
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    kernel = kernel_2d.expand(image.shape[0], 1, kernel_size, kernel_size)
    padded = F.pad(
        image.unsqueeze(0),
        (radius, radius, radius, radius),
        mode="reflect",
    )
    blurred = F.conv2d(
        padded,
        kernel,
        groups=image.shape[0],
    )
    return blurred.squeeze(0)


def _mean_occlusion(
    image: Tensor,
    *,
    area_min: float,
    area_max: float,
    aspect_min: float,
    aspect_max: float,
) -> tuple[Tensor, tuple[int, int, int, int]]:
    height, width = image.shape[-2:]
    area = _uniform(area_min, area_max) * height * width
    log_aspect = _uniform(math.log(aspect_min), math.log(aspect_max))
    aspect = math.exp(log_aspect)
    erase_width = max(1, min(width, int(round(math.sqrt(area * aspect)))))
    erase_height = max(1, min(height, int(round(math.sqrt(area / aspect)))))
    max_top = height - erase_height
    max_left = width - erase_width
    top = int(torch.randint(max_top + 1, ()).item())
    left = int(torch.randint(max_left + 1, ()).item())
    output = image.clone()
    fill = image.mean(dim=(-2, -1), keepdim=True)
    output[:, top : top + erase_height, left : left + erase_width] = fill
    return output, (top, left, erase_height, erase_width)
