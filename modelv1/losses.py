"""Loss and millimeter-space metrics for normalized ModelV1 UV predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data.normalization import UVTargetNormalizer, validate_uv_tensor
from .geometry import virtual_camera_xy_to_table_uv


@dataclass(frozen=True)
class UVLossConfig:
    """Configuration for the V1 primary UV regression objective."""

    beta_mm: float = 30.0

    def __post_init__(self) -> None:
        if self.beta_mm <= 0:
            raise ValueError("beta_mm must be positive.")


class UVRegressionLoss(nn.Module):
    """Per-axis Smooth L1 loss for normalized model outputs.

    The UV head and target are compared in the normalized space. The Huber
    transition is converted from ``beta_mm`` independently for u and v, so it
    still begins at the same physical coordinate error in each axis.
    """

    def __init__(
        self,
        normalizer: UVTargetNormalizer,
        config: UVLossConfig | None = None,
    ) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.config = config or UVLossConfig()

    def forward(self, uv_pred: Tensor, uv_target: Tensor) -> Tensor:
        uv_pred = validate_uv_tensor(uv_pred, "uv_pred")
        uv_target = validate_uv_tensor(uv_target, "uv_target").to(
            device=uv_pred.device, dtype=uv_pred.dtype
        )
        error = (uv_pred - uv_target).abs()
        beta = self.normalizer.normalized_beta(self.config.beta_mm, uv_pred)
        return torch.where(
            error < beta,
            0.5 * error.square() / beta,
            error - 0.5 * beta,
        ).mean()

    @torch.no_grad()
    def metrics(self, uv_pred: Tensor, uv_gt_mm: Tensor) -> dict[str, Tensor]:
        return compute_uv_metrics(uv_pred, uv_gt_mm, self.normalizer)


class DirectTableUVLoss(nn.Module):
    """Supervise a direct normalized table-UV prediction in physical mm.

    This path performs only the target z-score inverse transform. It does not
    reconstruct a 3D point or invoke any camera/table geometry after the model
    prediction.
    """

    def __init__(
        self,
        normalizer: UVTargetNormalizer,
        config: UVLossConfig | None = None,
    ) -> None:
        super().__init__()
        self.normalizer = normalizer
        self.config = config or UVLossConfig()

    def forward(
        self,
        uv_table_pred_normalized: Tensor | Mapping[str, Tensor],
        batch: Mapping[str, object],
    ) -> dict[str, Tensor]:
        uv_table_pred_normalized, delta_v_mm = _direct_uv_and_vertical_residual(
            uv_table_pred_normalized
        )
        uv_table_pred_normalized = validate_uv_tensor(
            uv_table_pred_normalized,
            "uv_table_pred_normalized",
        )
        device_type = uv_table_pred_normalized.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            uv_base_table_mm = self.normalizer.denormalize(
                uv_table_pred_normalized.float()
            )
            uv_pred_table_mm = _apply_table_v_residual(
                uv_base_table_mm,
                delta_v_mm,
            )
            uv_gt_table_mm = validate_uv_tensor(
                _required_batch_tensor(batch, "uv_gt_table_mm"),
                "uv_gt_table_mm",
            ).to(device=uv_pred_table_mm.device, dtype=torch.float32)
            loss = F.smooth_l1_loss(
                uv_pred_table_mm,
                uv_gt_table_mm,
                beta=self.config.beta_mm,
            )
        return {
            "loss": loss,
            "uv_pred_table_mm": uv_pred_table_mm,
            "uv_base_table_mm": uv_base_table_mm,
            "vertical_delta_v_mm": delta_v_mm,
        }


class DirectVirtualTableUVLoss(nn.Module):
    """Supervise predicted ``P_N[:2]`` after conversion to table-local UV.

    The network output remains normalized virtual-camera x/y. Loss and metrics,
    however, use the same physical table-UV coordinate system as the
    gaze-plus-geometry path.
    """

    def __init__(
        self,
        normalizer: UVTargetNormalizer,
        config: UVLossConfig | None = None,
        *,
        table_distance_scale_mm: float = 1000.0,
    ) -> None:
        super().__init__()
        if table_distance_scale_mm <= 0:
            raise ValueError("table_distance_scale_mm must be positive.")
        self.normalizer = normalizer
        self.config = config or UVLossConfig()
        self.table_distance_scale_mm = float(table_distance_scale_mm)

    def forward(
        self,
        point_xy_n_pred: Tensor | Mapping[str, Tensor],
        batch: Mapping[str, object],
    ) -> dict[str, Tensor]:
        point_xy_n_pred, delta_v_mm = _direct_uv_and_vertical_residual(
            point_xy_n_pred
        )
        point_xy_n_pred = validate_uv_tensor(
            point_xy_n_pred,
            "point_xy_n_pred",
        )
        device_type = point_xy_n_pred.device.type
        # Plane reconstruction and millimetre-space Huber are deliberately
        # float32 even when the learned image branches run under AMP.
        with torch.autocast(device_type=device_type, enabled=False):
            point_xy_n_mm = self.normalizer.denormalize(
                point_xy_n_pred.float()
            )
            table_frame7_n = _required_batch_tensor(
                batch,
                "table_frame7_n",
            ).to(device=point_xy_n_mm.device, dtype=torch.float32)
            uv_gt_table_mm = validate_uv_tensor(
                _required_batch_tensor(batch, "uv_gt_table_mm"),
                "uv_gt_table_mm",
            ).to(device=point_xy_n_mm.device, dtype=torch.float32)
            uv_base_table_mm = virtual_camera_xy_to_table_uv(
                point_xy_n_mm,
                table_frame7_n,
                distance_scale_mm=self.table_distance_scale_mm,
            )
            uv_pred_table_mm = _apply_table_v_residual(
                uv_base_table_mm,
                delta_v_mm,
            )
            loss = F.smooth_l1_loss(
                uv_pred_table_mm,
                uv_gt_table_mm,
                beta=self.config.beta_mm,
            )
        return {
            "loss": loss,
            "uv_pred_table_mm": uv_pred_table_mm,
            "uv_base_table_mm": uv_base_table_mm,
            "vertical_delta_v_mm": delta_v_mm,
            "point_xy_n_mm": point_xy_n_mm,
        }


def _direct_uv_and_vertical_residual(
    prediction: Tensor | Mapping[str, Tensor],
) -> tuple[Tensor, Tensor]:
    if isinstance(prediction, Mapping):
        uv_prediction = prediction.get("uv")
        if not torch.is_tensor(uv_prediction):
            raise TypeError("Direct-UV model mapping must contain tensor 'uv'.")
        delta_v_mm = prediction.get("vertical_delta_v_mm")
        if delta_v_mm is None:
            delta_v_mm = uv_prediction.new_zeros((uv_prediction.shape[0], 1))
        elif not torch.is_tensor(delta_v_mm):
            raise TypeError("vertical_delta_v_mm must be a tensor.")
    else:
        uv_prediction = prediction
        delta_v_mm = prediction.new_zeros((prediction.shape[0], 1))
    uv_prediction = validate_uv_tensor(uv_prediction, "direct_uv_prediction")
    delta_v_mm = delta_v_mm.to(
        device=uv_prediction.device,
        dtype=torch.float32,
    )
    if delta_v_mm.shape != (uv_prediction.shape[0], 1):
        raise ValueError("vertical_delta_v_mm must have shape [B, 1].")
    return uv_prediction, delta_v_mm


def _apply_table_v_residual(
    uv_base_table_mm: Tensor,
    delta_v_mm: Tensor,
) -> Tensor:
    """Apply a physical vertical correction while preserving u bit-for-bit."""

    return torch.stack(
        (
            uv_base_table_mm[:, 0],
            uv_base_table_mm[:, 1] + delta_v_mm[:, 0],
        ),
        dim=-1,
    )


@torch.no_grad()
def compute_uv_metrics(
    uv_pred: Tensor,
    uv_gt_mm: Tensor,
    normalizer: UVTargetNormalizer,
) -> dict[str, Tensor]:
    """Return physically interpretable validation metrics in millimeters."""

    uv_pred = validate_uv_tensor(uv_pred, "uv_pred").float()
    uv_gt_mm = validate_uv_tensor(uv_gt_mm, "uv_gt_mm").to(
        device=uv_pred.device, dtype=torch.float32
    )
    uv_pred_mm = normalizer.denormalize(uv_pred)
    error_mm = uv_pred_mm - uv_gt_mm
    abs_error_mm = error_mm.abs()
    epe_mm = torch.linalg.vector_norm(error_mm, dim=-1)
    return {
        "epe_mm": epe_mm.mean(),
        "median_epe_mm": epe_mm.median(),
        "mae_u_mm": abs_error_mm[..., 0].mean(),
        "mae_v_mm": abs_error_mm[..., 1].mean(),
    }


def batch_uv_targets(batch: Mapping[str, object]) -> tuple[Tensor, Tensor]:
    """Extract normalized and millimeter UV targets from one collated batch."""

    try:
        uv_target = batch["uv_target"]
        uv_gt = batch["uv_gt"]
    except KeyError as exc:
        raise KeyError("Batch must contain uv_target and uv_gt.") from exc
    if not torch.is_tensor(uv_target) or not torch.is_tensor(uv_gt):
        raise TypeError("uv_target and uv_gt must be torch tensors.")
    return validate_uv_tensor(uv_target, "uv_target"), validate_uv_tensor(uv_gt, "uv_gt")


def iris_heatmap_mse_loss(
    left_logits: Tensor,
    right_logits: Tensor,
    iris_center_xy: Tensor,
    iris_center_valid_mask: Tensor,
    *,
    sigma_pixels: float,
) -> Tensor:
    """Supervise one Gaussian iris-centre heatmap per eye.

    ``iris_center_xy`` is normalized to each eye crop and ordered left/right.
    Invalid MediaPipe observations are masked out rather than treated as a
    negative/background label.
    """

    if sigma_pixels <= 0:
        raise ValueError("sigma_pixels must be positive.")
    if left_logits.ndim != 4 or right_logits.ndim != 4:
        raise ValueError("Iris heatmap logits must have shape [B, 1, H, W].")
    if left_logits.shape != right_logits.shape or left_logits.shape[1] != 1:
        raise ValueError("Left/right iris logits must share shape [B, 1, H, W].")
    batch_size, _, height, width = left_logits.shape
    if iris_center_xy.shape != (batch_size, 2, 2):
        raise ValueError("iris_center_xy must have shape [B, 2, 2].")
    if iris_center_valid_mask.shape != (batch_size, 2):
        raise ValueError("iris_center_valid_mask must have shape [B, 2].")
    centres = iris_center_xy.to(device=left_logits.device, dtype=torch.float32)
    valid = iris_center_valid_mask.to(device=left_logits.device, dtype=torch.float32)
    if not torch.isfinite(centres).all() or not torch.isfinite(valid).all():
        raise ValueError("Iris auxiliary targets must be finite.")
    if torch.any(valid < 0) or torch.any(valid > 1):
        raise ValueError("iris_center_valid_mask must lie in [0, 1].")

    y_grid, x_grid = torch.meshgrid(
        torch.arange(height, device=left_logits.device, dtype=torch.float32),
        torch.arange(width, device=left_logits.device, dtype=torch.float32),
        indexing="ij",
    )
    logits = torch.cat((left_logits, right_logits), dim=1).float()
    centres_px = centres.clone()
    centres_px[..., 0] *= max(width - 1, 1)
    centres_px[..., 1] *= max(height - 1, 1)
    distance_sq = (
        (x_grid[None, None] - centres_px[..., 0, None, None]).square()
        + (y_grid[None, None] - centres_px[..., 1, None, None]).square()
    )
    targets = torch.exp(-0.5 * distance_sq / (sigma_pixels * sigma_pixels))
    per_eye_loss = F.mse_loss(torch.sigmoid(logits), targets, reduction="none").mean(
        dim=(-1, -2)
    )
    return (per_eye_loss * valid).sum() / valid.sum().clamp_min(1.0)


def probabilistic_eye_keypoint_nll(
    left_mean_xy: Tensor,
    right_mean_xy: Tensor,
    left_covariance: Tensor,
    right_covariance: Tensor,
    target_xy: Tensor,
    valid_mask: Tensor,
    eye_quality: Tensor,
    *,
    min_std_norm: float,
) -> dict[str, Tensor]:
    """Quality-weighted full-covariance NLL for binocular eye keypoints.

    Coordinates are normalized to each legacy eye crop. The covariance comes
    from the spatial probability maps' second moments. Invalid, low-quality,
    or augmentation-occluded pseudo-labels therefore cannot dominate the
    auxiliary objective.
    """

    if min_std_norm <= 0:
        raise ValueError("min_std_norm must be positive.")
    if left_mean_xy.ndim != 3 or left_mean_xy.shape[-1] != 2:
        raise ValueError("Keypoint means must have shape [B, K, 2].")
    if right_mean_xy.shape != left_mean_xy.shape:
        raise ValueError("Left/right keypoint means must share shape [B, K, 2].")
    batch_size, keypoint_count, _ = left_mean_xy.shape
    expected_covariance_shape = (batch_size, keypoint_count, 2, 2)
    if (
        left_covariance.shape != expected_covariance_shape
        or right_covariance.shape != expected_covariance_shape
    ):
        raise ValueError(
            "Left/right keypoint covariance must have shape "
            f"{expected_covariance_shape}."
        )
    if target_xy.shape != (batch_size, 2, keypoint_count, 2):
        raise ValueError(
            "target_xy must have shape "
            f"[{batch_size}, 2, {keypoint_count}, 2]."
        )
    if valid_mask.shape != (batch_size, 2, keypoint_count):
        raise ValueError(
            "valid_mask must have shape "
            f"[{batch_size}, 2, {keypoint_count}]."
        )
    if eye_quality.shape != (batch_size, 2):
        raise ValueError(f"eye_quality must have shape [{batch_size}, 2].")

    device = left_mean_xy.device
    means = torch.stack((left_mean_xy, right_mean_xy), dim=1).float()
    covariance = torch.stack(
        (left_covariance, right_covariance), dim=1
    ).float()
    targets = target_xy.to(device=device, dtype=torch.float32)
    valid = valid_mask.to(device=device, dtype=torch.float32)
    quality = eye_quality.to(device=device, dtype=torch.float32)
    if not all(
        torch.isfinite(value).all()
        for value in (means, covariance, targets, valid, quality)
    ):
        raise ValueError("Probabilistic keypoint inputs must be finite.")
    if torch.any(valid < 0) or torch.any(valid > 1):
        raise ValueError("Keypoint valid_mask must lie in [0, 1].")
    if torch.any(quality < 0) or torch.any(quality > 1):
        raise ValueError("Keypoint eye_quality must lie in [0, 1].")

    covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
    variance_floor = float(min_std_norm) ** 2
    variance_x = covariance[..., 0, 0].clamp_min(0.0) + variance_floor
    variance_y = covariance[..., 1, 1].clamp_min(0.0) + variance_floor
    covariance_xy = covariance[..., 0, 1]
    determinant_floor = variance_floor * variance_floor
    determinant = (
        variance_x * variance_y - covariance_xy.square()
    ).clamp_min(determinant_floor)
    delta = means - targets
    mahalanobis = (
        variance_y * delta[..., 0].square()
        + variance_x * delta[..., 1].square()
        - 2.0 * covariance_xy * delta[..., 0] * delta[..., 1]
    ) / determinant
    # Subtract the constant floor-density term. This keeps the auxiliary loss
    # non-negative while preserving the Gaussian NLL gradients.
    relative_log_determinant = torch.log(
        determinant / determinant_floor
    )
    per_keypoint_nll = 0.5 * (
        mahalanobis.clamp_min(0.0) + relative_log_determinant.clamp_min(0.0)
    )

    weights = valid * quality.unsqueeze(-1)
    denominator = weights.sum().clamp_min(1.0)
    loss = (per_keypoint_nll * weights).sum() / denominator
    error_norm = torch.linalg.vector_norm(delta, dim=-1)
    mean_error_norm = (error_norm * weights).sum() / denominator
    predicted_std_norm = torch.sqrt(
        0.5 * (variance_x + variance_y)
    )
    mean_std_norm = (predicted_std_norm * weights).sum() / denominator
    return {
        "loss": loss,
        "mean_error_norm": mean_error_norm,
        "mean_std_norm": mean_std_norm,
        "effective_weight": weights.sum(),
    }


@dataclass(frozen=True)
class GazeGeometryLossConfig:
    """Explicit V4 weights; all UV and ray distances are expressed in mm."""

    uv_huber_beta_mm: float
    uv_huber_weight: float
    mixture_nll_weight: float
    mixture_kernel_sigma_mm: float
    gaze_direction_weight: float
    gaze_angular_weight: float
    depth_prior_kl_weight: float
    ray_validity_weight: float
    ray_penalty_scale_mm: float
    depth_correction_prior_weight: float = 0.0
    uv_gaussian_nll_weight: float = 0.0
    uv_covariance_floor_mm2: float = 400.0

    def __post_init__(self) -> None:
        if self.uv_huber_beta_mm <= 0:
            raise ValueError("uv_huber_beta_mm must be positive.")
        if self.mixture_kernel_sigma_mm <= 0:
            raise ValueError("mixture_kernel_sigma_mm must be positive.")
        if self.ray_penalty_scale_mm <= 0:
            raise ValueError("ray_penalty_scale_mm must be positive.")
        if self.uv_covariance_floor_mm2 <= 0:
            raise ValueError("uv_covariance_floor_mm2 must be positive.")
        weights = (
            self.uv_huber_weight,
            self.mixture_nll_weight,
            self.gaze_direction_weight,
            self.gaze_angular_weight,
            self.depth_prior_kl_weight,
            self.ray_validity_weight,
            self.depth_correction_prior_weight,
            self.uv_gaussian_nll_weight,
        )
        if any(weight < 0 for weight in weights) or sum(weights) <= 0:
            raise ValueError("V4 loss weights must be non-negative and not all zero.")


def gaussian_uv_nll(
    uv_mean_mm: Tensor,
    uv_gt_mm: Tensor,
    uv_covariance_mm2: Tensor,
    *,
    covariance_floor_mm2: float,
) -> Tensor:
    """Compute a stable full-covariance 2D Gaussian NLL in float32."""

    if uv_covariance_mm2.shape != (uv_mean_mm.shape[0], 2, 2):
        raise ValueError("uv_covariance_mm2 must have shape [B, 2, 2].")
    device_type = uv_mean_mm.device.type
    with torch.autocast(device_type=device_type, enabled=False):
        mean = uv_mean_mm.float()
        target = uv_gt_mm.to(device=mean.device, dtype=torch.float32)
        covariance = uv_covariance_mm2.to(device=mean.device, dtype=torch.float32)
        covariance = 0.5 * (covariance + covariance.transpose(-1, -2))
        identity = torch.eye(2, device=mean.device, dtype=mean.dtype)
        covariance = covariance + covariance_floor_mm2 * identity
        cholesky, info = torch.linalg.cholesky_ex(covariance)
        if torch.any(info != 0):
            raise ValueError("UV covariance must be positive definite after flooring.")
        residual = (target - mean).unsqueeze(-1)
        mahalanobis = (
            residual.transpose(-1, -2)
            @ torch.cholesky_solve(residual, cholesky)
        ).squeeze(-1).squeeze(-1)
        log_determinant = 2.0 * torch.log(
            torch.diagonal(cholesky, dim1=-2, dim2=-1)
        ).sum(dim=-1)
        return 0.5 * (
            mahalanobis
            + log_determinant
            + 2.0 * torch.log(mean.new_tensor(2.0 * torch.pi))
        ).mean()


class GazeGeometryLoss(nn.Module):
    """End-to-end UV supervision through the parameter-free ray/table layer."""

    def __init__(self, config: GazeGeometryLossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self,
        output: Mapping[str, Tensor],
        batch: Mapping[str, object],
    ) -> dict[str, Tensor]:
        uv_gt = _required_batch_tensor(batch, "uv_gt")
        uv_mean = validate_uv_tensor(output["uv_mean_mm"], "uv_mean_mm")
        uv_gt = validate_uv_tensor(uv_gt, "uv_gt").to(
            device=uv_mean.device,
            dtype=uv_mean.dtype,
        )
        uv_huber = F.smooth_l1_loss(
            uv_mean,
            uv_gt,
            beta=self.config.uv_huber_beta_mm,
        )

        uv_hypotheses = output["uv_hypotheses_mm"]
        effective_weights = output["depth_effective_weights"]
        if self.config.mixture_nll_weight > 0:
            squared_error = (uv_hypotheses - uv_gt.unsqueeze(1)).square().sum(dim=-1)
            sigma2 = self.config.mixture_kernel_sigma_mm**2
            log_weights = torch.log(
                effective_weights.clamp_min(
                    torch.finfo(effective_weights.dtype).tiny
                )
            )
            component_log_likelihood = (
                log_weights
                - 0.5 * squared_error / sigma2
                - torch.log(uv_mean.new_tensor(2.0 * torch.pi * sigma2))
            )
            mixture_nll = -torch.logsumexp(
                component_log_likelihood,
                dim=-1,
            ).mean()
        else:
            mixture_nll = uv_mean.new_zeros(())

        if self.config.uv_gaussian_nll_weight > 0:
            covariance = output.get("uv_covariance_mm2")
            if covariance is None:
                raise KeyError(
                    "uv_gaussian_nll_weight requires output['uv_covariance_mm2']."
                )
            uv_gaussian_nll = gaussian_uv_nll(
                uv_mean,
                uv_gt,
                covariance,
                covariance_floor_mm2=self.config.uv_covariance_floor_mm2,
            )
        else:
            uv_gaussian_nll = uv_mean.new_zeros(())

        if (
            self.config.gaze_direction_weight > 0
            or self.config.gaze_angular_weight > 0
        ):
            raw_eyes = _required_batch_tensor(batch, "raw_eye_geometry_mm").to(
                device=uv_mean.device,
                dtype=uv_mean.dtype,
            )
            target_camera = _required_batch_tensor(
                batch,
                "gaze_target_camera_mm",
            ).to(device=uv_mean.device, dtype=uv_mean.dtype)
            eye_midpoint = 0.5 * (raw_eyes[:, :3] + raw_eyes[:, 3:])
            gaze_target_direction = F.normalize(
                target_camera - eye_midpoint,
                dim=-1,
                eps=1e-8,
            )
            gaze_direction = F.normalize(
                output["gaze_direction_c"],
                dim=-1,
                eps=1e-8,
            )
            gaze_direction_cosine = (
                gaze_direction * gaze_target_direction
            ).sum(dim=-1).clamp(-1.0, 1.0)
            # Sign-only constraint: require the prediction to lie in the same
            # hemisphere as eye->target GT, without forcing its noisy exact
            # angle when the PnP eye depth is imperfect.
            gaze_direction_loss = F.relu(-gaze_direction_cosine).mean()
            gaze_angular = (
                1.0
                - gaze_direction_cosine
            ).mean()
        else:
            gaze_direction_loss = uv_mean.new_zeros(())
            gaze_angular = uv_mean.new_zeros(())

        prior_log_weights = output["depth_prior_log_weights"]
        posterior_log_weights = output.get(
            "depth_posterior_log_weights",
            prior_log_weights,
        )
        if self.config.depth_prior_kl_weight > 0:
            posterior_weights = posterior_log_weights.exp()
            depth_prior_kl = (
                posterior_weights * (posterior_log_weights - prior_log_weights)
            ).sum(dim=-1).mean()
        else:
            depth_prior_kl = uv_mean.new_zeros(())

        if self.config.ray_validity_weight > 0:
            lambdas = output["lambda_hypotheses_mm"]
            normal_dot = output["normal_dot_gaze"].abs()
            min_lambda = output.get("geometry_min_lambda_mm")
            max_lambda = output.get("geometry_max_lambda_mm")
            min_dot = output.get("geometry_min_abs_normal_dot_gaze")
            if min_lambda is None or max_lambda is None or min_dot is None:
                raise KeyError(
                    "V4 model output must expose explicit geometry validity limits."
                )
            min_lambda_value = min_lambda.to(
                device=lambdas.device,
                dtype=lambdas.dtype,
            )
            max_lambda_value = max_lambda.to(
                device=lambdas.device,
                dtype=lambdas.dtype,
            )
            min_dot_value = min_dot.to(
                device=normal_dot.device,
                dtype=normal_dot.dtype,
            )
            ray_distance_penalty = (
                F.relu(min_lambda_value - lambdas)
                + F.relu(lambdas - max_lambda_value)
            ) / self.config.ray_penalty_scale_mm
            weighted_ray_penalty = (
                effective_weights * ray_distance_penalty
            ).sum(dim=-1).mean()
            parallel_penalty = (
                F.relu(min_dot_value - normal_dot) / min_dot_value
            ).mean()
            ray_validity = weighted_ray_penalty + parallel_penalty
        else:
            ray_validity = uv_mean.new_zeros(())

        if self.config.depth_correction_prior_weight > 0:
            standardized_correction = output.get(
                "depth_correction_standardized"
            )
            if standardized_correction is None:
                raise KeyError(
                    "depth_correction_prior_weight requires "
                    "DepthCorrectionHead outputs."
                )
            depth_correction_prior = 0.5 * standardized_correction.square().mean()
        else:
            depth_correction_prior = uv_mean.new_zeros(())

        total = (
            self.config.uv_huber_weight * uv_huber
            + self.config.mixture_nll_weight * mixture_nll
            + self.config.uv_gaussian_nll_weight * uv_gaussian_nll
            + self.config.gaze_direction_weight * gaze_direction_loss
            + self.config.gaze_angular_weight * gaze_angular
            + self.config.depth_prior_kl_weight * depth_prior_kl
            + self.config.ray_validity_weight * ray_validity
            + self.config.depth_correction_prior_weight * depth_correction_prior
        )
        return {
            "loss": total,
            "uv_huber": uv_huber.detach(),
            "mixture_nll": mixture_nll.detach(),
            "uv_gaussian_nll": uv_gaussian_nll.detach(),
            "gaze_direction": gaze_direction_loss.detach(),
            "gaze_angular": gaze_angular.detach(),
            "depth_prior_kl": depth_prior_kl.detach(),
            "ray_validity": ray_validity.detach(),
            "depth_correction_prior": depth_correction_prior.detach(),
        }

    @torch.no_grad()
    def metrics(self, output: Mapping[str, Tensor], uv_gt_mm: Tensor) -> dict[str, Tensor]:
        return compute_uv_metrics_mm(output["uv_mean_mm"], uv_gt_mm)


def _required_batch_tensor(batch: Mapping[str, object], key: str) -> Tensor:
    try:
        value = batch[key]
    except KeyError as exc:
        raise KeyError(f"Batch must contain {key!r} for geometry supervision.") from exc
    if not torch.is_tensor(value):
        raise TypeError(f"batch[{key!r}] must be a tensor.")
    return value


@torch.no_grad()
def compute_uv_metrics_mm(uv_pred_mm: Tensor, uv_gt_mm: Tensor) -> dict[str, Tensor]:
    uv_pred_mm = validate_uv_tensor(uv_pred_mm, "uv_pred_mm").float()
    uv_gt_mm = validate_uv_tensor(uv_gt_mm, "uv_gt_mm").to(
        device=uv_pred_mm.device,
        dtype=torch.float32,
    )
    error_mm = uv_pred_mm - uv_gt_mm
    abs_error_mm = error_mm.abs()
    epe_mm = torch.linalg.vector_norm(error_mm, dim=-1)
    return {
        "epe_mm": epe_mm.mean(),
        "median_epe_mm": epe_mm.median(),
        "mae_u_mm": abs_error_mm[..., 0].mean(),
        "mae_v_mm": abs_error_mm[..., 1].mean(),
    }
