"""First-order MAML components for the standalone Blaze-style model.

The representation encoder is deliberately kept outside the adaptable module.
This mirrors BlazeGaze's second training stage: a frozen encoder produces a
task-independent representation and only the gaze prediction head is adapted
on each support set.
"""

from __future__ import annotations

import copy
from collections import OrderedDict
from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.func import functional_call

from modelv1.blaze_style_model import BlazeStyleModel


FunctionalParameters = OrderedDict[str, Tensor]


class BlazeMAMLGazeHead(nn.Module):
    """Adaptable scene/FiLM/UV portion copied from a trained Blaze model."""

    def __init__(
        self,
        *,
        scene_branch: nn.Module,
        film_fusion: nn.Module,
        uv_head: nn.Module,
    ) -> None:
        super().__init__()
        self.scene_branch = scene_branch
        self.film_fusion = film_fusion
        self.uv_head = uv_head

    @classmethod
    def from_blaze_style_model(
        cls,
        model: BlazeStyleModel,
    ) -> "BlazeMAMLGazeHead":
        head = cls(
            scene_branch=copy.deepcopy(model.scene_branch),
            film_fusion=copy.deepcopy(
                model.direct_uv_table_frame_film_fusion
            ),
            uv_head=copy.deepcopy(model.uv_head),
        )
        head.requires_grad_(True)
        return head

    def set_adaptable_components(self, components: Sequence[str]) -> None:
        """Select the task-specific submodules updated by inner/outer loops."""

        normalized = tuple(str(value).strip() for value in components)
        if not normalized or any(not value for value in normalized):
            raise ValueError("At least one adaptable component is required.")
        known = {"scene_branch", "film_fusion", "uv_head"}
        unknown = set(normalized).difference(known)
        if unknown:
            raise ValueError(
                f"Unknown adaptable MAML components: {sorted(unknown)}."
            )
        for name, parameter in self.named_parameters():
            component = name.split(".", 1)[0]
            parameter.requires_grad_(component in normalized)

    def forward(
        self,
        visual_features: Tensor,
        table_frame7_n: Tensor,
    ) -> Tensor:
        if visual_features.ndim != 2:
            raise ValueError("visual_features must have shape [B, D].")
        if table_frame7_n.ndim != 2 or table_frame7_n.shape[-1] != 7:
            raise ValueError("table_frame7_n must have shape [B, 7].")
        if visual_features.shape[0] != table_frame7_n.shape[0]:
            raise ValueError("MAML head inputs must have matching batch sizes.")
        scene_features = self.scene_branch(table_frame7_n)
        fusion_input = torch.cat(
            (visual_features, scene_features),
            dim=-1,
        )
        fused_features, _ = self.film_fusion(
            fusion_input,
            table_frame7_n,
        )
        return self.uv_head(fused_features)


def blaze_maml_l2_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """BlazeGaze MAML loss: mean per-sample squared L2 distance."""

    if predictions.shape != targets.shape:
        raise ValueError(
            "MAML predictions and targets must have matching shapes; "
            f"got {tuple(predictions.shape)} and {tuple(targets.shape)}."
        )
    if predictions.ndim != 2 or predictions.shape[-1] != 2:
        raise ValueError("MAML predictions and targets must have shape [B, 2].")
    return (predictions - targets).square().sum(dim=-1).mean()


def functional_head_prediction(
    head: BlazeMAMLGazeHead,
    parameters: Mapping[str, Tensor],
    visual_features: Tensor,
    table_frame7_n: Tensor,
) -> Tensor:
    return functional_call(
        head,
        parameters,
        (visual_features, table_frame7_n),
        strict=False,
    )


def adapt_blaze_maml_head(
    head: BlazeMAMLGazeHead,
    support_visual_features: Tensor,
    support_table_frame7_n: Tensor,
    support_targets: Tensor,
    *,
    inner_lr: float,
    inner_steps: int,
    first_order: bool = True,
    detach_initial: bool = False,
) -> tuple[FunctionalParameters, Tensor]:
    """Return task-adapted parameters without mutating the meta head.

    With ``first_order=True`` the support gradients are not differentiated,
    yielding the first-order MAML approximation used by BlazeGaze. During
    meta-training the identity path from each adapted parameter to its
    initialization remains differentiable, so query gradients update the meta
    initialization.
    """

    if inner_lr <= 0:
        raise ValueError("inner_lr must be positive.")
    if inner_steps <= 0:
        raise ValueError("inner_steps must be positive.")
    if support_targets.shape[0] == 0:
        raise ValueError("The MAML support set must not be empty.")

    if detach_initial:
        parameters = OrderedDict(
            (
                name,
                parameter.detach().clone().requires_grad_(True),
            )
            for name, parameter in head.named_parameters()
            if parameter.requires_grad
        )
    else:
        parameters = OrderedDict(
            (name, parameter)
            for name, parameter in head.named_parameters()
            if parameter.requires_grad
        )
    if not parameters:
        raise ValueError("The MAML head has no adaptable parameters.")

    for step_index in range(inner_steps):
        predictions = functional_head_prediction(
            head,
            parameters,
            support_visual_features,
            support_table_frame7_n,
        )
        support_loss = blaze_maml_l2_loss(predictions, support_targets)
        if not torch.isfinite(support_loss):
            raise FloatingPointError(
                f"Non-finite MAML support loss at inner step {step_index + 1}."
            )
        gradients = torch.autograd.grad(
            support_loss,
            tuple(parameters.values()),
            create_graph=not first_order,
        )
        non_finite = [
            name
            for (name, _), gradient in zip(parameters.items(), gradients)
            if not torch.isfinite(gradient).all()
        ]
        if non_finite:
            raise FloatingPointError(
                "Non-finite MAML support gradients at inner step "
                f"{step_index + 1}: {non_finite}."
            )
        parameters = OrderedDict(
            (
                name,
                parameter - inner_lr * gradient,
            )
            for (name, parameter), gradient in zip(
                parameters.items(),
                gradients,
            )
        )

    final_predictions = functional_head_prediction(
        head,
        parameters,
        support_visual_features,
        support_table_frame7_n,
    )
    final_support_loss = blaze_maml_l2_loss(
        final_predictions,
        support_targets,
    )
    return parameters, final_support_loss


def sample_support_query_indices(
    candidate_indices: Sequence[int],
    *,
    support_size: int,
    query_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor]:
    """Sample disjoint support/query indices without replacement."""

    if support_size <= 0 or query_size <= 0:
        raise ValueError("support_size and query_size must be positive.")
    candidates = torch.as_tensor(candidate_indices, dtype=torch.long)
    required = support_size + query_size
    if candidates.numel() < required:
        raise ValueError(
            f"MAML task has {candidates.numel()} samples but requires {required}."
        )
    permutation = torch.randperm(candidates.numel(), generator=generator)
    selected = candidates[permutation[:required]]
    return selected[:support_size], selected[support_size:]
