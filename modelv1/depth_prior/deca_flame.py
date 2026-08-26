"""Official DECA and FLAME inference wrapper for offline depth priors.

This module deliberately reuses the official DECA implementation under
``DECA-master/decalib``.  It does not reimplement the FLAME layer.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DECA_ROOT = PROJECT_ROOT / "DECA-master"

class DecaFlameDependencyError(RuntimeError):
    """Raised when the local official DECA dependency is unavailable."""


@dataclass(frozen=True)
class DecaFlameConfig:
    """Configuration for :class:`DecaFlameExtractor`.

    ``pretrained_model_path`` defaults to the checkpoint configured by the
    official DECA config.  Input tensors must be RGB face crops of shape
    ``[batch, 3, image_size, image_size]`` with values in ``[0, 1]``.
    """

    deca_root: Path | str = DEFAULT_DECA_ROOT
    pretrained_model_path: Path | str | None = None
    device: str | None = None
    image_size: int = 224

    def __post_init__(self) -> None:
        object.__setattr__(self, "deca_root", Path(self.deca_root).resolve())
        if self.pretrained_model_path is not None:
            object.__setattr__(
                self,
                "pretrained_model_path",
                Path(self.pretrained_model_path).resolve(),
            )
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")


@dataclass(frozen=True)
class DecaFlameOutput:
    """DECA-FLAME geometry in FLAME local coordinates.

    ``head_local_landmarks3d`` has shape ``[B, 68, 3]`` and is evaluated with
    DECA's global head rotation zeroed while retaining shape, expression, and
    jaw pose. It is the only landmark set that may be passed to solvePnP.
    These values are not camera-space coordinates or millimetres yet.

    The former full mesh, globally posed landmarks, and two hard-coded eye
    vertices were removed because no ModelV1 consumer used them and the extra
    FLAME forward pass could accidentally reintroduce posed object points.
    """

    head_local_landmarks3d: Any
    parameters: Any

    @property
    def batch_size(self) -> int:
        return int(self.head_local_landmarks3d.shape[0])


def _require_torch() -> Any:
    try:
        import torch
    except ImportError as error:  # pragma: no cover - depends on environment
        raise DecaFlameDependencyError(
            "PyTorch is required to run the DECA-FLAME extractor."
        ) from error
    return torch


def _enable_chumpy_compatibility() -> None:
    """Restore NumPy and inspect APIs required by the legacy FLAME dependency.

    Official DECA's ``generic_model.pkl`` can import the unmaintained ``chumpy``
    package while unpickling. Chumpy still imports aliases removed by NumPy 1.24
    and calls ``inspect.getargspec``, which Python removed in 3.11. Define only
    the historical APIs that are absent, so the DECA preprocessing wrapper works
    without downgrading the whole project runtime.
    """

    try:
        import numpy as np
    except ImportError as error:  # pragma: no cover - external dependency
        raise DecaFlameDependencyError(
            "NumPy is required to load the official FLAME model."
        ) from error

    legacy_aliases = {
        "bool": np.bool_,
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "unicode": str,
        "str": str,
    }
    for name, value in legacy_aliases.items():
        if name not in np.__dict__:
            setattr(np, name, value)

    import inspect
    from collections import namedtuple

    if not hasattr(inspect, "getargspec"):
        arg_spec = namedtuple("ArgSpec", ("args", "varargs", "keywords", "defaults"))

        def getargspec(function: Any) -> Any:
            full_spec = inspect.getfullargspec(function)
            return arg_spec(
                full_spec.args,
                full_spec.varargs,
                full_spec.varkw,
                full_spec.defaults,
            )

        inspect.getargspec = getargspec  # type: ignore[attr-defined]


def _import_official_deca(deca_root: Path) -> tuple[Any, Any, Any]:
    """Import DECA from the repository-local official checkout only."""

    decalib_root = deca_root / "decalib"
    if not decalib_root.is_dir():
        raise DecaFlameDependencyError(
            f"Official DECA source was not found at: {decalib_root}"
        )

    root_text = str(deca_root)
    if root_text not in sys.path:
        sys.path.insert(0, root_text)

    try:
        from decalib.models.FLAME import FLAME
        from decalib.models.encoders import ResnetEncoder
        from decalib.utils.config import get_cfg_defaults
    except ImportError as error:  # pragma: no cover - external dependency
        raise DecaFlameDependencyError(
            "Could not import the official DECA modules. Check the DECA "
            f"dependencies in DECA-master. Original import error: {error}"
        ) from error

    return ResnetEncoder, FLAME, get_cfg_defaults


def _copy_matching_state_dict(
    current_state_dict: Any,
    pretrained_state_dict: Any,
) -> int:
    """Copy matching DECA encoder tensors without importing ``decalib.utils.util``.

    Official DECA's utility module imports SciPy and scikit-image at module
    import time even though depth-prior generation only needs its small
    ``copy_state_dict`` helper.  In environments where pip PyTorch bundles one
    Intel OpenMP runtime and Conda MKL provides another, those unrelated imports
    initialize both ``libiomp5md.dll`` copies and terminate the process with
    OMP Error #15.  Keeping the equivalent tensor-copy operation local avoids
    that binary dependency chain.
    """

    copied = 0
    for name, destination in current_state_dict.items():
        source = pretrained_state_dict.get(name)
        if source is None or getattr(source, "shape", None) != destination.shape:
            continue
        destination.copy_(source)
        copied += 1
    if copied == 0:
        raise ValueError(
            "No matching DECA encoder tensors were found in checkpoint E_flame."
        )
    return copied


def _split_deca_parameters(parameters: Any, param_sizes: dict[str, int]) -> dict[str, Any]:
    """Split the official DECA encoder vector into named parameter blocks."""

    torch = _require_torch()
    ordered_names = tuple(param_sizes)
    expected_size = sum(param_sizes.values())
    if parameters.ndim != 2 or parameters.shape[1] != expected_size:
        raise ValueError(
            "Unexpected DECA parameter tensor shape: "
            f"expected [B, {expected_size}], got {tuple(parameters.shape)}"
        )
    values = torch.split(parameters, tuple(param_sizes[name] for name in ordered_names), dim=1)
    return dict(zip(ordered_names, values))


def head_local_pose_parameters(pose_parameters: Any) -> Any:
    """Copy a DECA pose and remove only its global head rotation.

    The remaining jaw pose is part of the non-rigid landmark configuration;
    solvePnP must estimate the removed global rotation from 2D observations.
    """

    torch = _require_torch()
    pose = torch.as_tensor(pose_parameters)
    if pose.ndim != 2 or pose.shape[1] < 6:
        raise ValueError(
            "DECA pose parameters must have shape [B, >=6], got "
            f"{tuple(pose.shape)}"
        )
    local_pose = pose.clone()
    local_pose[:, :3] = 0.0
    return local_pose


class DecaFlameExtractor:
    """Run official DECA and FLAME once for each aligned face crop.

    This object is for preprocessing.  ``extract`` runs under ``no_grad`` and
    returns detached CPU tensors so its outputs can be saved as a depth prior.
    """

    def __init__(self, config: DecaFlameConfig | None = None) -> None:
        self.config = config or DecaFlameConfig()
        torch = _require_torch()
        ResnetEncoder, FLAME, get_cfg_defaults = _import_official_deca(
            self.config.deca_root
        )

        cfg = get_cfg_defaults()
        if self.config.pretrained_model_path is not None:
            cfg.pretrained_modelpath = str(self.config.pretrained_model_path)

        checkpoint_path = Path(cfg.pretrained_modelpath)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(
                "The DECA pretrained checkpoint was not found at: "
                f"{checkpoint_path}"
            )

        requested_device = self.config.device
        if requested_device is None:
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(requested_device)

        self._param_sizes = {
            "shape": int(cfg.model.n_shape),
            "tex": int(cfg.model.n_tex),
            "exp": int(cfg.model.n_exp),
            "pose": int(cfg.model.n_pose),
            "cam": int(cfg.model.n_cam),
            "light": int(cfg.model.n_light),
        }
        self.encoder = ResnetEncoder(outsize=sum(self._param_sizes.values())).to(self.device)
        _enable_chumpy_compatibility()
        self.flame = FLAME(cfg.model).to(self.device)

        checkpoint = torch.load(str(checkpoint_path), map_location="cpu")
        if "E_flame" not in checkpoint:
            raise KeyError(
                "The DECA checkpoint does not contain the required 'E_flame' weights."
            )
        _copy_matching_state_dict(
            self.encoder.state_dict(),
            checkpoint["E_flame"],
        )

        self.encoder.eval()
        self.flame.eval()

    def extract(
        self,
        face_images: Any,
        *,
        fixed_shape_params: Any | None = None,
    ) -> DecaFlameOutput:
        """Return FLAME mesh and landmarks for a batch of RGB face crops.

        Args:
            face_images: Float tensor with shape ``[B, 3, 224, 224]`` by
                default. Values must follow the same ``[0, 1]`` RGB convention
                used by the official DECA preprocessing.
        """

        torch = _require_torch()
        if not torch.is_tensor(face_images):
            raise TypeError("face_images must be a torch.Tensor")
        if face_images.ndim != 4 or face_images.shape[1] != 3:
            raise ValueError(
                "face_images must have shape [batch, 3, height, width], got "
                f"{tuple(face_images.shape)}"
            )
        if tuple(face_images.shape[-2:]) != (self.config.image_size, self.config.image_size):
            raise ValueError(
                "DECA expects aligned face crops of shape "
                f"[{self.config.image_size}, {self.config.image_size}], got "
                f"{tuple(face_images.shape[-2:])}"
            )

        images = face_images.to(device=self.device, dtype=torch.float32)
        with torch.no_grad():
            parameters = self.encoder(images)
            code_dict = _split_deca_parameters(parameters, self._param_sizes)
            if fixed_shape_params is not None:
                fixed_shape = torch.as_tensor(
                    fixed_shape_params,
                    device=self.device,
                    dtype=torch.float32,
                )
                shape_dim = self._param_sizes["shape"]
                if fixed_shape.ndim == 1:
                    fixed_shape = fixed_shape.unsqueeze(0)
                if fixed_shape.ndim != 2 or fixed_shape.shape[1] != shape_dim:
                    raise ValueError(
                        "fixed_shape_params must have shape "
                        f"[{shape_dim}] or [B, {shape_dim}], got "
                        f"{tuple(fixed_shape.shape)}"
                    )
                if fixed_shape.shape[0] == 1:
                    fixed_shape = fixed_shape.expand(images.shape[0], -1)
                elif fixed_shape.shape[0] != images.shape[0]:
                    raise ValueError(
                        "fixed_shape_params batch dimension must be 1 or match "
                        f"the image batch size {images.shape[0]}."
                    )
                code_dict["shape"] = fixed_shape
            head_local_pose = head_local_pose_parameters(code_dict["pose"])
            _, _, head_local_landmarks3d = self.flame(
                shape_params=code_dict["shape"],
                expression_params=code_dict["exp"],
                pose_params=head_local_pose,
            )

        return DecaFlameOutput(
            head_local_landmarks3d=head_local_landmarks3d.detach().cpu(),
            parameters=parameters.detach().cpu(),
        )
