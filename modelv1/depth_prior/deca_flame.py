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

# These are the FLAME vertex ids used by the existing CrossGaze baseline.
FLAME_LEFT_EYE_CENTER_VERTEX = 3933
FLAME_RIGHT_EYE_CENTER_VERTEX = 3930


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
    left_eye_vertex_index: int = FLAME_LEFT_EYE_CENTER_VERTEX
    right_eye_vertex_index: int = FLAME_RIGHT_EYE_CENTER_VERTEX

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
        if self.left_eye_vertex_index < 0 or self.right_eye_vertex_index < 0:
            raise ValueError("eye vertex indices must be non-negative")


@dataclass(frozen=True)
class DecaFlameOutput:
    """DECA-FLAME geometry in FLAME local coordinates.

    The tensors have batch-first layout. ``vertices`` is normally
    ``[B, 5023, 3]`` and ``landmarks3d`` is normally ``[B, 68, 3]``.
    ``landmarks2d`` is FLAME's dynamic 2D landmark set, but its values are
    still 3D FLAME-local coordinates. None of these outputs are camera-space
    coordinates or millimetres yet.
    """

    vertices: Any
    landmarks2d: Any
    landmarks3d: Any
    left_eye_vertex: Any
    right_eye_vertex: Any
    parameters: Any

    @property
    def batch_size(self) -> int:
        return int(self.vertices.shape[0])


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


def _import_official_deca(deca_root: Path) -> tuple[Any, Any, Any, Any]:
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
        from decalib.utils import util as deca_util
        from decalib.utils.config import get_cfg_defaults
    except ImportError as error:  # pragma: no cover - external dependency
        raise DecaFlameDependencyError(
            "Could not import the official DECA modules. Check the DECA "
            f"dependencies in DECA-master. Original import error: {error}"
        ) from error

    return ResnetEncoder, FLAME, deca_util, get_cfg_defaults


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


class DecaFlameExtractor:
    """Run official DECA and FLAME once for each aligned face crop.

    This object is for preprocessing.  ``extract`` runs under ``no_grad`` and
    returns detached CPU tensors so its outputs can be saved as a depth prior.
    """

    def __init__(self, config: DecaFlameConfig | None = None) -> None:
        self.config = config or DecaFlameConfig()
        torch = _require_torch()
        ResnetEncoder, FLAME, deca_util, get_cfg_defaults = _import_official_deca(
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
        deca_util.copy_state_dict(self.encoder.state_dict(), checkpoint["E_flame"])

        self.encoder.eval()
        self.flame.eval()

    def extract(self, face_images: Any) -> DecaFlameOutput:
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
            vertices, landmarks2d, landmarks3d = self.flame(
                shape_params=code_dict["shape"],
                expression_params=code_dict["exp"],
                pose_params=code_dict["pose"],
            )

        max_vertex_index = max(
            self.config.left_eye_vertex_index,
            self.config.right_eye_vertex_index,
        )
        if vertices.shape[1] <= max_vertex_index:
            raise ValueError(
                "Configured eye vertex index is outside the FLAME mesh: "
                f"mesh has {vertices.shape[1]} vertices, required {max_vertex_index}."
            )

        return DecaFlameOutput(
            vertices=vertices.detach().cpu(),
            landmarks2d=landmarks2d.detach().cpu(),
            landmarks3d=landmarks3d.detach().cpu(),
            left_eye_vertex=vertices[:, self.config.left_eye_vertex_index, :].detach().cpu(),
            right_eye_vertex=vertices[:, self.config.right_eye_vertex_index, :].detach().cpu(),
            parameters=parameters.detach().cpu(),
        )
