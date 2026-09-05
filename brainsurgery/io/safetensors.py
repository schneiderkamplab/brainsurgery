from pathlib import Path

import torch
from safetensors.torch import load_file as _load_file
from safetensors.torch import save_file as _save_file


def _load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    loaded = _load_file(str(path), device="cpu")
    return _validate_state_dict_mapping(loaded, path)


def _save_state_dict(state_dict: dict[str, torch.Tensor], path: Path) -> None:
    _save_file(_contiguous_state_dict(state_dict), str(path))


def _load_single_tensor(path: Path) -> torch.Tensor:
    loaded = _load_file(str(path), device="cpu")
    if len(loaded) != 1:
        raise RuntimeError(
            "loading tensor from safetensors requires exactly one tensor, or use state_dict load"
        )
    return next(iter(loaded.values()))


def _save_single_tensor(tensor_name: str, tensor: torch.Tensor, path: Path) -> None:
    _save_file({tensor_name: tensor.contiguous()}, str(path))


def _contiguous_state_dict(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Return a state_dict whose tensors are row-major contiguous.

    safetensors refuses non-contiguous tensors. Transforms such as ``permute`` or
    ``phlora`` (whose SVD factors come back column-major from LAPACK) leave such
    tensors in the in-memory provider, so pack them at save time. Tensors that are
    already contiguous are passed through unchanged, without a copy.
    """
    return {name: tensor.contiguous() for name, tensor in state_dict.items()}


def _validate_state_dict_mapping(loaded: object, path: Path) -> dict[str, torch.Tensor]:
    if not isinstance(loaded, dict):
        raise RuntimeError(f"checkpoint at {path} is not a state_dict mapping")
    if not all(isinstance(k, str) and torch.is_tensor(v) for k, v in loaded.items()):
        raise RuntimeError(f"checkpoint at {path} is not a plain tensor state_dict")
    return dict(loaded)
