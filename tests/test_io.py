from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brainsurgery.io import dcp as dcp_io
from brainsurgery.io import safetensors as safetensors_io
from brainsurgery.io import torch as torch_io


@pytest.mark.parametrize(
    "validator",
    [
        dcp_io._validate_state_dict_mapping,
        safetensors_io._validate_state_dict_mapping,
        torch_io._validate_state_dict_mapping,
    ],
)
def test_state_dict_validators_cover_non_mapping_non_tensor_and_success(validator) -> None:
    path = Path("/tmp/x")

    with pytest.raises(RuntimeError, match="not a state_dict mapping"):
        validator(123, path)

    with pytest.raises(RuntimeError, match="plain tensor state_dict"):
        validator({"bad": 1}, path)

    good = {"x": torch.ones(1)}
    loaded = validator(good, path)
    assert isinstance(loaded, dict)
    assert set(loaded) == {"x"}
    assert torch.equal(loaded["x"], torch.ones(1))


def test_safetensors_save_packs_non_contiguous_tensors(tmp_path: Path) -> None:
    # Regression: permute/phlora outputs are non-contiguous and safetensors refused
    # them at save time, aborting the run after all transforms had succeeded.
    transposed = torch.arange(12, dtype=torch.float32).reshape(3, 4).t()
    column_major = torch.linalg.svd(torch.randn(6, 5), full_matrices=False)[0][:, :2]
    assert not transposed.is_contiguous()

    path = tmp_path / "packed.safetensors"
    safetensors_io._save_state_dict({"t": transposed, "u": column_major}, path)
    loaded = safetensors_io._load_state_dict(path)
    assert torch.equal(loaded["t"], transposed)
    assert torch.equal(loaded["u"], column_major)

    single = tmp_path / "single.safetensors"
    safetensors_io._save_single_tensor("t", transposed, single)
    assert torch.equal(safetensors_io._load_single_tensor(single), transposed)
