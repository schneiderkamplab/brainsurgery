"""
T4: task-vector merge of two OLMo-1B fine-tunes.

out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])

for the 48 MLP tensors (gate_proj / up_proj / down_proj weights across the
16 layers); every other tensor is copied unchanged from the base after
verifying it is bit-identical across base, ft1 and ft2.
"""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
INPUTS = os.path.join(REPO_ROOT, "inputs")
OUT_DIR = os.path.join(REPO_ROOT, "out", "T4")
OUT_PATH = os.path.join(OUT_DIR, "model.safetensors")

LAMBDA = 0.4
NUM_LAYERS = 16
EXPECTED_TOTAL_TENSORS = 114
EXPECTED_MLP_TENSORS = 48


def mlp_tensor_names() -> set[str]:
    names = set()
    for i in range(NUM_LAYERS):
        for proj in ("gate_proj", "up_proj", "down_proj"):
            names.add(f"model.layers.{i}.mlp.{proj}.weight")
    return names


class ShardedCheckpoint:
    """Read-only view over a (possibly sharded) safetensors checkpoint."""

    def __init__(self, directory: str):
        index_path = os.path.join(directory, "model.safetensors.index.json")
        single_path = os.path.join(directory, "model.safetensors")
        self._handles = []
        self._key_to_handle = {}
        if os.path.exists(index_path):
            with open(index_path) as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            handle_by_file = {}
            for shard_file in shard_files:
                h = safe_open(os.path.join(directory, shard_file), framework="pt")
                handle_by_file[shard_file] = h
                self._handles.append(h)
            for key, shard_file in index["weight_map"].items():
                self._key_to_handle[key] = handle_by_file[shard_file]
        elif os.path.exists(single_path):
            h = safe_open(single_path, framework="pt")
            self._handles.append(h)
            for key in h.keys():
                self._key_to_handle[key] = h
        else:
            raise FileNotFoundError(f"no safetensors checkpoint found under {directory}")

    def keys(self) -> set[str]:
        return set(self._key_to_handle.keys())

    def get_tensor(self, key: str) -> torch.Tensor:
        return self._key_to_handle[key].get_tensor(key)


def main() -> None:
    base = ShardedCheckpoint(os.path.join(INPUTS, "base"))
    ft1 = ShardedCheckpoint(os.path.join(INPUTS, "ft1"))
    ft2 = ShardedCheckpoint(os.path.join(INPUTS, "ft2"))

    base_keys = base.keys()
    ft1_keys = ft1.keys()
    ft2_keys = ft2.keys()

    if not (base_keys == ft1_keys == ft2_keys):
        only_in_base = base_keys - ft1_keys - ft2_keys
        only_in_ft1 = ft1_keys - base_keys
        only_in_ft2 = ft2_keys - base_keys
        raise AssertionError(
            "checkpoints do not share the same tensor names: "
            f"only_in_base={sorted(only_in_base)}, only_in_ft1={sorted(only_in_ft1)}, "
            f"only_in_ft2={sorted(only_in_ft2)}"
        )

    mlp_names = mlp_tensor_names()
    if not mlp_names.issubset(base_keys):
        missing = mlp_names - base_keys
        raise AssertionError(f"expected MLP tensors missing from base checkpoint: {sorted(missing)}")
    if len(mlp_names) != EXPECTED_MLP_TENSORS:
        raise AssertionError(f"expected {EXPECTED_MLP_TENSORS} MLP tensor names, computed {len(mlp_names)}")

    # Step 1: verify every non-MLP tensor is bit-identical across all three checkpoints.
    shared_names = base_keys - mlp_names
    for key in shared_names:
        b = base.get_tensor(key)
        f1 = ft1.get_tensor(key)
        f2 = ft2.get_tensor(key)
        if b.shape != f1.shape or b.shape != f2.shape:
            raise AssertionError(f"shape mismatch on shared tensor {key!r}: {b.shape} vs {f1.shape} vs {f2.shape}")
        if b.dtype != f1.dtype or b.dtype != f2.dtype:
            raise AssertionError(f"dtype mismatch on shared tensor {key!r}: {b.dtype} vs {f1.dtype} vs {f2.dtype}")
        if not torch.equal(b, f1):
            raise AssertionError(f"shared tensor {key!r} differs between base and ft1; assumption violated")
        if not torch.equal(b, f2):
            raise AssertionError(f"shared tensor {key!r} differs between base and ft2; assumption violated")

    output: dict[str, torch.Tensor] = {}

    # Step 2: merge the 48 MLP tensors as task vectors taken against the unmodified base.
    merged_count = 0
    for key in sorted(mlp_names):
        b = base.get_tensor(key)
        f1 = ft1.get_tensor(key)
        f2 = ft2.get_tensor(key)
        if b.shape != f1.shape or b.shape != f2.shape:
            raise AssertionError(f"shape mismatch on MLP tensor {key!r}: {b.shape} vs {f1.shape} vs {f2.shape}")
        if b.dtype != torch.float32 or f1.dtype != torch.float32 or f2.dtype != torch.float32:
            raise AssertionError(f"MLP tensor {key!r} is not float32 in all three checkpoints")
        b32, f1_32, f2_32 = b.to(torch.float32), f1.to(torch.float32), f2.to(torch.float32)
        merged = b32 + LAMBDA * (f1_32 - b32) + LAMBDA * (f2_32 - b32)
        output[key] = merged.contiguous()
        merged_count += 1

    if merged_count != EXPECTED_MLP_TENSORS:
        raise AssertionError(f"merged {merged_count} tensors, expected {EXPECTED_MLP_TENSORS}")

    # Step 3: every other tensor is copied unchanged from the base.
    for key in sorted(shared_names):
        output[key] = base.get_tensor(key).contiguous()

    # Step 4: write a single-file checkpoint with exactly 114 tensors.
    if len(output) != EXPECTED_TOTAL_TENSORS:
        raise AssertionError(f"output has {len(output)} tensors, expected {EXPECTED_TOTAL_TENSORS}")

    os.makedirs(OUT_DIR, exist_ok=True)
    save_file(output, OUT_PATH, metadata={"format": "pt"})

    with safe_open(OUT_PATH, framework="pt") as h:
        written_keys = set(h.keys())
    if len(written_keys) != EXPECTED_TOTAL_TENSORS:
        raise AssertionError(f"written file has {len(written_keys)} tensors, expected {EXPECTED_TOTAL_TENSORS}")
    if written_keys != base_keys:
        raise AssertionError("written tensor names do not match the base checkpoint's tensor names")

    print(f"wrote {OUT_PATH} with {len(output)} tensors "
          f"({merged_count} merged, {len(shared_names)} unchanged)")


if __name__ == "__main__":
    main()
