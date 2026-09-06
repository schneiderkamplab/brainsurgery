"""T4: task-vector merge of two OLMo-1B fine-tunes into the base checkpoint.

out[X] = base[X] + lambda * (ft1[X] - base[X]) + lambda * (ft2[X] - base[X])
for the 48 MLP tensors; every other tensor is copied from the base.
"""

import json
import os

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
BASE_DIR = os.path.join(ROOT, "inputs", "base")
FT1 = os.path.join(ROOT, "inputs", "ft1", "model.safetensors")
FT2 = os.path.join(ROOT, "inputs", "ft2", "model.safetensors")
OUT = os.path.join(ROOT, "out", "T4", "model.safetensors")

LAMBDA = 0.4
N_LAYERS = 16
MLP_SUFFIXES = ("mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")
MLP_KEYS = {
    f"model.layers.{i}.{suffix}" for i in range(N_LAYERS) for suffix in MLP_SUFFIXES
}


class Checkpoint:
    """Lazy read-only view over one checkpoint (single file or sharded dir)."""

    def __init__(self, name, shard_of_key):
        self.name = name
        self._shard_of_key = shard_of_key
        self._handles = {}

    @classmethod
    def from_file(cls, name, path):
        with safe_open(path, framework="pt") as f:
            keys = list(f.keys())
        return cls(name, {k: path for k in keys})

    @classmethod
    def from_sharded_dir(cls, name, directory):
        index = os.path.join(directory, "model.safetensors.index.json")
        with open(index) as fh:
            weight_map = json.load(fh)["weight_map"]
        return cls(name, {k: os.path.join(directory, v) for k, v in weight_map.items()})

    def keys(self):
        return set(self._shard_of_key)

    def get(self, key):
        path = self._shard_of_key[key]
        handle = self._handles.get(path)
        if handle is None:
            handle = self._handles[path] = safe_open(path, framework="pt")
        return handle.get_tensor(key)


base = Checkpoint.from_sharded_dir("base", BASE_DIR)
ft1 = Checkpoint.from_file("ft1", FT1)
ft2 = Checkpoint.from_file("ft2", FT2)

# --- Step 1: verify the precondition before touching anything ----------------

for other in (ft1, ft2):
    if base.keys() != other.keys():
        missing = sorted(base.keys() - other.keys())
        extra = sorted(other.keys() - base.keys())
        raise SystemExit(
            f"key set mismatch base vs {other.name}: "
            f"missing={missing[:5]} extra={extra[:5]}"
        )

all_keys = sorted(base.keys())

missing_mlp = MLP_KEYS - set(all_keys)
if missing_mlp:
    raise SystemExit(f"expected MLP tensors absent from the checkpoints: {sorted(missing_mlp)}")
if len(MLP_KEYS) != 48:
    raise SystemExit(f"expected 48 MLP tensor names, built {len(MLP_KEYS)}")

shared_keys = [k for k in all_keys if k not in MLP_KEYS]
for key in shared_keys:
    b = base.get(key)
    for other in (ft1, ft2):
        t = other.get(key)
        if t.shape != b.shape or t.dtype != b.dtype:
            raise SystemExit(
                f"{key}: {other.name} has shape/dtype {tuple(t.shape)}/{t.dtype}, "
                f"base has {tuple(b.shape)}/{b.dtype}"
            )
        if not torch.equal(t, b):
            raise SystemExit(
                f"non-MLP tensor {key} differs between base and {other.name}; "
                "the frozen-backbone precondition does not hold, aborting"
            )
print(f"verified {len(shared_keys)} non-MLP tensors identical in all three checkpoints")

# --- Step 2/3: merge ---------------------------------------------------------

out = {}
merged = 0
for key in all_keys:
    b = base.get(key)
    if key not in MLP_KEYS:
        out[key] = b.contiguous().clone()
        continue
    t1 = ft1.get(key)
    t2 = ft2.get(key)
    for other_name, t in (("ft1", t1), ("ft2", t2)):
        if t.shape != b.shape or t.dtype != b.dtype:
            raise SystemExit(
                f"{key}: {other_name} has shape/dtype {tuple(t.shape)}/{t.dtype}, "
                f"base has {tuple(b.shape)}/{b.dtype}"
            )
    if b.dtype != torch.float32:
        raise SystemExit(f"{key}: expected float32, got {b.dtype}")
    # Both task vectors are taken against the *unmodified* base.
    merged_tensor = b + LAMBDA * (t1 - b) + LAMBDA * (t2 - b)
    out[key] = merged_tensor.to(torch.float32).contiguous()
    merged += 1

if merged != 48:
    raise SystemExit(f"expected to merge exactly 48 tensors, merged {merged}")
if len(out) != 114:
    raise SystemExit(f"expected 114 output tensors, have {len(out)}")

os.makedirs(os.path.dirname(OUT), exist_ok=True)
save_file(out, OUT)

with safe_open(OUT, framework="pt") as f:
    written = list(f.keys())
if len(written) != 114:
    raise SystemExit(f"output file holds {len(written)} tensors, expected 114")
if set(written) != set(all_keys):
    raise SystemExit("output key set differs from the base key set")

print(f"merged {merged} MLP tensors, wrote {len(written)} tensors to {OUT}")
