"""T4: task-vector merge of two OLMo-1B fine-tunes into the base checkpoint."""

import json
import os
import re

import torch
from safetensors import safe_open
from safetensors.torch import save_file

BASE_DIR = "inputs/base"
FT1_FILE = "inputs/ft1/model.safetensors"
FT2_FILE = "inputs/ft2/model.safetensors"
OUT_FILE = "out/T4/model.safetensors"

LAMBDA = 0.4
MLP_RE = re.compile(r"^model\.layers\.\d+\.mlp\.(gate_proj|up_proj|down_proj)\.weight$")
N_MLP = 48
N_TOTAL = 114


def load_sharded(directory):
    index_path = os.path.join(directory, "model.safetensors.index.json")
    with open(index_path) as fh:
        weight_map = json.load(fh)["weight_map"]
    state = {}
    for shard in sorted(set(weight_map.values())):
        with safe_open(os.path.join(directory, shard), framework="pt") as fh:
            for name in fh.keys():
                state[name] = fh.get_tensor(name)
    missing = set(weight_map) - set(state)
    if missing:
        raise RuntimeError(f"index lists tensors absent from the shards: {sorted(missing)}")
    return state


def load_single(path):
    with safe_open(path, framework="pt") as fh:
        return {name: fh.get_tensor(name) for name in fh.keys()}


def main():
    base = load_sharded(BASE_DIR)
    ft1 = load_single(FT1_FILE)
    ft2 = load_single(FT2_FILE)

    # 1. same tensor names everywhere
    for label, other in (("ft1", ft1), ("ft2", ft2)):
        if set(other) != set(base):
            only_base = sorted(set(base) - set(other))
            only_other = sorted(set(other) - set(base))
            raise RuntimeError(
                f"{label} tensor names differ from base; "
                f"only in base: {only_base[:5]}, only in {label}: {only_other[:5]}"
            )

    if len(base) != N_TOTAL:
        raise RuntimeError(f"expected {N_TOTAL} tensors in the base, found {len(base)}")

    mlp_names = sorted(n for n in base if MLP_RE.match(n))
    if len(mlp_names) != N_MLP:
        raise RuntimeError(f"expected {N_MLP} MLP tensors, matched {len(mlp_names)}")

    # 1 (cont.) every non-MLP tensor must be bit-identical in all three
    mlp_set = set(mlp_names)
    for name in sorted(base):
        if name in mlp_set:
            continue
        b = base[name]
        for label, other in (("ft1", ft1), ("ft2", ft2)):
            o = other[name]
            if o.shape != b.shape or o.dtype != b.dtype:
                raise RuntimeError(
                    f"{label}:{name} has shape/dtype {tuple(o.shape)}/{o.dtype}, "
                    f"base has {tuple(b.shape)}/{b.dtype}"
                )
            if not torch.equal(o, b):
                raise RuntimeError(
                    f"shared tensor {name!r} differs between base and {label}; "
                    "the frozen-backbone assumption does not hold"
                )

    # shapes/dtypes of the MLP tensors must line up too
    for name in mlp_names:
        b = base[name]
        for label, other in (("ft1", ft1), ("ft2", ft2)):
            o = other[name]
            if o.shape != b.shape or o.dtype != b.dtype:
                raise RuntimeError(
                    f"{label}:{name} has shape/dtype {tuple(o.shape)}/{o.dtype}, "
                    f"base has {tuple(b.shape)}/{b.dtype}"
                )
        if b.dtype != torch.float32:
            raise RuntimeError(f"{name} is {b.dtype}, expected float32")

    # 2./3. merge; each task vector is taken against the *unmodified* base
    out = {}
    merged = 0
    for name, tensor in base.items():
        if name in mlp_set:
            b = tensor.to(torch.float32)
            tv1 = ft1[name].to(torch.float32) - b
            tv2 = ft2[name].to(torch.float32) - b
            out[name] = (b + LAMBDA * tv1 + LAMBDA * tv2).contiguous()
            merged += 1
        else:
            out[name] = tensor.clone().contiguous()

    if merged != N_MLP:
        raise RuntimeError(f"merged {merged} tensors, expected {N_MLP}")
    if len(out) != N_TOTAL:
        raise RuntimeError(f"output has {len(out)} tensors, expected {N_TOTAL}")

    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    save_file(out, OUT_FILE)

    with safe_open(OUT_FILE, framework="pt") as fh:
        written = list(fh.keys())
    if len(written) != N_TOTAL:
        raise RuntimeError(f"wrote {len(written)} tensors, expected {N_TOTAL}")

    print(f"merged {merged} MLP tensors, wrote {len(written)} tensors to {OUT_FILE}")


if __name__ == "__main__":
    main()
