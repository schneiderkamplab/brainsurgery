"""T2: structured attention-head pruning of OLMo-1B-0724-hf.

Removes head 5 (of 16, head_dim 128) from every layer: rows 640..767 of
q/k/v_proj and columns 640..767 of o_proj. Everything else is copied verbatim.
"""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

HERE = Path(__file__).resolve().parent
BASE = HERE.parent.parent / "inputs" / "base"
OUT = HERE / "model.safetensors"

NUM_HEADS = 16
HEAD_DIM = 128
PRUNE_HEAD = 5
LO = PRUNE_HEAD * HEAD_DIM          # 640
HI = LO + HEAD_DIM                  # 768
HIDDEN = NUM_HEADS * HEAD_DIM       # 2048
KEPT = HIDDEN - HEAD_DIM            # 1920

ROW_PRUNED = re.compile(r"^model\.layers\.\d+\.self_attn\.[qkv]_proj\.weight$")
COL_PRUNED = re.compile(r"^model\.layers\.\d+\.self_attn\.o_proj\.weight$")


def keep_index(n: int) -> torch.Tensor:
    return torch.cat([torch.arange(0, LO), torch.arange(HI, n)])


def load_all() -> dict[str, torch.Tensor]:
    index = json.loads((BASE / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    tensors: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        shard_tensors = load_file(str(BASE / shard))
        for name, tensor in shard_tensors.items():
            if name in tensors:
                raise SystemExit(f"duplicate tensor across shards: {name}")
            tensors[name] = tensor
    missing = set(weight_map) - set(tensors)
    extra = set(tensors) - set(weight_map)
    if missing or extra:
        raise SystemExit(f"index/shard mismatch: missing={sorted(missing)} extra={sorted(extra)}")
    return tensors


def main() -> None:
    src = load_all()
    print(f"loaded {len(src)} tensors from {BASE}")
    if len(src) != 114:
        raise SystemExit(f"expected 114 input tensors, got {len(src)}")

    out: dict[str, torch.Tensor] = {}
    n_row = n_col = 0
    for name, tensor in src.items():
        if ROW_PRUNED.match(name):
            if tuple(tensor.shape) != (HIDDEN, HIDDEN):
                raise SystemExit(f"{name}: expected [{HIDDEN}, {HIDDEN}], got {list(tensor.shape)}")
            out[name] = tensor[keep_index(tensor.shape[0]), :].contiguous().clone()
            n_row += 1
        elif COL_PRUNED.match(name):
            if tuple(tensor.shape) != (HIDDEN, HIDDEN):
                raise SystemExit(f"{name}: expected [{HIDDEN}, {HIDDEN}], got {list(tensor.shape)}")
            out[name] = tensor[:, keep_index(tensor.shape[1])].contiguous().clone()
            n_col += 1
        else:
            out[name] = tensor.contiguous().clone()

    # Every layer must have been touched: 16 layers x 3 row-pruned, x 1 col-pruned.
    if n_row != 48 or n_col != 16:
        raise SystemExit(f"expected 48 row-pruned and 16 col-pruned tensors, got {n_row} and {n_col}")

    # Required checks.
    for proj, want in (
        ("q_proj", (KEPT, HIDDEN)),
        ("k_proj", (KEPT, HIDDEN)),
        ("v_proj", (KEPT, HIDDEN)),
        ("o_proj", (HIDDEN, KEPT)),
    ):
        key = f"model.layers.0.self_attn.{proj}.weight"
        got = tuple(out[key].shape)
        if got != want:
            raise SystemExit(f"{key}: expected {list(want)}, got {list(got)}")
        print(f"check ok: {key} -> {list(got)}")
    if len(out) != 114:
        raise SystemExit(f"expected 114 output tensors, got {len(out)}")
    print("check ok: 114 tensors")

    # Value spot-check: the kept blocks must be bit-identical to the source.
    ref_q = src["model.layers.0.self_attn.q_proj.weight"]
    new_q = out["model.layers.0.self_attn.q_proj.weight"]
    if not (torch.equal(new_q[:LO], ref_q[:LO]) and torch.equal(new_q[LO:], ref_q[HI:])):
        raise SystemExit("q_proj row blocks are not in the expected order")
    ref_o = src["model.layers.0.self_attn.o_proj.weight"]
    new_o = out["model.layers.0.self_attn.o_proj.weight"]
    if not (
        torch.equal(new_o[:, :LO], ref_o[:, :LO]) and torch.equal(new_o[:, LO:], ref_o[:, HI:])
    ):
        raise SystemExit("o_proj column blocks are not in the expected order")
    print("check ok: kept blocks bit-identical and in order")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUT))
    print(f"wrote {OUT} ({OUT.stat().st_size} bytes, {len(out)} tensors)")


if __name__ == "__main__":
    main()
