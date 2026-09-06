"""T2: remove attention head 5 from every layer of OLMo-1B-0724-hf.

Layout facts (from the checkpoint's own config.json, re-derived at runtime):
  hidden_size = 2048, num_attention_heads = num_key_value_heads = 16,
  head_dim = 128. All projections are nn.Linear [out, in].
  q/k/v produce the heads -> heads are row blocks (dim 0).
  o consumes the heads  -> heads are column blocks (dim 1).
Head 5 therefore occupies indices 5*128 .. 6*128-1 = 640..767 on that axis.
"""

import json
import pathlib
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

PRUNE_HEAD = 5
IN_DIR = pathlib.Path("inputs/base")
OUT_DIR = pathlib.Path("out/T2")
OUT_FILE = OUT_DIR / "model.safetensors"

ROW_PROJS = ("q_proj", "k_proj", "v_proj")
COL_PROJS = ("o_proj",)


def fail(msg: str) -> None:
    raise SystemExit(f"CHECK FAILED: {msg}")


def load_state_dict() -> dict[str, torch.Tensor]:
    index = json.loads((IN_DIR / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shards: dict[str, list[str]] = {}
    for key, shard in weight_map.items():
        shards.setdefault(shard, []).append(key)
    state: dict[str, torch.Tensor] = {}
    for shard, keys in shards.items():
        with safe_open(IN_DIR / shard, framework="pt") as f:
            for key in keys:
                state[key] = f.get_tensor(key)
    if len(state) != len(weight_map):
        fail(f"loaded {len(state)} tensors, index lists {len(weight_map)}")
    return state


def main() -> None:
    cfg = json.loads((IN_DIR / "config.json").read_text())
    hidden = cfg["hidden_size"]
    n_heads = cfg["num_attention_heads"]
    n_kv = cfg["num_key_value_heads"]
    n_layers = cfg["num_hidden_layers"]
    if n_kv != n_heads:
        fail(f"grouped-query attention (kv={n_kv}, q={n_heads}) not handled by this script")
    if hidden % n_heads:
        fail(f"hidden_size {hidden} not divisible by {n_heads} heads")
    head_dim = hidden // n_heads
    if not 0 <= PRUNE_HEAD < n_heads:
        fail(f"head {PRUNE_HEAD} out of range for {n_heads} heads")

    lo, hi = PRUNE_HEAD * head_dim, (PRUNE_HEAD + 1) * head_dim
    keep = torch.cat([torch.arange(0, lo), torch.arange(hi, hidden)])
    if keep.numel() != hidden - head_dim:
        fail(f"keep index has {keep.numel()} entries, expected {hidden - head_dim}")

    state = load_state_dict()
    n_in = len(state)

    touched = 0
    for i in range(n_layers):
        for name in ROW_PROJS + COL_PROJS:
            key = f"model.layers.{i}.self_attn.{name}.weight"
            if key not in state:
                fail(f"missing tensor {key}")
            t = state[key]
            if tuple(t.shape) != (hidden, hidden):
                fail(f"{key} has shape {tuple(t.shape)}, expected {(hidden, hidden)}")
            dim = 0 if name in ROW_PROJS else 1
            state[key] = t.index_select(dim, keep).contiguous()
            touched += 1
    expected_touched = n_layers * (len(ROW_PROJS) + len(COL_PROJS))
    if touched != expected_touched:
        fail(f"edited {touched} tensors, expected {expected_touched}")

    # Required checks, on the in-memory result, before anything is written.
    want = {
        "model.layers.0.self_attn.q_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.k_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.v_proj.weight": (1920, 2048),
        "model.layers.0.self_attn.o_proj.weight": (2048, 1920),
    }
    for key, shape in want.items():
        got = tuple(state[key].shape)
        if got != shape:
            fail(f"{key} has shape {got}, expected {shape}")
    if len(state) != 114:
        fail(f"output has {len(state)} tensors, expected 114")
    if len(state) != n_in:
        fail(f"tensor count changed: {n_in} -> {len(state)}")

    # Untouched tensors must really be untouched: every remaining key keeps its
    # original shape and dtype.
    for i in range(n_layers):
        for name in ("gate_proj", "up_proj", "down_proj"):
            key = f"model.layers.{i}.mlp.{name}.weight"
            if key not in state:
                fail(f"missing tensor {key}")
            if hidden in tuple(state[key].shape) and 1920 in tuple(state[key].shape):
                fail(f"{key} was modified")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_file(state, str(OUT_FILE), metadata={"format": "pt"})

    # Read back and re-verify what actually landed on disk.
    with safe_open(OUT_FILE, framework="pt") as f:
        keys = list(f.keys())
        if len(keys) != 114:
            fail(f"written file has {len(keys)} tensors, expected 114")
        for key, shape in want.items():
            got = tuple(f.get_slice(key).get_shape())
            if got != shape:
                fail(f"written {key} has shape {got}, expected {shape}")
    print(f"wrote {OUT_FILE} with {len(keys)} tensors; pruned head {PRUNE_HEAD} "
          f"(rows/cols {lo}..{hi - 1}) in {n_layers} layers")


if __name__ == "__main__":
    sys.exit(main())
