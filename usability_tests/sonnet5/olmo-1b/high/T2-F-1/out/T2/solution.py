"""T2: structured attention-head pruning for OLMo-1B-0724-hf.

Removes head 5 (0-indexed, head_dim=128, 16 heads) from every layer's
self-attention block, at the checkpoint level, without loading the model
into a transformers architecture (a 15-head OLMo isn't representable by the
unmodified 16-head config, so this edits raw tensors directly rather than
going through `PreTrainedModel.prune_heads`, whose head-index bookkeeping
assumes it can reload the same config).

Layout facts used here (see TASK.md):
- q_proj/k_proj/v_proj: [2048, 2048], heads are contiguous row blocks of 128.
- o_proj: [2048, 2048], heads are contiguous column blocks of 128.
- head 5 occupies rows/cols 640..767 inclusive; the kept span is
  0..639 followed by 768..2047, in that order.
- MLP tensors and everything else are untouched.
"""

import json
import sys
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
INPUT_DIR = HERE.parents[1] / "inputs" / "base"
OUTPUT_PATH = HERE / "model.safetensors"

NUM_LAYERS = 16
HEAD_DIM = 128
HEAD_TO_REMOVE = 5
REMOVE_START = HEAD_TO_REMOVE * HEAD_DIM  # 640
REMOVE_END = REMOVE_START + HEAD_DIM  # 768

Q_K_V = ("q_proj", "k_proj", "v_proj")


def load_all_tensors(input_dir: Path) -> dict[str, torch.Tensor]:
    index = json.loads((input_dir / "model.safetensors.index.json").read_text())
    weight_map = index["weight_map"]
    shard_names = sorted(set(weight_map.values()))
    tensors: dict[str, torch.Tensor] = {}
    for shard_name in shard_names:
        with safe_open(input_dir / shard_name, framework="pt") as f:
            for key in f.keys():
                tensors[key] = f.get_tensor(key)
    assert set(tensors) == set(weight_map), "loaded keys do not match index"
    return tensors


def drop_rows(t: torch.Tensor) -> torch.Tensor:
    return torch.cat([t[:REMOVE_START], t[REMOVE_END:]], dim=0).contiguous()


def drop_cols(t: torch.Tensor) -> torch.Tensor:
    return torch.cat([t[:, :REMOVE_START], t[:, REMOVE_END:]], dim=1).contiguous()


def main() -> None:
    tensors = load_all_tensors(INPUT_DIR)
    assert len(tensors) == 114, f"expected 114 input tensors, got {len(tensors)}"

    out: dict[str, torch.Tensor] = {}
    for name, t in tensors.items():
        is_head_bearing = False
        for i in range(NUM_LAYERS):
            prefix = f"model.layers.{i}.self_attn."
            if not name.startswith(prefix):
                continue
            proj = name[len(prefix):]
            if proj in ("q_proj.weight", "k_proj.weight", "v_proj.weight"):
                out[name] = drop_rows(t)
                is_head_bearing = True
            elif proj == "o_proj.weight":
                out[name] = drop_cols(t)
                is_head_bearing = True
            break
        if not is_head_bearing:
            out[name] = t

    # Required checks: fail loudly before writing anything.
    for i in range(NUM_LAYERS):
        for proj in Q_K_V:
            key = f"model.layers.{i}.self_attn.{proj}.weight"
            shape = tuple(out[key].shape)
            assert shape == (1920, 2048), f"{key} has shape {shape}, expected (1920, 2048)"
        key = f"model.layers.{i}.self_attn.o_proj.weight"
        shape = tuple(out[key].shape)
        assert shape == (2048, 1920), f"{key} has shape {shape}, expected (2048, 1920)"
    assert len(out) == 114, f"expected 114 output tensors, got {len(out)}"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    save_file(out, str(OUTPUT_PATH))
    print(f"wrote {len(out)} tensors to {OUTPUT_PATH}")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)
