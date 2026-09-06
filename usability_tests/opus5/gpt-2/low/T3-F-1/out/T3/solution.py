"""T3: mixed-precision sharded export of GPT-2 (124M).

Plain torch + safetensors + huggingface_hub's standard shard splitter.
"""
import json
import re
import sys
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file

SRC = Path("inputs/base/model.safetensors")
DST = Path("out/T3")
MAX_SHARD_BYTES = 64 * 1024 * 1024

# Exactly the four projection matrices per layer; anchored so nothing else matches.
PROJ = re.compile(r"^h\.\d+\.(?:attn\.c_attn|attn\.c_proj|mlp\.c_fc|mlp\.c_proj)\.weight$")
BUFFER = re.compile(r"^h\.\d+\.attn\.bias$")


def main() -> None:
    src = load_file(str(SRC))

    out: dict[str, torch.Tensor] = {}
    for name, t in src.items():
        if BUFFER.match(name):
            continue
        out[name] = t.to(torch.bfloat16) if PROJ.match(name) else t.to(torch.float32)

    # Required checks: fail loudly before writing anything.
    n_bf16 = sum(1 for t in out.values() if t.dtype is torch.bfloat16)
    assert n_bf16 == 48, f"expected 48 bfloat16 tensors, got {n_bf16}"
    assert out["h.0.attn.c_attn.weight"].dtype is torch.bfloat16, "h.0.attn.c_attn.weight not bfloat16"
    assert out["wte.weight"].dtype is torch.float32, "wte.weight not float32"
    assert len(out) == 148, f"expected 148 tensors, got {len(out)}"
    dropped = set(src) - set(out)
    assert dropped == {f"h.{i}.attn.bias" for i in range(12)}, f"unexpected deletions: {dropped}"

    DST.mkdir(parents=True, exist_ok=True)
    split = split_torch_state_dict_into_shards(
        out,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=MAX_SHARD_BYTES,
    )
    for filename, keys in split.filename_to_tensors.items():
        save_file(
            {k: out[k].contiguous() for k in keys},
            str(DST / filename),
            metadata={"format": "pt"},
        )

    index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
    (DST / "model.safetensors.index.json").write_text(json.dumps(index, indent=2) + "\n")

    # Post-write verification of the sharded layout.
    total = 0
    for filename, keys in split.filename_to_tensors.items():
        size = sum(out[k].numel() * out[k].element_size() for k in keys)
        assert size <= MAX_SHARD_BYTES or len(keys) == 1, f"{filename}: {size} bytes over budget"
        total += len(keys)
    assert total == 148, f"weight_map covers {total} tensors"
    print(f"wrote {len(split.filename_to_tensors)} shards, {total} tensors, {n_bf16} bfloat16")


if __name__ == "__main__":
    sys.exit(main())
