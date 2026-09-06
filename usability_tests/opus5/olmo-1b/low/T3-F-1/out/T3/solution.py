"""T3: mixed-precision sharded export of OLMo-1B-0724-hf.

Plain torch + safetensors + huggingface_hub sharding helper (the same helper
transformers' save_pretrained uses), so the shard layout and index format are
the canonical HF ones.
"""
import json
import os
import re
import shutil

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file

IN_DIR = "inputs/base"
OUT_DIR = "out/T3"
MAX_SHARD = 256 * 1024 * 1024

CAST = re.compile(
    r"^model\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)\.weight$"
)


def main() -> None:
    index = json.load(open(os.path.join(IN_DIR, "model.safetensors.index.json")))
    weight_map = index["weight_map"]
    shards = {}
    state = {}
    for name in weight_map:  # preserve the input index ordering
        f = weight_map[name]
        if f not in shards:
            shards[f] = load_file(os.path.join(IN_DIR, f))
        state[name] = shards[f][name]

    assert len(state) == 114, f"expected 114 input tensors, got {len(state)}"

    out = {}
    for name, t in state.items():
        assert t.dtype == torch.float32, f"{name} is {t.dtype}, expected float32"
        out[name] = t.to(torch.bfloat16) if CAST.match(name) else t.clone()

    # Required checks, before writing anything.
    bf16 = [n for n, t in out.items() if t.dtype == torch.bfloat16]
    assert len(bf16) == 112, f"expected 112 bfloat16 tensors, got {len(bf16)}"
    assert out["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
    assert out["model.embed_tokens.weight"].dtype == torch.float32
    assert all(
        t.dtype == torch.float32 for n, t in out.items() if n not in set(bf16)
    ), "non-cast tensors must stay float32"
    assert len(out) == 114, f"expected 114 output tensors, got {len(out)}"
    assert set(out) == set(weight_map), "tensor names changed"

    split = split_torch_state_dict_into_shards(out, max_shard_size=MAX_SHARD)
    for filename, tensors in split.filename_to_tensors.items():
        payload = {k: out[k].contiguous() for k in tensors}
        nbytes = sum(v.numel() * v.element_size() for v in payload.values())
        assert len(payload) == 1 or nbytes <= MAX_SHARD, f"{filename} over budget: {nbytes}"
        save_file(payload, os.path.join(OUT_DIR, filename), metadata={"format": "pt"})

    if split.is_sharded:
        idx = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
    else:
        raise AssertionError("output must be sharded")
    with open(os.path.join(OUT_DIR, "model.safetensors.index.json"), "w") as fh:
        json.dump(idx, fh, indent=2, sort_keys=True)

    # Verify what landed on disk.
    written = json.load(open(os.path.join(OUT_DIR, "model.safetensors.index.json")))["weight_map"]
    assert len(written) == 114, f"index maps {len(written)} tensors"
    seen = {}
    for f in sorted(set(written.values())):
        seen.update(load_file(os.path.join(OUT_DIR, f)))
    assert len(seen) == 114
    assert sum(1 for t in seen.values() if t.dtype == torch.bfloat16) == 112
    assert seen["model.embed_tokens.weight"].dtype == torch.float32
    assert seen["model.layers.0.self_attn.q_proj.weight"].dtype == torch.bfloat16
    for name, t in seen.items():
        ref = state[name]
        expect = ref.to(torch.bfloat16) if CAST.match(name) else ref
        assert t.dtype == expect.dtype and torch.equal(t, expect), f"value mismatch: {name}"

    for extra in ("config.json", "generation_config.json", "special_tokens_map.json",
                  "tokenizer_config.json", "tokenizer.json"):
        src = os.path.join(IN_DIR, extra)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(OUT_DIR, extra))

    print(f"OK: {len(written)} tensors, 112 bf16, {len(set(written.values()))} shards")


if __name__ == "__main__":
    main()
