"""T2: remove attention head 5 from every layer of Pythia-1B (GPT-NeoX layout)."""
import os
import sys

import torch
from safetensors.torch import load_file, save_file

SRC = "inputs/base/model.safetensors"
DST = "out/T2/model.safetensors"
N_LAYERS = 16
N_HEADS = 8
HEAD_DIM = 256
HIDDEN = 2048
PRUNE_HEAD = 5
QKV_BLOCK = 3 * HEAD_DIM  # 768 rows per head: q | k | v


def keep_index(n_blocks: int, block: int, removed: int) -> torch.Tensor:
    """Indices along an axis of n_blocks*block entries, dropping block `removed`."""
    idx = [i for h in range(n_blocks) if h != removed for i in range(h * block, (h + 1) * block)]
    return torch.tensor(idx, dtype=torch.long)


def main() -> None:
    sd = load_file(SRC)
    n_in = len(sd)
    assert n_in == 244, f"expected 244 input tensors, got {n_in}"

    qkv_keep = keep_index(N_HEADS, QKV_BLOCK, PRUNE_HEAD)   # 5376 entries
    dense_keep = keep_index(N_HEADS, HEAD_DIM, PRUNE_HEAD)  # 1792 entries
    assert qkv_keep.numel() == 5376 and dense_keep.numel() == 1792
    # Sanity on the spec's row ranges: 0..3839 then 4608..6143.
    assert qkv_keep[3839].item() == 3839 and qkv_keep[3840].item() == 4608

    out = {}
    for name, t in sd.items():
        out[name] = t

    for i in range(N_LAYERS):
        p = f"gpt_neox.layers.{i}.attention."
        w = sd[p + "query_key_value.weight"]
        b = sd[p + "query_key_value.bias"]
        d = sd[p + "dense.weight"]
        assert tuple(w.shape) == (N_HEADS * QKV_BLOCK, HIDDEN), (name, w.shape)
        assert tuple(b.shape) == (N_HEADS * QKV_BLOCK,), b.shape
        assert tuple(d.shape) == (HIDDEN, N_HEADS * HEAD_DIM), d.shape
        out[p + "query_key_value.weight"] = w.index_select(0, qkv_keep).contiguous()
        out[p + "query_key_value.bias"] = b.index_select(0, qkv_keep).contiguous()
        out[p + "dense.weight"] = d.index_select(1, dense_keep).contiguous()

    # Required checks.
    l0 = "gpt_neox.layers.0.attention."
    assert tuple(out[l0 + "query_key_value.weight"].shape) == (5376, 2048)
    assert tuple(out[l0 + "query_key_value.bias"].shape) == (5376,)
    assert tuple(out[l0 + "dense.weight"].shape) == (2048, 1792)
    assert len(out) == 244, len(out)
    # dtype / unchanged-tensor checks.
    for name, t in out.items():
        assert t.dtype == sd[name].dtype, name
        if ".attention.query_key_value." not in name and ".attention.dense.weight" not in name:
            assert t.shape == sd[name].shape, name
    # Values: the kept rows must equal the source rows.
    assert torch.equal(out[l0 + "query_key_value.weight"][:3840], sd[l0 + "query_key_value.weight"][:3840])
    assert torch.equal(out[l0 + "query_key_value.weight"][3840:], sd[l0 + "query_key_value.weight"][4608:])
    assert torch.equal(out[l0 + "dense.weight"][:, 1280:], sd[l0 + "dense.weight"][:, 1536:])

    os.makedirs(os.path.dirname(DST), exist_ok=True)
    save_file(out, DST, metadata={"format": "pt"})
    back = load_file(DST)
    assert len(back) == 244, len(back)
    print(f"wrote {DST} with {len(back)} tensors")


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"CHECK FAILED: {e}", file=sys.stderr)
        sys.exit(1)
