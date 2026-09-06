"""T2: structured attention-head pruning for GPT-2 (124M).

Removes head 5 from every layer at the checkpoint level.

GPT-2 uses Conv1D, i.e. weights are stored as [in, out] (the transpose of
nn.Linear). So for a projection that *produces* head-split activations
(c_attn) the head blocks live along the columns (dim 1); for the projection
that *consumes* them (c_proj) they live along the rows (dim 0).
"""

from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SRC = Path("inputs/base/model.safetensors")
DST_DIR = Path("out/T2")
DST = DST_DIR / "model.safetensors"

N_LAYERS = 12
N_HEADS = 12
HEAD_DIM = 64
HIDDEN = N_HEADS * HEAD_DIM  # 768
PRUNE_HEAD = 5
N_TENSORS = 160


def keep_indices(n_segments: int) -> list[int]:
    """Column/row indices to keep, for a tensor made of `n_segments`
    consecutive 768-wide segments each split into 12 head blocks."""
    idx: list[int] = []
    for seg in range(n_segments):
        base = seg * HIDDEN
        for head in range(N_HEADS):
            if head == PRUNE_HEAD:
                continue
            idx.extend(range(base + head * HEAD_DIM, base + (head + 1) * HEAD_DIM))
    return idx


def ranges_to_indices(ranges: list[tuple[int, int]]) -> list[int]:
    """Inclusive ranges -> flat index list."""
    out: list[int] = []
    for lo, hi in ranges:
        out.extend(range(lo, hi + 1))
    return out


# The index lists spelled out in the task, used as an independent cross-check
# of the computed ones (order matters).
SPEC_QKV = ranges_to_indices(
    [(0, 319), (384, 767), (768, 1087), (1152, 1535), (1536, 1855), (1920, 2303)]
)
SPEC_PROJ = ranges_to_indices([(0, 319), (384, 767)])

QKV_KEEP = keep_indices(3)  # c_attn: fused [q | k | v]
PROJ_KEEP = keep_indices(1)  # c_proj: single 768-wide input side

assert QKV_KEEP == SPEC_QKV, "computed c_attn keep-indices disagree with the task spec"
assert PROJ_KEEP == SPEC_PROJ, "computed c_proj keep-indices disagree with the task spec"
assert len(QKV_KEEP) == 2112, f"expected 2112 kept qkv indices, got {len(QKV_KEEP)}"
assert len(PROJ_KEEP) == 704, f"expected 704 kept proj indices, got {len(PROJ_KEEP)}"

qkv_index = torch.tensor(QKV_KEEP, dtype=torch.long)
proj_index = torch.tensor(PROJ_KEEP, dtype=torch.long)


def fail(msg: str) -> None:
    raise SystemExit(f"FAIL: {msg}")


if not SRC.is_file():
    fail(f"input checkpoint not found: {SRC}")

state = load_file(str(SRC))
print(f"loaded {len(state)} tensors from {SRC}")
if len(state) != N_TENSORS:
    fail(f"input has {len(state)} tensors, expected {N_TENSORS}")

# --- prune -----------------------------------------------------------------
out: dict[str, torch.Tensor] = {}
touched: list[str] = []

for name, tensor in state.items():
    out[name] = tensor

for i in range(N_LAYERS):
    w_name = f"h.{i}.attn.c_attn.weight"
    b_name = f"h.{i}.attn.c_attn.bias"
    p_name = f"h.{i}.attn.c_proj.weight"

    for name, want in ((w_name, (HIDDEN, 3 * HIDDEN)), (b_name, (3 * HIDDEN,)), (p_name, (HIDDEN, HIDDEN))):
        if name not in state:
            fail(f"missing expected tensor {name}")
        got = tuple(state[name].shape)
        if got != want:
            fail(f"{name} has input shape {list(got)}, expected {list(want)}")

    # c_attn: heads are column blocks inside each of the three 768-wide
    # q/k/v segments -> select along dim 1 (Conv1D [in, out]).
    out[w_name] = state[w_name].index_select(1, qkv_index).contiguous()
    # bias follows the columns of c_attn.weight -> same index list, dim 0.
    out[b_name] = state[b_name].index_select(0, qkv_index).contiguous()
    # c_proj: heads are row blocks on the input side -> select along dim 0.
    out[p_name] = state[p_name].index_select(0, proj_index).contiguous()

    touched.extend([w_name, b_name, p_name])

    for name, src in ((w_name, state[w_name]), (b_name, state[b_name]), (p_name, state[p_name])):
        if out[name].dtype != src.dtype:
            fail(f"{name} changed dtype {src.dtype} -> {out[name].dtype}")

print(f"pruned head {PRUNE_HEAD} from {N_LAYERS} layers ({len(touched)} tensors rewritten)")

# --- required checks, before writing ---------------------------------------
required = {
    "h.0.attn.c_attn.weight": (768, 2112),
    "h.0.attn.c_attn.bias": (2112,),
    "h.0.attn.c_proj.weight": (704, 768),
}
for name, want in required.items():
    if name not in out:
        fail(f"required tensor {name} missing from output")
    got = tuple(out[name].shape)
    if got != want:
        fail(f"{name} has shape {list(got)}, expected {list(want)}")
    print(f"check ok: {name} {list(got)}")

if len(out) != N_TENSORS:
    fail(f"output has {len(out)} tensors, expected exactly {N_TENSORS}")
print(f"check ok: output has exactly {len(out)} tensors")

# Every layer, not just layer 0.
for i in range(N_LAYERS):
    for name, want in (
        (f"h.{i}.attn.c_attn.weight", (768, 2112)),
        (f"h.{i}.attn.c_attn.bias", (2112,)),
        (f"h.{i}.attn.c_proj.weight", (704, 768)),
    ):
        got = tuple(out[name].shape)
        if got != want:
            fail(f"{name} has shape {list(got)}, expected {list(want)}")

# Names are unchanged, and nothing outside the head-bearing tensors moved.
if set(out) != set(state):
    fail("output key set differs from input key set")
untouched = set(state) - set(touched)
for name in untouched:
    if out[name].shape != state[name].shape:
        fail(f"{name} should have been left alone but changed shape")
    if out[name] is not state[name]:
        fail(f"{name} should have been left alone but was replaced")
print(f"check ok: {len(untouched)} tensors left untouched, key set identical")

# Spot-check that the kept values really are the original ones in order.
ref_w = state["h.0.attn.c_attn.weight"]
if not torch.equal(out["h.0.attn.c_attn.weight"][:, 320:704], ref_w[:, 384:768]):
    fail("kept c_attn columns are not in the expected order")
ref_p = state["h.0.attn.c_proj.weight"]
if not torch.equal(out["h.0.attn.c_proj.weight"][320:704], ref_p[384:768]):
    fail("kept c_proj rows are not in the expected order")
print("check ok: kept slices match the source in order")

# --- write -----------------------------------------------------------------
DST_DIR.mkdir(parents=True, exist_ok=True)
save_file(out, str(DST), metadata={"format": "pt"})
print(f"wrote {DST}")

# --- verify what landed on disk --------------------------------------------
back = load_file(str(DST))
if len(back) != N_TENSORS:
    fail(f"written file has {len(back)} tensors, expected {N_TENSORS}")
if set(back) != set(state):
    fail("written file key set differs from the input key set")
for name, tensor in back.items():
    if tuple(tensor.shape) != tuple(out[name].shape):
        fail(f"{name}: shape on disk {list(tensor.shape)} != in memory {list(out[name].shape)}")
    if tensor.dtype != out[name].dtype:
        fail(f"{name}: dtype on disk {tensor.dtype} != in memory {out[name].dtype}")
    if not torch.equal(tensor, out[name]):
        fail(f"{name}: values on disk differ from what was computed")
print(f"verified {len(back)} tensors on disk; all checks passed")
