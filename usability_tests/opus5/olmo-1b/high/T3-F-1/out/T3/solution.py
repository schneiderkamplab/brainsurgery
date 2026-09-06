"""T3: mixed-precision export with sharding for OLMo-1B-0724-hf.

Reads the float32 sharded checkpoint in ``inputs/base``, casts exactly the 112
per-layer projection matrices to bfloat16, leaves every other tensor untouched
in float32, and writes ``out/T3`` as a sharded safetensors checkpoint with an
index file.

Targeting uses an explicitly enumerated name set (16 layers x 7 projections),
never a regex, so it cannot drift onto ``model.embed_tokens`` or ``lm_head``.
Every required check is an assertion: the run aborts before any file is
written if one does not hold, and the output is re-read from disk and
re-verified against the input afterwards.

Tools: torch 2.14.0 (dtype cast), safetensors 0.5.3 (I/O), huggingface_hub
1.16.1 (its shard splitter is used to cross-check the packing).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import load_file, save_file

IN_DIR = Path("inputs/base")
OUT_DIR = Path("out/T3")
INDEX_NAME = "model.safetensors.index.json"
SHARD_GLOB = "model-*.safetensors"

NUM_LAYERS = 16
PROJECTIONS = (
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
)
# 256 MiB of tensor data per shard, file headers excluded.
MAX_SHARD_BYTES = 268_435_456

EXPECTED_TOTAL = 114
EXPECTED_BF16 = 112


class CheckFailed(AssertionError):
    """A required check did not hold."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def cast_targets() -> list[str]:
    """The exact names of the 112 projection matrices, in module order."""
    return [
        f"model.layers.{i}.{proj}.weight"
        for i in range(NUM_LAYERS)
        for proj in PROJECTIONS
    ]


def module_order(targets: list[str]) -> list[str]:
    """Canonical HuggingFace ``state_dict()`` order for this architecture."""
    return ["model.embed_tokens.weight", *targets, "lm_head.weight"]


def nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def load_input() -> dict[str, torch.Tensor]:
    weight_map: dict[str, str] = json.loads((IN_DIR / INDEX_NAME).read_text())["weight_map"]
    state: dict[str, torch.Tensor] = {}
    for shard in sorted(set(weight_map.values())):
        state.update(load_file(IN_DIR / shard))
    check(
        set(state) == set(weight_map),
        "tensors on disk do not match the input index weight_map",
    )
    return state


def pack(state: dict[str, torch.Tensor], order: list[str]) -> list[list[str]]:
    """Greedily group tensors into shards of at most ``MAX_SHARD_BYTES``.

    Tensors are visited in ``order``, so shard *n* precedes shard *n+1* in
    state-dict order.  A tensor larger than the budget lands in a shard of its
    own, since it neither fits beside a predecessor nor leaves room for a
    successor.
    """
    groups: list[list[str]] = []
    current: list[str] = []
    used = 0
    for name in order:
        size = nbytes(state[name])
        if current and used + size > MAX_SHARD_BYTES:
            groups.append(current)
            current, used = [], 0
        current.append(name)
        used += size
    if current:
        groups.append(current)
    return groups


def main() -> None:
    targets = cast_targets()
    check(len(targets) == EXPECTED_BF16, f"expected {EXPECTED_BF16} target names")
    check(len(set(targets)) == len(targets), "duplicate names in the target list")

    state = load_input()
    check(
        len(state) == EXPECTED_TOTAL,
        f"input has {len(state)} tensors, expected {EXPECTED_TOTAL}",
    )
    missing = [name for name in targets if name not in state]
    check(not missing, f"projection matrices absent from the input: {missing[:5]}")

    order = module_order(targets)
    check(
        set(order) == set(state),
        "module-order key list does not cover the input key set exactly",
    )

    # Build the output. Names are preserved; nothing is dropped -- this
    # checkpoint has no non-parameter buffers to drop.
    out_state: dict[str, torch.Tensor] = {}
    for name in order:
        tensor = state[name]
        check(
            tensor.dtype is torch.float32,
            f"{name} is {tensor.dtype} on input, expected float32",
        )
        out_state[name] = tensor.to(torch.bfloat16) if name in targets else tensor

    # ---- Required checks, all before anything is written. ----
    bf16 = [n for n, t in out_state.items() if t.dtype is torch.bfloat16]
    check(
        len(bf16) == EXPECTED_BF16,
        f"{len(bf16)} tensors are bfloat16, expected {EXPECTED_BF16}",
    )
    check(
        out_state["model.layers.0.self_attn.q_proj.weight"].dtype is torch.bfloat16,
        "model.layers.0.self_attn.q_proj.weight is not bfloat16",
    )
    check(
        out_state["model.embed_tokens.weight"].dtype is torch.float32,
        "model.embed_tokens.weight is not float32",
    )
    check(
        len(out_state) == EXPECTED_TOTAL,
        f"output has {len(out_state)} tensors, expected {EXPECTED_TOTAL}",
    )

    # ---- Additional checks: exact targeting, no value or shape drift. ----
    check(sorted(bf16) == sorted(targets), "the bfloat16 set is not exactly the 112 projections")
    for name, tensor in out_state.items():
        check(tensor.shape == state[name].shape, f"{name} changed shape")
        if name in targets:
            continue
        check(tensor.dtype is torch.float32, f"{name} should have stayed float32")
        check(
            torch.equal(tensor.view(torch.int32), state[name].view(torch.int32)),
            f"{name} was modified but should be bit-identical to the input",
        )

    # ---- Shard the output. ----
    groups = pack(out_state, order)
    shard_names = [
        f"model-{i + 1:05d}-of-{len(groups):05d}.safetensors" for i in range(len(groups))
    ]
    weight_map = {n: shard_names[i] for i, names in enumerate(groups) for n in names}
    total_size = sum(nbytes(t) for t in out_state.values())

    for filename, names in zip(shard_names, groups):
        total = sum(nbytes(out_state[n]) for n in names)
        check(
            total <= MAX_SHARD_BYTES or len(names) == 1,
            f"{filename} holds {total} bytes over budget in {len(names)} tensors",
        )
    check(set(weight_map) == set(out_state), "the weight_map does not cover every tensor")

    # Cross-check the packing against the splitter shipped with
    # huggingface_hub, comparing groupings rather than file names: that helper
    # does not always number its shards in state-dict order.
    reference = split_torch_state_dict_into_shards(
        out_state,
        filename_pattern="model{suffix}.safetensors",
        max_shard_size=MAX_SHARD_BYTES,
    )
    check(
        {frozenset(g) for g in groups}
        == {frozenset(g) for g in reference.filename_to_tensors.values()},
        "packing disagrees with huggingface_hub.split_torch_state_dict_into_shards",
    )
    check(
        total_size == reference.metadata["total_size"],
        "total_size disagrees with the reference splitter",
    )

    # ---- Write. Only checkpoint files are cleared; this directory also holds
    # the authored solution and report. ----
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in [*OUT_DIR.glob(SHARD_GLOB), OUT_DIR / INDEX_NAME]:
        stale.unlink(missing_ok=True)
    for filename, names in zip(shard_names, groups):
        save_file(
            {n: out_state[n].contiguous() for n in names},
            OUT_DIR / filename,
            metadata={"format": "pt"},
        )
    (OUT_DIR / INDEX_NAME).write_text(
        json.dumps(
            {"metadata": {"total_size": total_size}, "weight_map": weight_map},
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    verify(state, targets)
    print(
        f"wrote {len(out_state)} tensors ({EXPECTED_BF16} bfloat16) "
        f"into {len(groups)} shards under {OUT_DIR}"
    )


def verify(source: dict[str, torch.Tensor], targets: list[str]) -> None:
    """Re-read the written checkpoint and re-check it against the input."""
    weight_map: dict[str, str] = json.loads((OUT_DIR / INDEX_NAME).read_text())["weight_map"]
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        check((OUT_DIR / shard).is_file(), f"{shard} named in the index but missing on disk")
    on_disk = {p.name for p in OUT_DIR.glob(SHARD_GLOB)}
    check(on_disk == set(shards), f"stray shard files: {sorted(on_disk - set(shards))}")

    reloaded: dict[str, torch.Tensor] = {}
    for shard in shards:
        with safe_open(OUT_DIR / shard, framework="pt") as handle:
            names = list(handle.keys())
            check(
                all(weight_map[n] == shard for n in names),
                f"{shard} holds tensors the index maps elsewhere",
            )
            tensors = {name: handle.get_tensor(name) for name in names}
        total = sum(nbytes(t) for t in tensors.values())
        check(
            total <= MAX_SHARD_BYTES or len(names) == 1,
            f"{shard} holds {total} bytes of tensor data, over the 256 MiB budget",
        )
        reloaded.update(tensors)

    check(
        len(reloaded) == EXPECTED_TOTAL,
        f"reloaded {len(reloaded)} tensors, expected {EXPECTED_TOTAL}",
    )
    check(set(reloaded) == set(weight_map), "index weight_map and shard contents disagree")
    check(set(reloaded) == set(source), "tensor names changed")

    bf16 = [n for n, t in reloaded.items() if t.dtype is torch.bfloat16]
    check(
        sorted(bf16) == sorted(targets),
        f"{len(bf16)} bfloat16 tensors on disk, expected exactly the 112 projections",
    )
    check(
        reloaded["model.layers.0.self_attn.q_proj.weight"].dtype is torch.bfloat16,
        "model.layers.0.self_attn.q_proj.weight is not bfloat16 on disk",
    )
    check(
        reloaded["model.embed_tokens.weight"].dtype is torch.float32,
        "model.embed_tokens.weight is not float32 on disk",
    )

    for name, tensor in reloaded.items():
        original = source[name]
        check(tensor.shape == original.shape, f"{name} has shape {tuple(tensor.shape)} on disk")
        if name in targets:
            check(
                torch.equal(
                    tensor.view(torch.int16), original.to(torch.bfloat16).view(torch.int16)
                ),
                f"{name} is not bit-exactly the round-to-nearest-even bfloat16 cast",
            )
        else:
            check(
                tensor.dtype is torch.float32
                and torch.equal(tensor.view(torch.int32), original.view(torch.int32)),
                f"{name} is not bit-identical to the input",
            )
    print(f"verified {len(reloaded)} tensors across {len(shards)} shards on disk")


if __name__ == "__main__":
    try:
        main()
    except CheckFailed as exc:
        print(f"CHECK FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
