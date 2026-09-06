#!/usr/bin/env python
"""T5: fold a PEFT LoRA adapter into Pythia-1B base weights, export sharded.

Operates directly on the checkpoint files (safetensors + torch), never
instantiating the model: transformers 5.x does not register GPTNeoX's
`attention.bias`, `attention.masked_bias` and `rotary_emb.inv_freq` buffers as
persistent state, so a `peft.merge_and_unload()` + `save_pretrained()` route
would drop 48 of the 244 tensors.

Every "Required check" from TASK.md is asserted before anything is written,
and the written output is re-read and re-verified afterwards.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors.torch import load_file, save_file

BASE = Path("inputs/base/model.safetensors")
LORA = Path("inputs/lora/adapter_model.safetensors")
LORA_CFG = Path("inputs/lora/adapter_config.json")
OUT_DIR = Path("out/T5")
INDEX_NAME = "model.safetensors.index.json"

PEFT_PREFIX = "base_model.model."
MAX_SHARD_BYTES = 512 * 1024 * 1024  # 512 MiB of tensor data per shard

EXPECT_PAIRS = 16
EXPECT_TENSORS = 244
PROBE_KEY = "gpt_neox.layers.0.attention.query_key_value.weight"
PROBE_SHAPE = (6144, 2048)


class CheckFailed(AssertionError):
    """A required check did not hold."""


def check(condition: bool, message: str) -> None:
    if not condition:
        raise CheckFailed(message)


def main() -> None:
    # ---- load -----------------------------------------------------------
    cfg = json.loads(LORA_CFG.read_text())
    r = int(cfg["r"])
    alpha = float(cfg["lora_alpha"])
    fan_in_fan_out = bool(cfg["fan_in_fan_out"])
    check(r > 0, f"adapter r must be positive, got {r}")
    scale = alpha / r

    base = load_file(BASE)  # ordered dict, safetensors key order
    lora = load_file(LORA)
    print(f"base tensors: {len(base)}  adapter tensors: {len(lora)}")
    print(f"r={r} alpha={alpha} scale={scale} fan_in_fan_out={fan_in_fan_out}")

    # ---- pair lora_A / lora_B by module ---------------------------------
    pairs: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in lora.items():
        for side in ("lora_A", "lora_B"):
            marker = f".{side}."
            if marker in name:
                module = name.split(marker, 1)[0]
                check(
                    module.startswith(PEFT_PREFIX),
                    f"adapter tensor {name!r} lacks the PEFT prefix {PEFT_PREFIX!r}",
                )
                module = module[len(PEFT_PREFIX) :]
                slot = pairs.setdefault(module, {})
                check(side not in slot, f"duplicate {side} for module {module!r}")
                slot[side] = tensor
                break
        else:
            raise CheckFailed(f"adapter tensor {name!r} is neither lora_A nor lora_B")

    check(
        len(pairs) == EXPECT_PAIRS,
        f"expected {EXPECT_PAIRS} adapter pairs, found {len(pairs)}: {sorted(pairs)}",
    )
    for module, slot in pairs.items():
        check(
            set(slot) == {"lora_A", "lora_B"},
            f"module {module!r} has an incomplete pair: {sorted(slot)}",
        )

    # ---- merge ----------------------------------------------------------
    merged = 0
    for module in sorted(pairs):
        a = pairs[module]["lora_A"]
        b = pairs[module]["lora_B"]
        key = f"{module}.weight"
        check(key in base, f"adapter targets {module!r} but base has no tensor {key!r}")

        check(a.ndim == 2 and b.ndim == 2, f"{module}: LoRA factors must be 2-D")
        check(
            a.shape[0] == r and b.shape[1] == r,
            f"{module}: factor ranks {tuple(b.shape)} @ {tuple(a.shape)} disagree with r={r}",
        )

        w = base[key]
        orig_dtype, orig_shape = w.dtype, w.shape

        # fan_in_fan_out=False: base is [out, in] like the factors, so B @ A
        # ([out, r] @ [r, in]) is added as-is.  True would mean base is
        # [in, out] (Conv1D layout) and the delta needs transposing.
        delta = (b.to(torch.float32) @ a.to(torch.float32)) * scale
        if fan_in_fan_out:
            delta = delta.T
        check(
            delta.shape == orig_shape,
            f"{key}: delta {tuple(delta.shape)} does not match base {tuple(orig_shape)}",
        )

        base[key] = (w.to(torch.float32) + delta).to(orig_dtype).contiguous()
        check(base[key].shape == orig_shape, f"{key}: shape changed during merge")
        check(base[key].dtype == orig_dtype, f"{key}: dtype changed during merge")
        merged += 1

    check(merged == EXPECT_PAIRS, f"merged {merged} modules, expected {EXPECT_PAIRS}")
    print(f"merged {merged} adapter pairs")

    # ---- required checks, before writing --------------------------------
    offenders = [k for k in base if "lora_" in k]
    check(not offenders, f"adapter tensor names leaked into the output: {offenders}")
    check(PROBE_KEY in base, f"{PROBE_KEY!r} missing from the output")
    check(
        tuple(base[PROBE_KEY].shape) == PROBE_SHAPE,
        f"{PROBE_KEY} has shape {tuple(base[PROBE_KEY].shape)}, expected {PROBE_SHAPE}",
    )
    check(
        len(base) == EXPECT_TENSORS,
        f"output has {len(base)} tensors, expected {EXPECT_TENSORS}",
    )

    # ---- shard and write ------------------------------------------------
    # Clear only checkpoint files from a previous run.  Do NOT rmtree OUT_DIR:
    # this script lives in it.
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for stale in [*OUT_DIR.glob("*.safetensors"), OUT_DIR / INDEX_NAME]:
        stale.unlink(missing_ok=True)

    split = split_torch_state_dict_into_shards(
        base, filename_pattern="model{suffix}.safetensors", max_shard_size=MAX_SHARD_BYTES
    )
    for filename, keys in split.filename_to_tensors.items():
        shard = {k: base[k].contiguous() for k in keys}
        save_file(shard, OUT_DIR / filename, metadata={"format": "pt"})

    index = {"metadata": split.metadata, "weight_map": split.tensor_to_filename}
    (OUT_DIR / INDEX_NAME).write_text(json.dumps(index, indent=2) + "\n")
    print(f"wrote {len(split.filename_to_tensors)} shards + index to {OUT_DIR}/")

    verify(base)


def verify(expected: dict[str, torch.Tensor]) -> None:
    """Re-read what was written and re-assert the requirements against it."""
    index = json.loads((OUT_DIR / INDEX_NAME).read_text())
    weight_map = index["weight_map"]

    seen: dict[str, torch.Tensor] = {}
    shard_files = sorted(p.name for p in OUT_DIR.glob("*.safetensors"))
    for filename in shard_files:
        shard = load_file(OUT_DIR / filename)
        payload = sum(t.numel() * t.element_size() for t in shard.values())
        check(
            payload <= MAX_SHARD_BYTES,
            f"{filename} holds {payload} bytes of tensor data, over the {MAX_SHARD_BYTES} limit",
        )
        if len(shard) > 1:
            check(
                all(t.numel() * t.element_size() <= MAX_SHARD_BYTES for t in shard.values()),
                f"{filename} mixes an oversized tensor with others",
            )
        for k, t in shard.items():
            check(k not in seen, f"tensor {k!r} appears in more than one shard")
            check(weight_map.get(k) == filename, f"weight_map does not map {k!r} to {filename}")
            seen[k] = t
        print(f"  {filename}: {len(shard)} tensors, {payload} bytes")

    check(
        set(weight_map) == set(seen),
        "weight_map and the shard contents disagree on the tensor set",
    )
    check(
        set(weight_map.values()) == set(shard_files),
        f"weight_map references {sorted(set(weight_map.values()))}, shards are {shard_files}",
    )
    check(
        len(seen) == EXPECT_TENSORS, f"round-trip has {len(seen)} tensors, expected {EXPECT_TENSORS}"
    )
    check(not [k for k in seen if "lora_" in k], "adapter tensor names found in the written output")
    check(
        tuple(seen[PROBE_KEY].shape) == PROBE_SHAPE,
        f"{PROBE_KEY} round-tripped with shape {tuple(seen[PROBE_KEY].shape)}",
    )
    check(set(seen) == set(expected), "written tensor names differ from the merged state dict")
    for k, t in seen.items():
        check(t.shape == expected[k].shape, f"{k}: shape changed on write")
        check(t.dtype == expected[k].dtype, f"{k}: dtype changed on write")
        check(torch.equal(t, expected[k]), f"{k}: values changed on write")

    # independent re-derivation of the merge from the originals
    original = load_file(BASE)
    cfg = json.loads(LORA_CFG.read_text())
    scale = float(cfg["lora_alpha"]) / int(cfg["r"])
    lora = load_file(LORA)
    changed = 0
    for k in original:
        if torch.equal(original[k], seen[k]):
            continue
        changed += 1
        module = k[: -len(".weight")]
        a = lora[f"{PEFT_PREFIX}{module}.lora_A.weight"].to(torch.float32)
        b = lora[f"{PEFT_PREFIX}{module}.lora_B.weight"].to(torch.float32)
        want = (original[k].to(torch.float32) + scale * (b @ a)).to(original[k].dtype)
        err = torch.linalg.norm(seen[k].to(torch.float32) - want.to(torch.float32)).item()
        ref = torch.linalg.norm(want.to(torch.float32)).item()
        check(err / ref <= 1e-3, f"{k}: relative Frobenius error {err / ref:.2e} exceeds 1e-3")
    check(
        changed == EXPECT_PAIRS,
        f"{changed} tensors differ from the base, expected exactly {EXPECT_PAIRS}",
    )
    print(f"verified: {changed} tensors merged, {len(seen) - changed} bit-identical to the base")
    print("OK")


if __name__ == "__main__":
    main()
