"""T5: fold a PEFT LoRA adapter into GPT-2 base weights and export sharded."""

import json
import re
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

SANDBOX = Path(__file__).resolve().parents[2]
BASE = SANDBOX / "inputs" / "base" / "model.safetensors"
LORA_DIR = SANDBOX / "inputs" / "lora"
OUT = SANDBOX / "out" / "T5"
MAX_SHARD_BYTES = 100 * 1024 * 1024  # 100 MiB of tensor data per shard

LORA_A = re.compile(r"^base_model\.model\.(.+)\.lora_A\.weight$")


def main() -> None:
    cfg = json.loads((LORA_DIR / "adapter_config.json").read_text())
    scale = cfg["lora_alpha"] / cfg["r"]
    fan_in_fan_out = bool(cfg["fan_in_fan_out"])

    base = load_file(str(BASE))
    adapter = load_file(str(LORA_DIR / "adapter_model.safetensors"))
    n_base = len(base)

    merged = 0
    for a_key in sorted(adapter):
        m = LORA_A.match(a_key)
        if m is None:
            continue
        stem = m.group(1)
        b_key = f"base_model.model.{stem}.lora_B.weight"
        if b_key not in adapter:
            raise RuntimeError(f"missing lora_B partner for {a_key}")
        target = f"{stem}.weight"
        if target not in base:
            raise RuntimeError(f"adapter target {target!r} not in base checkpoint")

        A = adapter[a_key].to(torch.float32)
        B = adapter[b_key].to(torch.float32)
        delta = scale * (B @ A)  # [out, in], nn.Linear convention
        if fan_in_fan_out:
            delta = delta.T  # base is Conv1D [in, out]

        W = base[target]
        if W.shape != delta.shape:
            raise RuntimeError(f"shape mismatch for {target}: {tuple(W.shape)} vs {tuple(delta.shape)}")
        if W.dtype != torch.float32:
            raise RuntimeError(f"expected float32 base weight for {target}, got {W.dtype}")
        base[target] = (W.to(torch.float32) + delta).contiguous()
        merged += 1

    # --- required checks -------------------------------------------------
    if merged != 12:
        raise RuntimeError(f"expected exactly 12 adapter pairs merged, got {merged}")
    leaked = [k for k in base if "lora_" in k]
    if leaked:
        raise RuntimeError(f"adapter/intermediate tensors leaked into output: {leaked}")
    probe = "h.0.attn.c_attn.weight"
    if tuple(base[probe].shape) != (768, 2304):
        raise RuntimeError(f"{probe} has shape {tuple(base[probe].shape)}, expected (768, 2304)")
    if len(base) != 160:
        raise RuntimeError(f"output has {len(base)} tensors, expected 160")
    if len(base) != n_base:
        raise RuntimeError("tensor count changed relative to the base checkpoint")

    # --- sharded export --------------------------------------------------
    shards: list[list[str]] = []
    cur: list[str] = []
    cur_bytes = 0
    for name, t in base.items():
        nbytes = t.numel() * t.element_size()
        if cur and cur_bytes + nbytes > MAX_SHARD_BYTES:
            shards.append(cur)
            cur, cur_bytes = [], 0
        cur.append(name)
        cur_bytes += nbytes
        if cur_bytes > MAX_SHARD_BYTES:  # single oversized tensor: seal it alone
            shards.append(cur)
            cur, cur_bytes = [], 0
    if cur:
        shards.append(cur)

    total = len(shards)
    OUT.mkdir(parents=True, exist_ok=True)
    for stale in OUT.glob("model*.safetensors"):
        stale.unlink()
    (OUT / "model.safetensors.index.json").unlink(missing_ok=True)

    weight_map: dict[str, str] = {}
    total_size = 0
    for i, names in enumerate(shards, start=1):
        fname = f"model-{i:05d}-of-{total:05d}.safetensors"
        payload = {n: base[n].contiguous() for n in names}
        size = sum(t.numel() * t.element_size() for t in payload.values())
        if size > MAX_SHARD_BYTES and len(payload) > 1:
            raise RuntimeError(f"shard {fname} holds {size} bytes over budget with {len(payload)} tensors")
        total_size += size
        save_file(payload, str(OUT / fname), metadata={"format": "pt"})
        for n in names:
            weight_map[n] = fname

    if len(weight_map) != 160:
        raise RuntimeError(f"weight_map covers {len(weight_map)} tensors, expected 160")
    (OUT / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {"total_size": total_size}, "weight_map": weight_map}, indent=2) + "\n"
    )
    print(f"merged {merged} adapter pairs; wrote {len(weight_map)} tensors in {total} shards to {OUT}")


if __name__ == "__main__":
    main()
