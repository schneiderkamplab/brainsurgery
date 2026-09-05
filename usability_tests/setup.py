#!/usr/bin/env python
"""Build the inputs and the hidden reference outputs for every surgery target.

    .venv/bin/python usability_tests/setup.py [--target NAME ...] [--data-root DIR] [--force]

Targets come from targets.py (gpt-2, olmo-1b, pythia-1b). Their base
checkpoints must already be under ``models/`` (see README, "Setup").
Everything generated is written under --data-root (default
``models/usability_tests``); this directory gets ``inputs/<target>`` and
``references/<target>`` symlinks:

    inputs/<target>/base/      base checkpoint files symlinked from models/<...>
                               plus config and tokenizer files
    inputs/<target>/ft1/       synthetic frozen-backbone fine-tune 1 (MLP tensors only)
    inputs/<target>/ft2/       synthetic frozen-backbone fine-tune 2
    inputs/<target>/lora/      PEFT-style LoRA adapter (adapter_model.safetensors,
                               adapter_config.json)
    references/<target>/T1..5  hidden reference outputs from solutions/<target>/P

References come from the Python baselines only; BrainSurgery is not used
here, so grading stays independent of the tool under test.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

from targets import TARGETS, TESTS

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
SEED = 20260905
DELTA_RANK = 32
BASE_FILE_SUFFIXES = (".safetensors", ".json", ".txt", ".model")


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    index = path / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        sd: dict[str, torch.Tensor] = {}
        for shard in sorted(set(weight_map.values())):
            sd.update(load_file(str(path / shard)))
        return sd
    return load_file(str(path / "model.safetensors"))


def make_finetune(base: dict[str, torch.Tensor], t: dict, seed: int) -> dict[str, torch.Tensor]:
    """Base plus a low-rank-dominated delta on every MLP weight and small noise on MLP biases.

    Each MLP weight gets a rank-32 delta with geometrically decaying singular
    values (2.0 * 0.85**k) plus tiny Gaussian noise, computed in float32 and
    cast back to the base dtype. Everything else is bit-identical to the base.
    """
    g = torch.Generator().manual_seed(seed)
    out = dict(base)
    for layer in range(t["n_layers"]):
        for rel, _shape in t["mlp_tensors"]:
            name = t["layer_fmt"].format(i=layer) + rel
            tensor = base[name].float()
            if tensor.ndim == 2:
                rows, cols = tensor.shape
                u, _ = torch.linalg.qr(torch.randn(rows, DELTA_RANK, generator=g))
                v, _ = torch.linalg.qr(torch.randn(cols, DELTA_RANK, generator=g))
                s = 2.0 * 0.85 ** torch.arange(DELTA_RANK, dtype=torch.float32)
                tensor = tensor + (u * s) @ v.T + 1e-5 * torch.randn(rows, cols, generator=g)
            else:
                tensor = tensor + 1e-3 * torch.randn(tensor.shape, generator=g)
            out[name] = tensor.to(base[name].dtype).contiguous()
    return out


def make_lora_adapter(t: dict, seed: int) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    lo = t["lora"]
    out: dict[str, torch.Tensor] = {}
    for layer in range(t["n_layers"]):
        for module in lo["modules"]:
            prefix = lo["peft_prefix"] + t["layer_fmt"].format(i=layer) + module
            out[f"{prefix}.lora_A.weight"] = torch.randn(lo["r"], lo["in"], generator=g) / lo["in"] ** 0.5
            out[f"{prefix}.lora_B.weight"] = 0.02 * torch.randn(lo["out"], lo["r"], generator=g)
    return out


def link_files(src_dir: Path, dst_dir: Path, *, weights: bool) -> None:
    dst_dir.mkdir(parents=True, exist_ok=True)
    for src in sorted(src_dir.iterdir()):
        if src.name.startswith(".") or not src.suffix.lower() in BASE_FILE_SUFFIXES:
            continue
        if not weights and (src.suffix == ".safetensors" or src.name.endswith(".index.json")):
            continue
        dst = dst_dir / src.name
        if not dst.exists():
            dst.symlink_to(src.resolve())


def make_read_only(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_file() and not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)


def ensure_link(link: Path, target: Path) -> None:
    if link.is_symlink():
        if Path(os.readlink(link)) == target:
            return
        link.unlink()
    elif link.exists():
        raise SystemExit(f"[setup] {link} exists and is not a symlink; remove it first")
    link.parent.mkdir(parents=True, exist_ok=True)
    link.symlink_to(target)


def run_reference(tname: str, test: str, cwd: Path, out_dir: Path) -> None:
    script = HERE / "solutions" / tname / "P" / f"{test}.py"
    print(f"[setup] {tname} reference {test}: {script.relative_to(HERE)} -> {out_dir}", flush=True)
    subprocess.run([sys.executable, str(script), str(out_dir)], cwd=cwd, check=True)


def build_target(tname: str, t: dict, data_root: Path, force: bool) -> None:
    model_dir = (REPO / t["model_dir"]).resolve()
    if not model_dir.exists():
        raise SystemExit(f"[setup] {tname}: base checkpoint directory missing: {model_dir} "
                         f"(download {t['hf_id']} there first)")
    root = data_root / tname
    inputs, references = root / "inputs", root / "references"
    inputs.mkdir(parents=True, exist_ok=True)
    references.mkdir(parents=True, exist_ok=True)
    ensure_link(HERE / "inputs" / tname, inputs)
    ensure_link(HERE / "references" / tname, references)

    link_files(model_dir, inputs / "base", weights=True)
    print(f"[setup] {tname}: inputs/base -> {model_dir}", flush=True)

    base = None
    for k, name in ((1, "ft1"), (2, "ft2")):
        path = inputs / name / "model.safetensors"
        link_files(model_dir, path.parent, weights=False)
        if force or not path.exists():
            print(f"[setup] {tname}: building synthetic fine-tune inputs/{name}", flush=True)
            base = base if base is not None else load_checkpoint(inputs / "base")
            if path.exists():
                path.chmod(0o644)
            save_file(make_finetune(base, t, SEED + k), str(path))

    lora_dir = inputs / "lora"
    lora_path = lora_dir / "adapter_model.safetensors"
    if force or not lora_path.exists():
        print(f"[setup] {tname}: building LoRA adapter inputs/lora", flush=True)
        lora_dir.mkdir(exist_ok=True)
        for old in (lora_path, lora_dir / "adapter_config.json"):
            if old.exists():
                old.chmod(0o644)
        save_file(make_lora_adapter(t, SEED + 3), str(lora_path))
        lo = t["lora"]
        (lora_dir / "adapter_config.json").write_text(json.dumps({
            "peft_type": "LORA",
            "base_model_name_or_path": t["hf_id"],
            "r": lo["r"],
            "lora_alpha": lo["alpha"],
            "lora_dropout": 0.0,
            "target_modules": [m.split(".")[-1] for m in lo["modules"]],
            "fan_in_fan_out": lo["fan_in_fan_out"],
            "bias": "none",
            "task_type": "CAUSAL_LM",
        }, indent=2) + "\n")

    for test in TESTS:
        out_dir = references / test
        if force and out_dir.exists():
            shutil.rmtree(out_dir)
        if not out_dir.exists():
            run_reference(tname, test, root, out_dir)

    # Inputs are shared by every sandbox through symlinks. Make every input file,
    # including the base checkpoint files they point at, read-only: a participant
    # that copies the directory with `cp -r` gets the symlinks and would otherwise
    # write straight through them into the shared checkpoint.
    make_read_only(inputs)
    make_read_only(model_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", action="append", choices=sorted(TARGETS), help="default: all targets")
    parser.add_argument("--data-root", type=Path, default=REPO / "models" / "usability_tests")
    parser.add_argument("--force", action="store_true", help="rebuild even if outputs exist")
    args = parser.parse_args()
    data_root = args.data_root.resolve()
    (HERE / "out").mkdir(exist_ok=True)
    for tname in args.target or sorted(TARGETS):
        build_target(tname, TARGETS[tname], data_root, args.force)
    print("[setup] done", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
