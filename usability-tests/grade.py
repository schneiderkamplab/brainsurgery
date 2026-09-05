#!/usr/bin/env python
"""Grade a participant's output against the hidden reference.

    .venv/bin/python usability-tests/grade.py T3 --target gpt-2 [--out PATH] [--reference PATH] [--json] [--write FILE]

Independent of brainsurgery: it only needs torch and safetensors. The output
may be a single .safetensors file, a torch .pt/.pth/.bin file, or a directory
(sharded with model.safetensors.index.json, or holding model.safetensors).

Checks, in order: loadability, sharding requirement, exact key set, per-tensor
shape and dtype, values. Values must be bit-exact except for tensors produced
by floating-point arithmetic whose result depends on operation order or BLAS
implementation (the merged MLP tensors of T4 and the merged c_attn weights of
T5); those are compared with a small relative Frobenius tolerance.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

import torch
from safetensors.torch import load_file

sys.path.insert(0, str(Path(__file__).resolve().parent))
from targets import TARGETS, TESTS, esc  # noqa: E402

HERE = Path(__file__).resolve().parent
MIB = 1024 * 1024

# Relative Frobenius tolerance for arithmetic results (association order, BLAS),
# by output dtype: half-precision outputs also absorb the final rounding.
ARITH_REL_TOL = {"full": 1e-5, "half": 1e-3}
HALF = (torch.float16, torch.bfloat16)


def task_rules(test: str, target: dict) -> dict:
    """Sharding requirement and tolerant-tensor pattern for one (test, target)."""
    lre = target["layer_re"]
    if test == "T3":
        return {"sharded": True, "shard_limit": target["shard_t3"][1], "tolerant": None}
    if test == "T4":
        alt = "|".join(esc(n) for n, _ in target["mlp_tensors"])
        return {"sharded": False, "shard_limit": None, "tolerant": re.compile(f"{lre}({alt})")}
    if test == "T5":
        alt = "|".join(esc(m) for m in target["lora"]["modules"])
        return {"sharded": True, "shard_limit": target["shard_t5"][1],
                "tolerant": re.compile(f"{lre}({alt})\\.weight")}
    return {"sharded": False, "shard_limit": None, "tolerant": None}

@dataclass
class Report:
    task: str
    passed: bool = True
    findings: list[str] = field(default_factory=list)
    metrics: dict[str, float | int | str] = field(default_factory=dict)

    def fail(self, message: str) -> None:
        self.passed = False
        self.findings.append(message)


def load_checkpoint(path: Path, report: Report, *, require_sharded: bool, shard_limit: int | None) -> dict[str, torch.Tensor] | None:
    if not path.exists():
        report.fail(f"output path does not exist: {path}")
        return None
    if path.is_file():
        if require_sharded:
            report.fail("expected a sharded directory output, got a single file")
        if path.suffix == ".safetensors":
            return load_file(str(path))
        if path.suffix in {".pt", ".pth", ".bin"}:
            report.findings.append("note: torch file loaded; task asked for safetensors")
            return torch.load(path, map_location="cpu", weights_only=True)
        report.fail(f"unsupported output file type: {path.suffix}")
        return None

    index = path / "model.safetensors.index.json"
    single = path / "model.safetensors"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        sd: dict[str, torch.Tensor] = {}
        for shard_name in sorted(set(weight_map.values())):
            shard_path = path / shard_name
            if not shard_path.exists():
                report.fail(f"index references missing shard: {shard_name}")
                continue
            shard = load_file(str(shard_path))
            payload = sum(t.numel() * t.element_size() for t in shard.values())
            # HF convention: a single tensor larger than the budget sits alone in its shard.
            if require_sharded and shard_limit and payload > shard_limit and len(shard) > 1:
                report.fail(f"shard {shard_name} holds {payload} bytes of tensor data > {shard_limit}")
            for name, tensor in shard.items():
                if weight_map.get(name) != shard_name:
                    report.fail(f"index maps {name} to {weight_map.get(name)} but it is in {shard_name}")
                sd[name] = tensor
        missing = sorted(set(weight_map) - set(sd))
        if missing:
            report.fail(f"index lists {len(missing)} tensors absent from shards, e.g. {missing[:3]}")
        report.metrics["shards"] = len(set(weight_map.values()))
        if not require_sharded:
            report.findings.append("note: sharded output loaded; task asked for a single file")
        return sd
    if single.exists():
        if require_sharded:
            report.fail("expected sharded output with model.safetensors.index.json, found a single model.safetensors")
        return load_file(str(single))
    shards = sorted(path.glob("*.safetensors"))
    if shards:
        report.fail(f"directory has {len(shards)} safetensors files but no model.safetensors.index.json")
        sd = {}
        for shard_path in shards:
            sd.update(load_file(str(shard_path)))
        return sd
    report.fail(f"no checkpoint found under {path}")
    return None


def rel_fro(left: torch.Tensor, right: torch.Tensor) -> float:
    denom = right.norm().item()
    return (left - right).norm().item() / denom if denom > 0 else (left - right).norm().item()


def compare(rules: dict, out: dict[str, torch.Tensor], ref: dict[str, torch.Tensor], report: Report) -> None:
    missing = sorted(set(ref) - set(out))
    extra = sorted(set(out) - set(ref))
    if missing:
        report.fail(f"{len(missing)} expected tensors missing, e.g. {missing[:5]}")
    if extra:
        report.fail(f"{len(extra)} unexpected tensors present, e.g. {extra[:5]}")
    report.metrics["tensors_expected"] = len(ref)
    report.metrics["tensors_found"] = len(out)

    shape_dtype_bad: list[str] = []
    value_bad: list[str] = []
    tolerant = rules["tolerant"]
    max_rel = 0.0
    for name in sorted(set(ref) & set(out)):
        o, r = out[name], ref[name]
        if tuple(o.shape) != tuple(r.shape) or o.dtype != r.dtype:
            shape_dtype_bad.append(f"{name}: got {tuple(o.shape)} {o.dtype}, want {tuple(r.shape)} {r.dtype}")
            continue
        if tolerant is not None and tolerant.fullmatch(name):
            rel = rel_fro(o.float(), r.float())
            max_rel = max(max_rel, rel)
            tol = ARITH_REL_TOL["half" if r.dtype in HALF else "full"]
            if rel > tol:
                value_bad.append(f"{name}: relative error {rel:.3e} > {tol}")
            continue
        if not torch.equal(o, r):
            diff = (o.float() - r.float()).abs().max().item()
            value_bad.append(f"{name}: values differ (max abs diff {diff:.3e})")

    if tolerant is not None:
        report.metrics["max_rel_error_tolerant"] = max_rel
    for line in shape_dtype_bad[:10]:
        report.fail(line)
    if len(shape_dtype_bad) > 10:
        report.fail(f"... {len(shape_dtype_bad) - 10} more shape/dtype mismatches")
    for line in value_bad[:10]:
        report.fail(line)
    if len(value_bad) > 10:
        report.fail(f"... {len(value_bad) - 10} more value mismatches")
    report.metrics["shape_dtype_mismatches"] = len(shape_dtype_bad)
    report.metrics["value_mismatches"] = len(value_bad)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("task", choices=sorted(TESTS))
    parser.add_argument("--target", required=True, choices=sorted(TARGETS), help="surgery-target model")
    parser.add_argument("--out", type=Path, help="participant output (default: out/<task>)")
    parser.add_argument("--reference", type=Path, help="reference output (default: references/<target>/<task>)")
    parser.add_argument("--json", action="store_true", help="print a JSON report instead of text")
    parser.add_argument("--write", type=Path, help="also write the JSON report to this file (e.g. <sandbox>/grade.json)")
    args = parser.parse_args()

    out_path = args.out or HERE / "out" / args.task
    ref_path = args.reference or HERE / "references" / args.target / args.task
    rules = task_rules(args.task, TARGETS[args.target])
    sharded, shard_limit = rules["sharded"], rules["shard_limit"]
    report = Report(task=args.task)
    report.metrics["target"] = args.target

    ref_report = Report(task=args.task)
    ref = load_checkpoint(ref_path, ref_report, require_sharded=sharded, shard_limit=shard_limit)
    if ref is None or not ref_report.passed:
        print(f"reference unusable at {ref_path}: {ref_report.findings}", file=sys.stderr)
        return 2

    out = load_checkpoint(out_path, report, require_sharded=sharded, shard_limit=shard_limit)
    if out is not None:
        compare(rules, out, ref, report)

    payload = {"task": report.task, "passed": report.passed, "findings": report.findings, "metrics": report.metrics}
    if args.write:
        args.write.parent.mkdir(parents=True, exist_ok=True)
        args.write.write_text(json.dumps(payload, indent=2) + "\n")
    if args.json:
        print(json.dumps(payload, indent=2))
    else:
        print(f"{report.task}: {'PASS' if report.passed else 'FAIL'}")
        for line in report.findings:
            print(f"  - {line}")
        for key, value in report.metrics.items():
            print(f"  {key}: {value}")
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
