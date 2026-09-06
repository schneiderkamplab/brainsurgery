#!/usr/bin/env python3
"""Run one model role over the frozen behavioral regression manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import transformers
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
DEFAULT_MANIFEST = HERE / "prompt_manifest.jsonl"
PROTOCOL_ID = "eacl2027_behavioral_v1"
DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
CHOICE_CANDIDATES = [" A", " B", " C", " D"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=("reference", "transformed"), required=True)
    parser.add_argument("--model", required=True, help="Local checkpoint path or Hugging Face ID")
    parser.add_argument(
        "--tokenizer", required=True, help="Shared tokenizer path or Hugging Face ID"
    )
    parser.add_argument(
        "--config", help="Optional reference config path/ID for transformed tensors"
    )
    parser.add_argument(
        "--revision", required=True, help="Pinned model revision or transformation ID"
    )
    parser.add_argument(
        "--tokenizer-revision", help="Pinned tokenizer revision; defaults to --revision"
    )
    parser.add_argument(
        "--config-revision", help="Pinned config revision; defaults to tokenizer revision"
    )
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--device", required=True, help="For example cuda, cuda:0, cpu, or mps")
    parser.add_argument("--dtype", choices=tuple(DTYPES), default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument(
        "--smoke-limit",
        type=int,
        help="Run only the leading N prompts and mark the output non-reportable",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def prompt_ids_sha256(prompt_ids: list[str]) -> str:
    return hashlib.sha256("\n".join(prompt_ids).encode("utf-8")).hexdigest()


def read_manifest(path: Path) -> list[dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if len(rows) != 70 or any(row.get("protocol_id") != PROTOCOL_ID for row in rows):
        raise ValueError("manifest is not the frozen 70-prompt behavioral protocol")
    return rows


def git_value(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args], cwd=REPO_ROOT, check=True, capture_output=True, text=True
        )
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"
    return result.stdout.strip()


def source_kwargs(identifier: str, revision: str, args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "local_files_only": args.local_files_only,
        "trust_remote_code": args.trust_remote_code,
    }
    if not Path(identifier).expanduser().exists():
        kwargs["revision"] = revision
    return kwargs


def tokenizer_fingerprint(tokenizer: Any) -> str:
    backend = None
    if getattr(tokenizer, "backend_tokenizer", None) is not None:
        backend = tokenizer.backend_tokenizer.to_str()
    data = {
        "class": tokenizer.__class__.__name__,
        "backend": backend,
        "vocab": tokenizer.get_vocab(),
        "added_vocab": tokenizer.get_added_vocab(),
        "special_tokens_map": tokenizer.special_tokens_map,
        "padding_side": tokenizer.padding_side,
        "truncation_side": tokenizer.truncation_side,
        "model_max_length": tokenizer.model_max_length,
    }
    return sha256_json(data)


def config_fingerprint(config: Any) -> str:
    data = config.to_dict()
    for key in ("_name_or_path", "name_or_path", "transformers_version"):
        data.pop(key, None)
    return sha256_json(data)


def device_fingerprint(device: torch.device) -> dict[str, Any]:
    result: dict[str, Any] = {"type": device.type, "index": device.index}
    if device.type == "cuda":
        index = device.index if device.index is not None else torch.cuda.current_device()
        properties = torch.cuda.get_device_properties(index)
        result.update(
            {
                "name": properties.name,
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory": properties.total_memory,
                "cuda_runtime": torch.version.cuda,
            }
        )
    elif device.type == "mps":
        result["available"] = torch.backends.mps.is_available()
    else:
        result["processor"] = platform.processor()
    return result


def context_limit(config: Any, tokenizer: Any) -> int | None:
    values = []
    for key in ("max_position_embeddings", "n_positions", "max_sequence_length"):
        value = getattr(config, key, None)
        if isinstance(value, int) and value > 0:
            values.append(value)
    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10**9:
        values.append(tokenizer_limit)
    return min(values) if values else None


def score_choices(
    model: Any, tokenizer: Any, prompt_ids: torch.Tensor, device: torch.device
) -> tuple[str, dict[str, float]]:
    scores: dict[str, float] = {}
    for label, candidate in zip(("A", "B", "C", "D"), CHOICE_CANDIDATES, strict=True):
        candidate_ids = tokenizer(candidate, add_special_tokens=False, return_tensors="pt")[
            "input_ids"
        ].to(device)
        if candidate_ids.shape[1] == 0:
            raise ValueError(f"tokenizer produced no tokens for choice {candidate!r}")
        joint = torch.cat((prompt_ids, candidate_ids), dim=1)
        output = model(input_ids=joint, use_cache=False)
        log_probs = torch.log_softmax(output.logits[0].float(), dim=-1)
        start = prompt_ids.shape[1] - 1
        selected = [
            log_probs[start + offset, token_id]
            for offset, token_id in enumerate(candidate_ids[0].tolist())
        ]
        scores[label] = float(torch.stack(selected).mean().cpu().item())
    prediction = max(scores, key=lambda label: (scores[label], -ord(label)))
    return prediction, scores


def main() -> int:
    args = parse_args()
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing output: {args.output}")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be positive")
    if args.smoke_limit is not None and not 1 <= args.smoke_limit <= 70:
        raise SystemExit("--smoke-limit must be between 1 and 70")

    rows = read_manifest(args.manifest)
    if args.smoke_limit is not None:
        rows = rows[: args.smoke_limit]
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is unavailable")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise SystemExit("MPS was requested but is unavailable")

    random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    torch.use_deterministic_algorithms(True)

    tokenizer_revision = args.tokenizer_revision or args.revision
    config_revision = args.config_revision or tokenizer_revision
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, **source_kwargs(args.tokenizer, tokenizer_revision, args)
    )
    config_source = args.config or args.model
    config = AutoConfig.from_pretrained(
        config_source, **source_kwargs(config_source, config_revision, args)
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        config=config,
        dtype=DTYPES[args.dtype],
        **source_kwargs(args.model, args.revision, args),
    )
    model.to(device)
    model.eval()
    actual_dtype = str(next(model.parameters()).dtype).removeprefix("torch.")
    if actual_dtype != args.dtype:
        raise RuntimeError(f"requested dtype {args.dtype}, loaded {actual_dtype}")

    maximum_context = context_limit(config, tokenizer)
    prediction_rows: list[dict[str, Any]] = []
    logits_tensors: dict[str, torch.Tensor] = {}
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id

    with torch.inference_mode():
        for row in rows:
            encoded = tokenizer(row["prompt"], return_tensors="pt", truncation=False)
            input_ids = encoded["input_ids"].to(device)
            attention_mask = encoded.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)
            input_length = input_ids.shape[1]
            if maximum_context is not None and input_length + args.max_new_tokens > maximum_context:
                raise ValueError(
                    f"{row['prompt_id']}: {input_length} input + {args.max_new_tokens} generated "
                    f"tokens exceeds context {maximum_context}; truncation is forbidden"
                )
            output = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            final_logits = output.logits[0, -1].float().cpu().contiguous()
            logits_key = f"p{row['ordinal']:04d}"
            logits_tensors[logits_key] = final_logits
            generation = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=pad_token_id,
            )
            generated_ids = generation[0, input_length:].tolist()
            expected = row["expected"]
            if expected["kind"] == "multiple_choice":
                mcq_prediction, choice_scores = score_choices(model, tokenizer, input_ids, device)
                correct_label = expected["correct_label"]
            else:
                mcq_prediction, choice_scores, correct_label = None, None, None
            prediction_rows.append(
                {
                    "prompt_id": row["prompt_id"],
                    "ordinal": row["ordinal"],
                    "input_token_count": input_length,
                    "generated_token_ids": generated_ids,
                    "generated_text": tokenizer.decode(generated_ids, skip_special_tokens=False),
                    "next_token_top1_id": int(final_logits.argmax().item()),
                    "mcq_predicted_label": mcq_prediction,
                    "mcq_choice_mean_logprobs": choice_scores,
                    "correct_label": correct_label,
                    "logits_key": logits_key,
                }
            )

    prompt_ids = [row["prompt_id"] for row in rows]
    git_commit = git_value("rev-parse", "HEAD")
    git_status = git_value("status", "--short")
    reported_eligible = (
        args.smoke_limit is None
        and len(rows) == 70
        and device.type == "cuda"
        and git_commit != "unavailable"
        and git_status == ""
    )
    metadata = {
        "protocol_id": PROTOCOL_ID,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "role": args.role,
        "reported_eligible": reported_eligible,
        "smoke_limit": args.smoke_limit,
        "manifest": str(args.manifest.resolve()),
        "manifest_sha256": sha256_file(args.manifest),
        "selected_prompt_ids_sha256": prompt_ids_sha256(prompt_ids),
        "prompt_count": len(rows),
        "model": args.model,
        "model_revision": args.revision,
        "tokenizer": args.tokenizer,
        "tokenizer_revision": tokenizer_revision,
        "config": config_source,
        "config_revision": config_revision,
        "tokenizer_fingerprint": tokenizer_fingerprint(tokenizer),
        "model_architecture_fingerprint": config_fingerprint(config),
        "device_fingerprint": device_fingerprint(device),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "dtype": actual_dtype,
        "max_new_tokens": args.max_new_tokens,
        "do_sample": False,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "seed": 0,
        "tokenizer_call": {
            "add_special_tokens": "tokenizer_default",
            "truncation": False,
            "batch_size": 1,
            "padding_side": tokenizer.padding_side,
            "special_tokens_map": tokenizer.special_tokens_map,
        },
        "choice_candidates": CHOICE_CANDIDATES,
        "git_commit": git_commit,
        "git_status_porcelain": git_status,
        "command": [sys.executable, *sys.argv],
        "environment": {
            "hostname": platform.node(),
            "machine": platform.machine(),
            "executable": sys.executable,
            "pid": os.getpid(),
        },
    }
    args.output.mkdir(parents=True)
    (args.output / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output / "predictions.jsonl").write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in prediction_rows
        ),
        encoding="utf-8",
    )
    save_file(logits_tensors, args.output / "last_token_logits.safetensors")
    print(
        f"wrote {len(rows)} {args.role} predictions to {args.output}; "
        f"reported_eligible={str(metadata['reported_eligible']).lower()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
