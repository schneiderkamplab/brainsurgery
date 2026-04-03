from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer


def load_tokenizer(
    tokenizer_source: str,
    *,
    fallback_repo_id: str | None = None,
    trust_remote_code: bool = False,
) -> Any:
    candidate = Path(tokenizer_source).expanduser()
    if candidate.exists():
        source = str(candidate.resolve())
        try:
            return AutoTokenizer.from_pretrained(
                source, local_files_only=True, trust_remote_code=trust_remote_code
            )
        except Exception:
            return AutoTokenizer.from_pretrained(
                source,
                local_files_only=True,
                use_fast=False,
                trust_remote_code=trust_remote_code,
            )

    try:
        return AutoTokenizer.from_pretrained(
            tokenizer_source, local_files_only=False, trust_remote_code=trust_remote_code
        )
    except Exception:
        if fallback_repo_id and fallback_repo_id != tokenizer_source:
            try:
                return AutoTokenizer.from_pretrained(
                    fallback_repo_id,
                    local_files_only=False,
                    trust_remote_code=trust_remote_code,
                )
            except Exception:
                pass
        return AutoTokenizer.from_pretrained(
            tokenizer_source,
            local_files_only=False,
            use_fast=False,
            trust_remote_code=trust_remote_code,
        )


def looks_like_tokenizer_dir(path: Path) -> bool:
    if (
        (path / "tokenizer.json").exists()
        or (path / "tokenizer.model").exists()
        or ((path / "vocab.json").exists() and (path / "merges.txt").exists())
    ):
        return True
    # Some custom-code tokenizers (for example Phi-3-small) rely on
    # tokenization_*.py plus a companion tokenizer config / .tiktoken asset.
    return (path / "tokenizer_config.json").exists() and (
        any(path.glob("tokenization*.py")) or any(path.glob("*.tiktoken"))
    )


def candidate_tokenizer_dirs(model_dir: Path) -> list[Path]:
    candidates: list[Path] = []
    candidates.append(model_dir)
    candidates.append(model_dir.with_name(f"{model_dir.name}.old"))
    parts = model_dir.name.split("_")
    for cut in range(len(parts) - 1, 1, -1):
        candidates.append(model_dir.with_name("_".join(parts[:cut])))

    out: list[Path] = []
    seen: set[Path] = set()
    for path in candidates:
        resolved = path.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        out.append(resolved)
    return out


def spec_padding_side(spec: dict[str, Any]) -> str | None:
    model = spec.get("model", {})
    if not isinstance(model, dict):
        return None
    meta = model.get("meta", {})
    if not isinstance(meta, dict):
        return None
    value = meta.get("padding_side")
    if value is None:
        return None
    normalized = str(value).strip().lower()
    if normalized not in {"left", "right"}:
        raise ValueError(f"Invalid model.meta.padding_side={value!r}; expected 'left' or 'right'.")
    return normalized


def preferred_padding_side(spec: dict[str, Any]) -> str:
    explicit = spec_padding_side(spec)
    if explicit is not None:
        return explicit
    return "left"


def tokenize_prompts(
    *,
    prompts: list[str],
    tokenizer_source: str,
    device: torch.device,
    lowered_spec: dict[str, Any] | None = None,
    tokenizer_fallback: str | None = None,
    trust_remote_code: bool = False,
) -> tuple[Any, torch.Tensor, torch.Tensor | None]:
    tokenizer_obj = load_tokenizer(
        tokenizer_source,
        fallback_repo_id=tokenizer_fallback,
        trust_remote_code=trust_remote_code,
    )
    if len(prompts) > 1:
        tokenizer_obj.padding_side = (
            preferred_padding_side(lowered_spec) if lowered_spec is not None else "left"
        )
        if tokenizer_obj.pad_token_id is None:
            if tokenizer_obj.eos_token_id is None:
                raise ValueError(
                    "Tokenizer has no pad token and no eos token; cannot batch prompts"
                )
            tokenizer_obj.pad_token = tokenizer_obj.eos_token
    inputs = tokenizer_obj(prompts, return_tensors="pt", padding=(len(prompts) > 1)).to(device)
    input_ids = inputs["input_ids"]
    attention_mask = inputs.get("attention_mask")
    return tokenizer_obj, input_ids, attention_mask


__all__ = [
    "candidate_tokenizer_dirs",
    "load_tokenizer",
    "looks_like_tokenizer_dir",
    "preferred_padding_side",
    "spec_padding_side",
    "tokenize_prompts",
]
