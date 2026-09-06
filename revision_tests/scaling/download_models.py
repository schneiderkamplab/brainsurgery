#!/usr/bin/env python3
"""Download the scaling matrix at the revisions frozen in cases.yaml."""

from __future__ import annotations

from huggingface_hub import snapshot_download

try:
    from .validate_protocol import REPO, load_cases
except ImportError:
    from validate_protocol import REPO, load_cases


def main() -> int:
    doc = load_cases()
    for model in doc["models"]:
        destination = (REPO / model["input"]).resolve()
        print(f"[{model['id']}] {model['model_id']}@{model['revision']} -> {destination}")
        snapshot_download(
            model["model_id"],
            revision=model["revision"],
            local_dir=destination,
            allow_patterns=["*.json", "*.safetensors"],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
