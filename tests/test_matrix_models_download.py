from __future__ import annotations

import subprocess
import sys
import types

from brainsurgery.synapse import matrix_models


def test_pytorch_weight_filename_rejects_tokenizer_binary() -> None:
    assert matrix_models._is_pytorch_weight_filename("pytorch_model.bin")
    assert matrix_models._is_pytorch_weight_filename("pytorch_model-00001-of-00002.bin")
    assert matrix_models._is_pytorch_weight_filename("model.bin")
    assert not matrix_models._is_pytorch_weight_filename("bpe_encoder.bin")


def test_vocab_txt_is_a_complete_tokenizer_artifact(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "vocab.txt").write_text("[PAD]\n[UNK]\n", encoding="utf-8")

    assert matrix_models._is_complete_model_dir(tmp_path, require_tokenizer=True)


def test_named_tokenizer_is_a_complete_tokenizer_artifact(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "model.tokenizer").write_bytes(b"tokenizer")

    assert matrix_models._is_complete_model_dir(tmp_path, require_tokenizer=True)


def test_sentencepiece_filename_variants_are_tokenizer_artifacts(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "sentencepiece.bpe.model").write_bytes(b"tokenizer")

    assert matrix_models._is_complete_model_dir(tmp_path, require_tokenizer=True)
    assert matrix_models._is_tokenizer_asset_name("sentencepiece.bpe.model")
    assert matrix_models._is_tokenizer_asset_name("source.spm")


def test_tokenizer_config_alone_is_not_a_complete_tokenizer_artifact(tmp_path) -> None:  # type: ignore[no-untyped-def]
    (tmp_path / "config.json").write_text("{}\n", encoding="utf-8")
    (tmp_path / "model.safetensors").touch()
    (tmp_path / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")

    assert not matrix_models._is_complete_model_dir(tmp_path, require_tokenizer=True)


def test_head_content_length_prefers_resolved_payload_size(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = """\
HTTP/2 302
content-length: 1099
x-linked-size: 509910379
HTTP/2 200
content-length: 509910379
"""
    monkeypatch.setattr(
        matrix_models.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""),
    )

    assert (
        matrix_models._head_content_length(
            url="https://example.invalid/model.bin",
            cwd=tmp_path,
        )
        == 509910379
    )


def test_head_content_length_uses_final_non_lfs_response(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    output = """\
HTTP/2 307
content-length: 294
HTTP/2 200
content-length: 293
"""
    monkeypatch.setattr(
        matrix_models.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(args[0], 0, output, ""),
    )

    assert (
        matrix_models._head_content_length(
            url="https://example.invalid/generation_config.json",
            cwd=tmp_path,
        )
        == 293
    )


def test_hf_hub_download_places_resolved_file(monkeypatch, tmp_path) -> None:  # type: ignore[no-untyped-def]
    module = types.ModuleType("huggingface_hub")

    def fake_download(*, repo_id, filename, revision, local_dir):  # type: ignore[no-untyped-def]
        assert repo_id == "example/model"
        assert filename == "pytorch_model.bin"
        assert revision == "main"
        target = local_dir / filename
        target.write_bytes(b"weights")
        return str(target)

    module.hf_hub_download = fake_download  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "huggingface_hub", module)
    target = tmp_path / "pytorch_model.bin"

    assert matrix_models._run_hf_hub_download(
        url="https://huggingface.co/example/model/resolve/main/pytorch_model.bin",
        out_path=target,
    )
    assert target.read_bytes() == b"weights"
