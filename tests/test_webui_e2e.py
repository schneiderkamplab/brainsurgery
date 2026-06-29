from __future__ import annotations

import base64
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib import request
from urllib.error import HTTPError

import torch
from safetensors.torch import save_file

import brainsurgery  # noqa: F401
from brainsurgery.engine import create_state_dict_provider
from brainsurgery.web.ui.handler import _handler_factory
from brainsurgery.web.ui.session import _SessionState

REPO_ROOT = Path(__file__).resolve().parents[1]


def _post_json(base_url: str, path: str, payload: dict[str, Any]) -> dict[str, Any]:
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(
        url=base_url + path,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=300) as resp:  # noqa: S310
            body = resp.read().decode("utf-8")
    except HTTPError as exc:
        body = exc.read().decode("utf-8")
    decoded = json.loads(body)
    assert isinstance(decoded, dict)
    return decoded


def _get_json(base_url: str, path: str) -> dict[str, Any]:
    req = request.Request(
        url=base_url + path,
        method="GET",
    )
    with request.urlopen(req, timeout=300) as resp:  # noqa: S310
        body = resp.read().decode("utf-8")
    decoded = json.loads(body)
    assert isinstance(decoded, dict)
    return decoded


@contextmanager
def _running_webui(tmp_path: Path) -> Iterator[str]:
    provider = create_state_dict_provider(
        provider="inmemory",
        model_paths={},
        max_io_workers=8,
        arena_root=Path(".brainsurgery"),
        arena_segment_size="1GB",
    )
    session = _SessionState(
        provider=provider,
        lock=threading.Lock(),
        upload_root=tmp_path / "uploads",
    )
    session.upload_root.mkdir(parents=True, exist_ok=True)

    try:
        server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_factory(session))
    except PermissionError as exc:
        provider.close()
        raise RuntimeError(f"local webui bind is not permitted in this environment: {exc}") from exc
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"
    try:
        yield base_url
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        provider.close()


def test_webui_state_exposes_runtime_flags_and_set_updates_them(tmp_path: Path) -> None:
    with _running_webui(tmp_path) as base_url:
        state_before = _get_json(base_url, "/api/state")
        assert state_before.get("ok") is True
        flags_before = state_before.get("runtime_flags")
        assert isinstance(flags_before, dict)
        assert isinstance(flags_before.get("dry_run"), bool)
        assert isinstance(flags_before.get("preview"), bool)
        assert isinstance(flags_before.get("verbose"), bool)

        set_resp = _post_json(
            base_url,
            "/api/_apply_transform",
            {
                "transform": "set",
                "payload": {"dry-run": True, "preview": True, "verbose": True},
            },
        )
        assert set_resp.get("ok") is True, set_resp.get("error")
        flags_after_set = set_resp.get("runtime_flags")
        assert isinstance(flags_after_set, dict)
        assert flags_after_set.get("dry_run") is True
        assert flags_after_set.get("preview") is True
        assert flags_after_set.get("verbose") is True

        state_after = _get_json(base_url, "/api/state")
        flags_after_state = state_after.get("runtime_flags")
        assert isinstance(flags_after_state, dict)
        assert flags_after_state.get("dry_run") is True
        assert flags_after_state.get("preview") is True
        assert flags_after_state.get("verbose") is True


def test_webui_e2e_transforms_and_upload_dump_and_save_flows(tmp_path: Path) -> None:
    tiny_path = tmp_path / "tiny.safetensors"
    save_file({"weight": torch.arange(4, dtype=torch.float32)}, str(tiny_path))
    raw = tiny_path.read_bytes()
    payload_b64 = base64.b64encode(raw).decode("ascii")

    with _running_webui(tmp_path) as base_url:
        transforms_resp = _get_json(base_url, "/api/transforms")
        assert transforms_resp.get("ok") is True
        transforms = transforms_resp.get("transforms")
        assert isinstance(transforms, list) and transforms
        names = {item.get("name") for item in transforms if isinstance(item, dict)}
        assert {"load", "save", "dump", "help", "exit"}.issubset(names)

        upload_resp = _post_json(
            base_url,
            "/api/load",
            {
                "filename": "tiny.safetensors",
                "content_b64": payload_b64,
                "alias": "tiny",
            },
        )
        assert upload_resp.get("ok") is True, upload_resp.get("error")

        state = _get_json(base_url, "/api/state")
        assert state.get("ok") is True
        models = state.get("models")
        assert isinstance(models, list)
        assert any(isinstance(item, dict) and item.get("alias") == "tiny" for item in models)

        dump_ok = _post_json(
            base_url,
            "/api/model_dump",
            {
                "alias": "tiny",
                "format": "compact",
                "verbosity": "shape",
                "filter": ".*",
            },
        )
        assert dump_ok.get("ok") is True, dump_ok.get("error")
        assert isinstance(dump_ok.get("dump"), str)
        assert dump_ok.get("matched_count") == 1
        assert dump_ok.get("total_count") == 1

        dump_bad = _post_json(
            base_url,
            "/api/model_dump",
            {
                "alias": "tiny",
                "format": "compact",
                "verbosity": "shape",
                "filter": "[",
            },
        )
        assert dump_bad.get("ok") is False
        assert isinstance(dump_bad.get("error"), str)

        save_download = _post_json(
            base_url,
            "/api/save_download",
            {
                "payload": {
                    "path": "tiny_export",
                    "alias": "tiny",
                    "format": "safetensors",
                }
            },
        )
        assert save_download.get("ok") is True, save_download.get("error")
        assert isinstance(save_download.get("download_b64"), str) and save_download.get(
            "download_b64"
        )
        assert str(save_download.get("download_filename", "")).endswith(".safetensors")

        server_out = tmp_path / "server_saved.safetensors"
        save_server = _post_json(
            base_url,
            "/api/_apply_transform",
            {
                "transform": "save",
                "payload": {
                    "path": str(server_out),
                    "alias": "tiny",
                    "format": "safetensors",
                },
            },
        )
        assert save_server.get("ok") is True, save_server.get("error")
        assert server_out.exists()


def test_webui_preview_requires_confirmation_and_honors_go_no_go(tmp_path: Path) -> None:
    tiny_path = tmp_path / "tiny_preview.safetensors"
    save_file({"weight": torch.arange(4, dtype=torch.float32)}, str(tiny_path))

    with _running_webui(tmp_path) as base_url:
        load_resp = _post_json(
            base_url,
            "/api/load",
            {
                "server_path": str(tiny_path),
                "alias": "tiny",
            },
        )
        assert load_resp.get("ok") is True

        set_resp = _post_json(
            base_url,
            "/api/_apply_transform",
            {
                "transform": "set",
                "payload": {"preview": True, "dry-run": False},
            },
        )
        assert set_resp.get("ok") is True

        preview_resp = _post_json(
            base_url,
            "/api/_apply_transform",
            {
                "transform": "delete",
                "payload": {"target": "tiny::weight"},
            },
        )
        assert preview_resp.get("ok") is True
        assert preview_resp.get("preview_confirmation_required") is True
        assert "preview 1/1 delete:" in str(preview_resp.get("output"))
        models_after_preview = preview_resp.get("models")
        assert isinstance(models_after_preview, list)
        tiny_after_preview = next(
            item for item in models_after_preview if item.get("alias") == "tiny"
        )
        assert tiny_after_preview.get("tensor_count") == 1

        no_go_resp = _post_json(
            base_url,
            "/api/_apply_transform",
            {
                "transform": "delete",
                "payload": {"target": "tiny::weight"},
                "preview_decision": "no-go",
            },
        )
        assert no_go_resp.get("ok") is True
        assert no_go_resp.get("preview_confirmation_required") is False
        assert "no-go, apply skipped" in str(no_go_resp.get("output"))
        models_after_no_go = no_go_resp.get("models")
        assert isinstance(models_after_no_go, list)
        tiny_after_no_go = next(item for item in models_after_no_go if item.get("alias") == "tiny")
        assert tiny_after_no_go.get("tensor_count") == 1

        go_resp = _post_json(
            base_url,
            "/api/_apply_transform",
            {
                "transform": "delete",
                "payload": {"target": "tiny::weight"},
                "preview_decision": "go",
            },
        )
        assert go_resp.get("ok") is True
        assert go_resp.get("preview_confirmation_required") is False
        models_after_go = go_resp.get("models")
        assert isinstance(models_after_go, list)
        tiny_after_go = next(item for item in models_after_go if item.get("alias") == "tiny")
        assert tiny_after_go.get("tensor_count") == 0
