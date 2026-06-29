from __future__ import annotations

import io
import tempfile
import threading
from dataclasses import dataclass
from pathlib import Path

import pytest
import typer

import brainsurgery.web.ui.backend as webui_backend
import brainsurgery.web.ui.cli as webui_cli
import brainsurgery.web.ui.handler as webui_handler
import brainsurgery.web.ui.page as webui_page
import brainsurgery.web.ui.server as webui_server
from brainsurgery.engine import SurgeryPlan
from brainsurgery.web.ui.session import _SessionState


class _Provider:
    def __init__(self) -> None:
        self.closed = False
        self.models: dict[str, dict[str, object]] = {}

    def list_model_aliases(self) -> list[str]:
        return sorted(self.models)

    def get_state_dict(self, alias: str):
        return self.models[alias]

    def close(self) -> None:
        self.closed = True


def test_webui_configure_logging_and_bad_level() -> None:
    webui_cli.configure_logging("warning")
    with pytest.raises(typer.BadParameter):
        webui_cli.configure_logging("bogus")


def test_webui_callback_handles_browser_error_and_wildcard_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    served: list[tuple[str, int]] = []
    monkeypatch.setattr(webui_cli, "configure_logging", lambda _lvl: None)

    def _boom(_url: str) -> None:
        raise RuntimeError("browser failed")

    monkeypatch.setattr(webui_cli.webbrowser, "open", _boom)
    monkeypatch.setattr(
        webui_cli, "_serve_webui", lambda *, host, port: served.append((host, port))
    )
    webui_cli.webui(host="::", port=8124, log_level="info", open_browser=True)
    assert served == [("::", 8124)]


def test_webui_backend_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    original_disabled = set(webui_backend.DISABLED_TRANSFORMS)
    try:
        webui_backend.DISABLED_TRANSFORMS.add("help")
        names = {item["name"] for item in webui_backend._transform_items()}
        assert "help" not in names
    finally:
        webui_backend.DISABLED_TRANSFORMS.clear()
        webui_backend.DISABLED_TRANSFORMS.update(original_disabled)

    with pytest.raises(ValueError):
        webui_backend._require_string("", "x")

    p = _Provider()
    assert webui_backend._default_alias(p) == "model"
    p.models = {"model": {}, "model_2": {}}
    assert webui_backend._default_alias(p) == "model_3"

    assert webui_backend._parse_filter_expr(None, alias="m") == ".*"
    assert webui_backend._parse_filter_expr(" ", alias="m") == ".*"
    assert isinstance(webui_backend._parse_filter_expr('["a","b"]', alias="m"), list)
    assert isinstance(webui_backend._parse_filter_expr(["a"], alias="m"), list)
    with pytest.raises(ValueError):
        webui_backend._parse_filter_expr(1, alias="m")


def test_webui_backend_boolean_keys_and_apply_transform_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    @dataclass
    class _Spec:
        dry_run: bool | None

    class _DummyTransform:
        spec_type = _Spec
        help_text = "- dry-run (boolean)"

    out = webui_backend._boolean_keys_for_transform(_DummyTransform(), ["dry-run"])
    assert out == ["dry-run"]

    monkeypatch.setattr(
        webui_backend, "get_type_hints", lambda _s: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    assert webui_backend._boolean_keys_for_transform(_DummyTransform(), ["dry-run"]) == ["dry-run"]

    p = _Provider()
    p.models = {"only": {"w": object()}}
    plan = SurgeryPlan(inputs={}, output=None)
    original_disabled = set(webui_backend.DISABLED_TRANSFORMS)
    try:
        webui_backend.DISABLED_TRANSFORMS.add("copy")
        with pytest.raises(ValueError, match="disabled"):
            webui_backend._apply_transform(
                provider=p,
                plan=plan,
                transform_name="copy",
                payload={},
                default_model="only",
            )
    finally:
        webui_backend.DISABLED_TRANSFORMS.clear()
        webui_backend.DISABLED_TRANSFORMS.update(original_disabled)

    with pytest.raises(ValueError, match="cannot be empty"):
        webui_backend._apply_transform(
            provider=p,
            plan=plan,
            transform_name="assert",
            payload="  ",
            default_model="only",
        )

    with pytest.raises(ValueError, match="invalid assert YAML payload"):
        webui_backend._apply_transform(
            provider=p,
            plan=plan,
            transform_name="assert",
            payload=":\n",
            default_model="only",
        )

    with pytest.raises(Exception, match="count payload must be a mapping"):
        webui_backend._apply_transform(
            provider=p,
            plan=plan,
            transform_name="assert",
            payload={"count": "of: only::w\nis: 1"},
            default_model="only",
        )

    plan = SurgeryPlan(inputs={}, output=None)
    _output, default_model = webui_backend._apply_transform(
        provider=p,
        plan=plan,
        transform_name="set",
        payload={"dry-run": True},
        default_model=None,
    )
    assert default_model == "only"

    webui_backend._apply_transform(
        provider=p,
        plan=plan,
        transform_name="help",
        payload={},
        default_model="only",
    )


def test_webui_backend_render_dump_for_alias_no_matches() -> None:
    p = _Provider()
    p.models = {"m": {"w": object()}}
    dumped, matched, total = webui_backend._render_dump_for_alias(
        provider=p,
        alias="m",
        format_name="compact",
        verbosity="shape",
        target="^nope$",
    )
    assert dumped == "(no tensors matched filter)"
    assert matched == 0
    assert total == 1


def _make_session(tmp_path: Path) -> _SessionState:
    provider = _Provider()
    upload_root = tmp_path / "uploads"
    upload_root.mkdir(parents=True, exist_ok=True)
    return _SessionState(provider=provider, lock=threading.Lock(), upload_root=upload_root)


def test_webui_handler_factory_resets_runtime_flags_for_webui_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        webui_handler,
        "reset_runtime_flags_for_scope",
        lambda scope: calls.append(scope),
    )
    webui_handler._handler_factory(_make_session(tmp_path))
    assert calls == [webui_handler.RuntimeFlagLifecycleScope.WEBUI_SESSION]


def test_webui_handler_routes_and_errors(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    handler_cls = webui_handler._handler_factory(session)

    h = object.__new__(handler_cls)
    h._json = []
    h._html = []
    h._errors = []
    h._send_json = lambda payload, status=200: h._json.append((status, payload))
    h._send_html = lambda body: h._html.append(body)
    h.send_error = lambda code, message: h._errors.append((code, message))
    h._read_json_body = lambda: {}

    h.path = "/"
    handler_cls.do_GET(h)
    assert h._html

    h.path = "/api/transforms"
    handler_cls.do_GET(h)
    assert h._json[-1][1]["ok"] is True

    h.path = "/api/state"
    handler_cls.do_GET(h)
    assert h._json[-1][1]["ok"] is True

    h.path = "/api/progress"
    handler_cls.do_GET(h)
    assert h._json[-1][1]["ok"] is True
    assert "progress" in h._json[-1][1]

    h.path = "/missing"
    handler_cls.do_GET(h)
    assert h._errors[-1][0] == 404

    h.path = "/static/../../etc/passwd"
    handler_cls.do_GET(h)
    assert h._errors[-1][0] == 404

    h.path = "/api/load"
    h._read_json_body = lambda: {"alias": ""}
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is False

    h.path = "/api/load"
    h._read_json_body = lambda: {"server_path": "/x", "alias": "a"}
    monkeypatch.setattr(
        webui_handler,
        "_apply_load_transform",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("x")),
    )
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is False

    h.path = "/api/_apply_transform"
    h._read_json_body = lambda: {"transform": "copy", "payload": 1}
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is False

    h.path = "/api/_apply_transform"
    h._read_json_body = lambda: {"transform": "copy", "payload": {}, "preview_decision": 1}
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is False

    h.path = "/api/_apply_transform"
    h._read_json_body = lambda: {"transform": "copy", "payload": {}, "preview_decision": "maybe"}
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is False

    h.path = "/api/_apply_transform"
    h._read_json_body = lambda: {"transform": "copy", "payload": {}}
    original_get_transform = webui_handler.get_transform
    monkeypatch.setattr(
        webui_handler,
        "get_transform",
        lambda _name: (_ for _ in ()).throw(RuntimeError("missing transform")),
    )
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is False
    monkeypatch.setattr(webui_handler, "get_transform", original_get_transform)

    h.path = "/api/_apply_transform"
    h._read_json_body = lambda: {"transform": "assert", "payload": " "}
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is False
    assert h._json[-1][1]["error_info"]["code"] == "assert_error"
    assert h._json[-1][1]["error_info"]["location"] == {"transform": "assert", "field": "payload"}
    assert h._json[-1][1]["error_info"]["endpoint"] == "/api/_apply_transform"
    assert h._json[-1][1]["error_info"]["context"] == {"raw_payload": ""}

    h.path = "/api/_apply_transform"
    h._read_json_body = lambda: {"transform": "assert", "payload": ":"}
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is False
    assert h._json[-1][1]["error_info"]["code"] == "assert_error"
    assert h._json[-1][1]["error_info"]["location"] == {"transform": "assert", "field": "payload"}
    assert h._json[-1][1]["error_info"]["context"] == {"raw_payload": ":"}

    h.path = "/api/_apply_transform"
    h._read_json_body = lambda: {"transform": "exit", "payload": {}}
    monkeypatch.setattr(webui_handler, "_apply_transform", lambda **_k: ("out", "m"))
    monkeypatch.setattr(webui_handler, "_serialize_models", lambda _p: [])
    seen_modes: list[str] = []
    monkeypatch.setattr(
        webui_handler,
        "_render_execution_summary",
        lambda **kwargs: seen_modes.append(kwargs["mode"]) or "transforms: []\n",
    )
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is True
    assert "transforms:" in h._json[-1][1]["output"]
    assert seen_modes[-1] == "raw"

    h.path = "/api/_apply_transform"
    h._read_json_body = lambda: {"transform": "exit", "payload": {}, "summary_mode": "resolve"}
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is True
    assert seen_modes[-1] == "resolve"

    h.path = "/api/_apply_transform"
    h._read_json_body = lambda: {"transform": "exit", "payload": {}, "summary_mode": 1}
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is False

    h.path = "/api/_apply_transform"
    h._read_json_body = lambda: {"transform": "help"}
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is True

    h.path = "/api/_apply_transform"
    h._read_json_body = lambda: {
        "transform": "assert",
        "payload": "equal:\n  left: a\n  right: a\n",
    }
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is True

    h.path = "/api/save_download"
    h._read_json_body = lambda: {"payload": 1}
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is False

    h.path = "/api/save_download"
    h._read_json_body = lambda: {"payload": {}}
    h._run_save_download = lambda _payload: (_ for _ in ()).throw(RuntimeError("save boom"))
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is False

    h.path = "/api/save_download"
    h._read_json_body = lambda: {}
    h._run_save_download = lambda _payload: ("saved", "m.bin", "application/octet-stream", "YQ==")
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is True

    h.path = "/api/model_dump"
    h._read_json_body = lambda: {"alias": "", "format": "compact", "verbosity": "shape"}
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is False

    h.path = "/api/model_dump"
    h._read_json_body = lambda: {"alias": "x", "format": "nope", "verbosity": "shape"}
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is False

    h.path = "/api/model_dump"
    h._read_json_body = lambda: {"alias": "x", "format": "compact", "verbosity": "bad"}
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is False

    h.path = "/api/model_dump"
    h._read_json_body = lambda: {
        "alias": "x",
        "format": "compact",
        "verbosity": "shape",
        "filter": ".*",
    }
    monkeypatch.setattr(webui_handler, "_parse_filter_expr", lambda _v, alias: ".*")
    monkeypatch.setattr(webui_handler, "list_model_aliases", lambda _p: [])
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is False

    h.path = "/api/model_dump"
    monkeypatch.setattr(webui_handler, "list_model_aliases", lambda _p: ["x"])
    monkeypatch.setattr(
        webui_handler,
        "_render_dump_for_alias",
        lambda **_k: (_ for _ in ()).throw(RuntimeError("dump boom")),
    )
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is False

    h.path = "/api/model_dump"
    monkeypatch.setattr(webui_handler, "_render_dump_for_alias", lambda **_k: ("d", 1, 2))
    handler_cls.do_POST(h)
    assert h._json[-1][1]["ok"] is True

    h.path = "/api/none"
    handler_cls.do_POST(h)
    assert h._errors[-1][0] == 404


def test_webui_page_contains_exit_summary_mode_selector() -> None:
    transforms_ui_js = (webui_page._STATIC_DIR / "js" / "transforms_ui.js").read_text(
        encoding="utf-8"
    )
    assert '<script type="module" src="/static/js/main.js"></script>' in webui_page._HTML_PAGE
    assert "summary mode: raw" in transforms_ui_js
    assert "summaryModeSelect" in transforms_ui_js
    assert 'modeKey + ": " + String(name).toLowerCase()' in transforms_ui_js


def test_webui_handler_read_send_helpers_and_filename_suggestion(tmp_path: Path) -> None:
    session = _make_session(tmp_path)
    handler_cls = webui_handler._handler_factory(session)
    h = object.__new__(handler_cls)
    h.wfile = io.BytesIO()
    h._responses = []
    h._headers = []
    h.end_headers = lambda: None
    h.send_response = lambda code: h._responses.append(code)
    h.send_header = lambda key, val: h._headers.append((key, val))

    handler_cls._send_html(h, "x")
    handler_cls._send_json(h, {"ok": True})
    assert h._responses == [200, 200]
    assert any(k == "Cache-Control" for k, _v in h._headers)

    static = object.__new__(handler_cls)
    static.path = "/static/js/main.js"
    static.wfile = io.BytesIO()
    static._responses = []
    static._headers = []
    static._errors = []
    static.end_headers = lambda: None
    static.send_response = lambda code: static._responses.append(code)
    static.send_header = lambda key, val: static._headers.append((key, val))
    static.send_error = lambda code, message: static._errors.append((code, message))
    handler_cls.do_GET(static)
    assert static._responses == [200]
    assert any(k == "Content-Type" and "application/javascript" in v for k, v in static._headers)
    assert static.wfile.getvalue()

    static_missing = object.__new__(handler_cls)
    static_missing.path = "/static/missing.js"
    static_missing._errors = []
    static_missing.send_error = lambda code, message: static_missing._errors.append((code, message))
    handler_cls.do_GET(static_missing)
    assert static_missing._errors[-1][0] == 404

    missing_len = object.__new__(handler_cls)
    missing_len.headers = {}
    missing_len.rfile = io.BytesIO(b"{}")
    with pytest.raises(ValueError, match="Missing Content-Length"):
        handler_cls._read_json_body(missing_len)

    non_obj = object.__new__(handler_cls)
    non_obj.headers = {"Content-Length": "2"}
    non_obj.rfile = io.BytesIO(b"[]")
    with pytest.raises(ValueError, match="JSON object"):
        handler_cls._read_json_body(non_obj)

    assert (
        webui_handler._suggest_download_filename(
            requested_name="x",
            out_path=Path("x.safetensors"),
            payload={},
        )
        == "x.safetensors"
    )
    assert (
        webui_handler._suggest_download_filename(
            requested_name="x.txt",
            out_path=Path("x"),
            payload={},
        )
        == "x.txt"
    )
    assert (
        webui_handler._suggest_download_filename(
            requested_name="x",
            out_path=Path("x"),
            payload={"format": "numpy"},
        )
        == "x.npy"
    )
    assert (
        webui_handler._suggest_download_filename(
            requested_name="x",
            out_path=Path("x"),
            payload={"format": "torch"},
        )
        == "x.pt"
    )
    assert (
        webui_handler._suggest_download_filename(
            requested_name="x",
            out_path=Path("x"),
            payload={"format": "unknown"},
        )
        == "x.bin"
    )


def test_webui_handler_run_save_download_branches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = _make_session(tmp_path)
    handler_cls = webui_handler._handler_factory(session)
    h = object.__new__(handler_cls)

    with pytest.raises(ValueError, match="payload.path"):
        handler_cls._run_save_download(h, {})

    class _TempDir:
        def __init__(self, path: Path) -> None:
            self.path = path

        def __enter__(self) -> str:
            self.path.mkdir(parents=True, exist_ok=True)
            return str(self.path)

        def __exit__(self, exc_type, exc, tb) -> bool:
            return False

    dir_path = tmp_path / "tmp_dir_case"
    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda prefix=None: _TempDir(dir_path))
    monkeypatch.setattr(webui_handler, "_apply_transform", lambda **_k: ("ok", None))
    (dir_path / "dir_only").mkdir(parents=True, exist_ok=True)
    with pytest.raises(ValueError, match="supports files only"):
        handler_cls._run_save_download(h, {"path": "dir_only"})

    no_file_path = tmp_path / "tmp_missing_case"
    monkeypatch.setattr(tempfile, "TemporaryDirectory", lambda prefix=None: _TempDir(no_file_path))
    with pytest.raises(ValueError, match="did not produce a file or directory"):
        handler_cls._run_save_download(h, {"path": "missing"})


def test_webui_server_loop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    events: list[str] = []
    provider = _Provider()

    monkeypatch.setattr(webui_server, "create_state_dict_provider", lambda **_k: provider)
    monkeypatch.setattr(webui_server, "_handler_factory", lambda _s: object)
    monkeypatch.setattr(webui_server.tempfile, "gettempdir", lambda: str(tmp_path))
    monkeypatch.setattr(
        webui_server,
        "serve_http",
        lambda **kwargs: (
            events.extend(["serve", "close"]),
            kwargs["on_close"](),
        ),
    )
    webui_server._serve_webui(host="127.0.0.1", port=9022)
    assert events == ["serve", "close"]
    assert provider.closed is True
