from pathlib import Path

import pytest
import torch

import brainsurgery.transforms.load as load_module
from brainsurgery.core import TensorRef
from brainsurgery.engine import (
    RuntimeFlagLifecycleScope,
    reset_runtime_flags_for_scope,
    set_runtime_flag,
)
from brainsurgery.engine.providers import InMemoryStateDictProvider
from brainsurgery.engine.state_dicts import _InMemoryStateDict
from brainsurgery.transforms.load import LoadSpec, LoadTransform, LoadTransformError, ProviderError


def test_load_compile_defaults_alias_to_model_without_context() -> None:
    spec = LoadTransform().compile({"path": "/tmp/x.safetensors"}, default_model=None)
    assert spec.alias == "model"
    assert spec.tensor_name is None


def test_load_compile_to_conflict_raises() -> None:
    try:
        LoadTransform().compile(
            {"path": "/tmp/t.pt", "alias": "a", "to": "b::x"},
            default_model=None,
        )
    except LoadTransformError as exc:
        assert "conflicts" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected alias conflict error")


def test_load_rejects_non_auto_format_for_state_dict() -> None:
    try:
        LoadTransform().compile(
            {"path": "/tmp/x.safetensors", "format": "torch"},
            default_model="model",
        )
    except LoadTransformError as exc:
        assert "only supported for tensor loads" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("expected load.format validation error")


def test_load_compile_additional_validation_paths() -> None:
    spec = LoadTransform().compile("/tmp/x.safetensors", default_model=None)
    assert spec.path.name == "x.safetensors"

    with pytest.raises(LoadTransformError, match="load.alias must be a non-empty string"):
        LoadTransform().compile({"path": "/tmp/x.safetensors", "alias": ""}, default_model=None)
    with pytest.raises(LoadTransformError, match="load.format must be a non-empty string"):
        LoadTransform().compile({"path": "/tmp/x.safetensors", "format": ""}, default_model=None)
    with pytest.raises(LoadTransformError, match="one of: auto, torch, safetensors, numpy"):
        LoadTransform().compile(
            {"path": "/tmp/x.safetensors", "format": "weird"}, default_model=None
        )
    with pytest.raises(LoadTransformError, match="load.to must be a non-empty string"):
        LoadTransform().compile({"path": "/tmp/x.safetensors", "to": ""}, default_model=None)
    with pytest.raises(LoadTransformError, match="must not be sliced"):
        LoadTransform().compile(
            {"path": "/tmp/x.safetensors", "to": "a::x::[:1]"}, default_model=None
        )
    original_parse_model_expr = load_module.parse_model_expr
    try:
        load_module.parse_model_expr = lambda raw, default_model=None: TensorRef(
            model="a", expr=["x"], slice_spec=None
        )
        with pytest.raises(LoadTransformError, match="single tensor name"):
            LoadTransform().compile(
                {"path": "/tmp/x.safetensors", "to": "a::x"}, default_model=None
            )
    finally:
        load_module.parse_model_expr = original_parse_model_expr


def test_load_compile_backend_axon_validation_paths() -> None:
    tr = LoadTransform()
    with pytest.raises(Exception, match="load.path is required"):
        tr.compile({"backend": "axon"}, default_model=None)
    with pytest.raises(LoadTransformError, match="point to a .axon file"):
        tr.compile({"backend": "axon", "path": "/tmp/model.txt"}, default_model=None)
    with pytest.raises(LoadTransformError, match="unknown keys"):
        tr.compile(
            {"backend": "axon", "path": "/tmp/model.axon", "to": "m::x"},
            default_model=None,
        )
    with pytest.raises(LoadTransformError, match="not supported for axon model load mode"):
        tr.compile(
            {"backend": "axon", "path": "/tmp/model.axon", "format": "torch"},
            default_model=None,
        )
    with pytest.raises(LoadTransformError, match="must be one of: auto, axon, checkpoint, tensor"):
        tr.compile(
            {"backend": "invalid", "path": "/tmp/model.axon"},
            default_model=None,
        )
    with pytest.raises(LoadTransformError, match="requires load.to"):
        tr.compile({"backend": "tensor", "path": "/tmp/model.npy"}, default_model=None)
    spec = tr.compile(
        {"backend": "axon", "path": "/tmp/model.axon", "weights": "/tmp/w", "alias": "m"},
        default_model=None,
    )
    assert spec.backend == "axon"
    assert spec.mode == "axon"
    assert spec.path == Path("/tmp/model.axon")
    assert spec.weights == Path("/tmp/w")
    assert spec.alias == "m"


def test_load_apply_additional_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    state_spec = LoadSpec(
        path=Path("/tmp/x.safetensors"), alias="a", tensor_name=None, format="auto"
    )

    class _NoBaseProvider:
        pass

    with pytest.raises(LoadTransformError, match="supports creating new aliases"):
        LoadTransform().apply(state_spec, _NoBaseProvider())  # type: ignore[arg-type]

    provider = InMemoryStateDictProvider({}, max_io_workers=1)
    monkeypatch.setattr(
        provider,
        "load_alias_from_path",
        lambda alias, path: (_ for _ in ()).throw(ProviderError("model alias already exists")),
    )
    with pytest.raises(LoadTransformError, match="load alias already exists"):
        LoadTransform().apply(state_spec, provider)

    monkeypatch.setattr(
        provider,
        "load_alias_from_path",
        lambda alias, path: (_ for _ in ()).throw(RuntimeError("bad checkpoint")),
    )
    with pytest.raises(LoadTransformError, match="bad checkpoint"):
        LoadTransform().apply(state_spec, provider)

    tensor_spec = LoadSpec(path=Path("/tmp/t.npy"), alias="a", tensor_name="x", format="numpy")
    monkeypatch.setattr(
        load_module,
        "load_tensor_from_path",
        lambda path, format: (_ for _ in ()).throw(RuntimeError("bad tensor")),
    )
    with pytest.raises(LoadTransformError, match="bad tensor"):
        LoadTransform().apply(tensor_spec, provider)

    sd = _InMemoryStateDict()
    sd["x"] = torch.ones(1)
    monkeypatch.setattr(load_module, "load_tensor_from_path", lambda path, format: torch.zeros(1))
    monkeypatch.setattr(load_module, "get_or_create_alias_state_dict", lambda *args, **kwargs: sd)
    with pytest.raises(LoadTransformError, match="destination already exists"):
        LoadTransform().apply(tensor_spec, provider)


def test_load_apply_backend_axon_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    tr = LoadTransform()
    provider = InMemoryStateDictProvider({}, max_io_workers=1)

    missing_weights_spec = LoadSpec(
        path=Path("/tmp/model.axon"),
        alias="axon_model",
        tensor_name=None,
        format="auto",
        backend="axon",
        mode="axon",
        weights=None,
    )
    with pytest.raises(
        LoadTransformError, match="requires load.weights when alias does not already exist"
    ):
        tr.apply(missing_weights_spec, provider)

    called = {"loaded": False}
    sd = _InMemoryStateDict()
    sd["x"] = torch.ones(1)
    monkeypatch.setattr(
        provider,
        "load_alias_from_path",
        lambda alias, path: (called.__setitem__("loaded", True), sd)[1],
    )
    load_spec = LoadSpec(
        path=Path("/tmp/model.axon"),
        alias="axon_model",
        tensor_name=None,
        format="auto",
        backend="axon",
        mode="axon",
        weights=Path("/tmp/w.safetensors"),
    )
    result = tr.apply(load_spec, provider)
    assert result.count == 1
    assert called["loaded"] is True

    existing = provider.get_or_create_alias_state_dict("existing")
    existing["w"] = torch.zeros(1)
    conflict_spec = LoadSpec(
        path=Path("/tmp/model.axon"),
        alias="existing",
        tensor_name=None,
        format="auto",
        backend="axon",
        mode="axon",
        weights=Path("/tmp/w.safetensors"),
    )
    with pytest.raises(
        LoadTransformError, match="received both an existing alias and load.weights"
    ):
        tr.apply(conflict_spec, provider)

    reuse_spec = LoadSpec(
        path=Path("/tmp/model.axon"),
        alias="existing",
        tensor_name=None,
        format="auto",
        backend="axon",
        mode="axon",
        weights=None,
    )
    reuse_result = tr.apply(reuse_spec, provider)
    assert reuse_result.count == 0


def test_load_apply_dry_run_uses_checkpoint_loader_and_tensor_alias_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = InMemoryStateDictProvider({}, max_io_workers=1)
    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)
    set_runtime_flag("dry_run", True)

    calls: list[str] = []
    sd = _InMemoryStateDict()
    sd["x"] = torch.ones(1)

    monkeypatch.setattr(
        provider,
        "load_state_dict_from_checkpoint_path",
        lambda path: (calls.append("checkpoint"), sd)[1],
    )
    monkeypatch.setattr(
        provider,
        "load_alias_from_path",
        lambda alias, path: (_ for _ in ()).throw(
            AssertionError("should not call load_alias_from_path")
        ),
    )

    state_spec = LoadSpec(
        path=Path("/tmp/x.safetensors"), alias="a", tensor_name=None, format="auto"
    )
    result = LoadTransform().apply(state_spec, provider)
    assert result.count == 1
    assert calls == ["checkpoint"]
    assert provider.has_model_alias("a") is False

    tensor_spec = LoadSpec(
        path=Path("/tmp/t.npy"), alias="new_alias", tensor_name="x", format="numpy"
    )
    monkeypatch.setattr(load_module, "load_tensor_from_path", lambda path, format: torch.zeros(1))
    result = LoadTransform().apply(tensor_spec, provider)
    assert result.count == 1
    assert provider.has_model_alias("new_alias") is False

    reset_runtime_flags_for_scope(RuntimeFlagLifecycleScope.CLI_RUN)


def test_load_completion_backend_candidates() -> None:
    tr = LoadTransform()
    assert tr.completion_value_candidates("backend", "a", []) == ["auto", "axon"]


def test_load_compile_auto_mode_infers_axon_vs_tensor_vs_checkpoint() -> None:
    tr = LoadTransform()
    axon_spec = tr.compile({"path": "/tmp/model.axon"}, default_model=None)
    assert axon_spec.mode == "axon"
    tensor_spec = tr.compile({"path": "/tmp/t.npy", "to": "m::x"}, default_model=None)
    assert tensor_spec.mode == "tensor"
    checkpoint_spec = tr.compile({"path": "/tmp/model.safetensors"}, default_model=None)
    assert checkpoint_spec.mode == "checkpoint"
