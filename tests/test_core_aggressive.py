from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from brainsurgery.core import (
    BaseTransform,
    BinaryMappingSpec,
    BinaryMappingTransform,
    DeclarativeBinaryTransform,
    DeclarativeTernaryTransform,
    DeclarativeUnaryTransform,
    DestinationPolicy,
    Docs,
    TensorRef,
    TernaryMappingSpec,
    TernaryMappingTransform,
    TransformError,
    TypedTransform,
    UnarySpec,
    UnaryTransform,
)
from brainsurgery.core.compile.expression import compile_tensor_ref_expr, get_assert_expr_help
from brainsurgery.core.compile.matching import StructuredMatch, _MatchError, _StructuredPathMatcher
from brainsurgery.core.compile.name_mapping import resolve_name_mappings
from brainsurgery.core.compile.resolver import _resolve_target_names, _resolve_tensors
from brainsurgery.core.runtime.transform import REGISTRY, register_transform
from brainsurgery.core.specs.scalar_comparison import ScalarComparison, parse_scalar_comparison
from brainsurgery.core.specs.types import StateDictLike
from brainsurgery.core.specs.validation import (
    TransformPayloadSchema,
    require_expr,
    require_same_shape_dtype_device,
    require_same_shape_dtype_device3,
)
from brainsurgery.engine.state_dicts import _InMemoryStateDict


class _Provider:
    def __init__(self) -> None:
        self.state_dicts = {
            "a": _InMemoryStateDict(),
            "b": _InMemoryStateDict(),
            "dst": _InMemoryStateDict(),
        }
        self.state_dicts["a"]["x.0"] = torch.ones(2, 2)
        self.state_dicts["a"]["x.1"] = torch.zeros(2, 2)
        self.state_dicts["b"]["y.0"] = torch.ones(2, 2)
        self.state_dicts["dst"]["z.0"] = torch.ones(2, 2)

    def get_state_dict(self, model: str) -> _InMemoryStateDict:
        return self.state_dicts[model]


class _CoreError(TransformError):
    pass


class _NoDocsUnary(DeclarativeUnaryTransform[UnarySpec]):
    name = "nodocs_u"
    apply_fn = staticmethod(lambda spec, name, provider: None)


class _NoDocsBinary(DeclarativeBinaryTransform[BinaryMappingSpec]):
    name = "nodocs_b"
    apply_fn = staticmethod(lambda spec, src_name, dst_name, provider: None)


class _NoDocsTernary(DeclarativeTernaryTransform[TernaryMappingSpec]):
    name = "nodocs_t"
    apply_fn = staticmethod(lambda spec, src_a_name, src_b_name, dst_name, provider: None)


class _DocUnary(DeclarativeUnaryTransform[UnarySpec]):
    name = "doc_u"
    docs = Docs(summary="doc unary")
    apply_fn = staticmethod(lambda spec, name, provider: None)


class _DocBinaryAny(DeclarativeBinaryTransform[BinaryMappingSpec]):
    name = "doc_b_any"
    docs = Docs(summary="doc binary any")
    destination_policy = DestinationPolicy.ANY
    apply_fn = staticmethod(lambda spec, src_name, dst_name, provider: None)


class _DocTernaryAny(DeclarativeTernaryTransform[TernaryMappingSpec]):
    name = "doc_t_any"
    docs = Docs(summary="doc ternary any")
    destination_policy = DestinationPolicy.ANY
    apply_fn = staticmethod(lambda spec, src_a_name, src_b_name, dst_name, provider: None)


def test_core_base_abstract_methods_raise() -> None:
    with pytest.raises(NotImplementedError):
        BaseTransform.compile(object(), {}, None)
    with pytest.raises(NotImplementedError):
        BaseTransform.apply(object(), object(), object())
    with pytest.raises(NotImplementedError):
        BaseTransform._infer_output_model(object(), object())

    with pytest.raises(NotImplementedError):
        StateDictLike.slot(object(), "x")
    with pytest.raises(NotImplementedError):
        StateDictLike.bind_slot(object(), "x", object())
    with pytest.raises(NotImplementedError):
        StateDictLike.access_counts(object(), "x")
    with pytest.raises(NotImplementedError):
        StateDictLike.mark_write(object(), "x")


def test_typed_transform_require_spec_and_register_transform_validation() -> None:
    class _Typed(TypedTransform[int]):
        name = "typed"
        spec_type = int

        def payload_schema(self) -> TransformPayloadSchema:
            return TransformPayloadSchema(
                mode_key=None,
                default_mode="default",
                common_required=set(),
                common_allowed=set(),
            )

        def compile(self, payload: dict, default_model: str | None) -> int:
            del payload, default_model
            return 1

        def apply(self, spec: object, provider: object):
            del provider
            return self.require_spec(spec)

        def _infer_output_model(self, spec: object) -> str:
            del spec
            return "a"

    typed = _Typed()
    with pytest.raises(TransformError, match="expected int"):
        typed.require_spec("bad")

    class _NoName(BaseTransform):
        name = ""

        def payload_schema(self) -> TransformPayloadSchema:
            return TransformPayloadSchema(
                mode_key=None,
                default_mode="default",
                common_required=set(),
                common_allowed=set(),
            )

        def compile(self, payload: dict, default_model: str | None):
            del payload, default_model
            return {}

        def apply(self, spec: object, provider: object):
            del spec, provider
            return object()

        def _infer_output_model(self, spec: object) -> str:
            del spec
            return "a"

    with pytest.raises(TransformError, match="non-empty string 'name'"):
        register_transform(_NoName())

    snapshot = dict(REGISTRY)
    try:
        with pytest.raises(TransformError, match="already registered"):
            register_transform(next(iter(snapshot.values())))
    finally:
        REGISTRY.clear()
        REGISTRY.update(snapshot)


class _Unary(UnaryTransform[UnarySpec]):
    name = "u"
    spec_type = UnarySpec
    slice_policy = "allow"

    def apply_to_target(self, spec: UnarySpec, name: str, provider: _Provider) -> None:
        del spec, name, provider


class _BinaryNoApply(BinaryMappingTransform[BinaryMappingSpec]):
    name = "b"
    spec_type = BinaryMappingSpec

    def validate_refs(self, from_ref: TensorRef, to_ref: TensorRef) -> None:
        del from_ref, to_ref


@dataclass(frozen=True)
class _TernarySpec(TernaryMappingSpec):
    pass


class _Ternary(TernaryMappingTransform[_TernarySpec]):
    name = "t"
    spec_type = _TernarySpec
    destination_policy = DestinationPolicy.ANY

    def __init__(self) -> None:
        self.applied: list[str] = []

    def validate_refs(
        self, from_a_ref: TensorRef, from_b_ref: TensorRef, to_ref: TensorRef
    ) -> None:
        del from_a_ref, from_b_ref, to_ref

    def apply_mapping(
        self,
        spec: _TernarySpec,
        src_a_name: str,
        src_b_name: str,
        dst_name: str,
        provider: _Provider,
    ) -> None:
        del spec, src_a_name, src_b_name, provider
        self.applied.append(dst_name)


def test_core_transform_infer_and_apply_error_paths() -> None:
    unary = _Unary()
    with pytest.raises(TransformError, match="output model missing"):
        unary._infer_output_model(UnarySpec(target_ref=TensorRef(model=None, expr="x")))

    binary = _BinaryNoApply()
    with pytest.raises(TransformError, match="output model missing"):
        binary._infer_output_model(
            BinaryMappingSpec(
                from_ref=TensorRef(model="a", expr="x"),
                to_ref=TensorRef(model=None, expr="y"),
            )
        )

    item = binary.build_spec(TensorRef(model="a", expr="x"), TensorRef(model="dst", expr="z"))
    with pytest.raises(NotImplementedError):
        binary.apply_item(item, ("x", "z"), _Provider())

    ternary = _Ternary()
    with pytest.raises(TransformError, match="output model missing"):
        ternary._infer_output_model(
            _TernarySpec(
                from_a_ref=TensorRef(model="a", expr="x"),
                from_b_ref=TensorRef(model="b", expr="y"),
                to_ref=TensorRef(model=None, expr="z"),
            )
        )

    mapping = ("x.0", "y.0", "z.1")
    ternary.apply_item(
        _TernarySpec(
            from_a_ref=TensorRef(model="a", expr="x"),
            from_b_ref=TensorRef(model="b", expr="y"),
            to_ref=TensorRef(model="dst", expr="z"),
        ),
        mapping,
        _Provider(),
    )
    assert ternary.applied == ["z.1"]


def test_core_ternary_transform_error_branches() -> None:
    provider = _Provider()
    t = _Ternary()

    with pytest.raises(TransformError, match="matched zero tensors"):
        t.resolve_items(
            _TernarySpec(
                from_a_ref=TensorRef(model="a", expr=r"missing\..*"),
                from_b_ref=TensorRef(model="b", expr=r"y.\g<0>"),
                to_ref=TensorRef(model="dst", expr=r"z.\g<0>"),
            ),
            provider,
        )

    with pytest.raises(TransformError, match="invalid regex rewrite"):
        t.resolve_items(
            _TernarySpec(
                from_a_ref=TensorRef(model="a", expr=r"x\.(\d+)"),
                from_b_ref=TensorRef(model="b", expr=r"y.\2"),
                to_ref=TensorRef(model="dst", expr=r"z.\1"),
            ),
            provider,
        )

    with pytest.raises(TransformError, match="source_b missing"):
        t.resolve_items(
            _TernarySpec(
                from_a_ref=TensorRef(model="a", expr=r"x\.(\d+)"),
                from_b_ref=TensorRef(model="b", expr=r"missing.\1"),
                to_ref=TensorRef(model="dst", expr=r"z.\1"),
            ),
            provider,
        )

    provider.state_dicts["b"]["y.1"] = torch.ones(2, 2)
    with pytest.raises(TransformError, match="destination collision"):
        t.resolve_items(
            _TernarySpec(
                from_a_ref=TensorRef(model="a", expr=r"x\.(\d+)"),
                from_b_ref=TensorRef(model="b", expr=r"y.\1"),
                to_ref=TensorRef(model="dst", expr="same"),
            ),
            provider,
        )

    with pytest.raises(TransformError, match="source_b missing"):
        t.resolve_items(
            _TernarySpec(
                from_a_ref=TensorRef(model="a", expr=["x", "$i"]),
                from_b_ref=TensorRef(model="b", expr=["missing", "${i}"]),
                to_ref=TensorRef(model="dst", expr=["z", "${i}"]),
            ),
            provider,
        )

    with pytest.raises(TransformError, match="destination collision"):
        t.resolve_items(
            _TernarySpec(
                from_a_ref=TensorRef(model="a", expr=["x", "$i"]),
                from_b_ref=TensorRef(model="b", expr=["y", "${i}"]),
                to_ref=TensorRef(model="dst", expr=["same"]),
            ),
            provider,
        )

    with pytest.raises(TransformError, match="matched zero tensors"):
        t.resolve_items(
            _TernarySpec(
                from_a_ref=TensorRef(model="a", expr=["missing", "$i"]),
                from_b_ref=TensorRef(model="b", expr=["y", "${i}"]),
                to_ref=TensorRef(model="dst", expr=["z", "${i}"]),
            ),
            provider,
        )

    with pytest.raises(TransformError, match="same kind"):
        t.resolve_items(
            _TernarySpec(
                from_a_ref=TensorRef(model="a", expr=r"x\.(\d+)"),
                from_b_ref=TensorRef(model="b", expr=["y", "${i}"]),
                to_ref=TensorRef(model="dst", expr=r"z.\1"),
            ),
            provider,
        )

    t.destination_policy = DestinationPolicy.MUST_NOT_EXIST
    mappings = [("x.0", "y.0", "z.0")]
    with pytest.raises(TransformError, match="destination already exists"):
        t.validate_resolved_items(
            _TernarySpec(
                from_a_ref=TensorRef(model="a", expr=r"x\.(\d+)"),
                from_b_ref=TensorRef(model="b", expr=r"y.\1"),
                to_ref=TensorRef(model="dst", expr=r"z.\1"),
            ),
            mappings,
            provider,
        )


def test_core_resolver_and_expression_error_paths() -> None:
    provider = _Provider()

    with pytest.raises(_CoreError, match="boom"):
        _resolve_target_names(
            target_ref=TensorRef(model="a", expr=r"x\..*"),
            provider=provider,
            op_name="assert",
            match_names=lambda **kwargs: (_ for _ in ()).throw(TransformError("boom")),
            error_type=_CoreError,
        )

    assert (
        _resolve_tensors(
            TensorRef(model="a", expr=r"x\..*"),
            provider,
            op_name="assert",
            resolve_names=lambda ref, p, op_name: [],
        )
        == []
    )

    with pytest.raises(TransformError, match="unknown assert op"):
        get_assert_expr_help("missing-op")

    assert isinstance(get_assert_expr_help(), dict)

    with pytest.raises(TransformError, match="non-empty string reference"):
        compile_tensor_ref_expr(["ok", ""], default_model="a", op_name="assert.equal")


def test_core_scalar_comparison_and_validation_error_paths() -> None:
    cmp = ScalarComparison(exact=2, ge=1, gt=None, le=3, lt=None)
    assert cmp.describe() == "exactly 2 and >= 1 and <= 3"
    assert cmp.matches(2) is True
    assert cmp.matches(4) is False
    assert ScalarComparison(exact=None, ge=None, gt=2, le=None, lt=None).describe() == "> 2"
    assert ScalarComparison(exact=None, ge=None, gt=2, le=None, lt=None).matches(2) is False
    assert ScalarComparison(exact=None, ge=None, gt=None, le=2, lt=None).matches(3) is False
    assert ScalarComparison(exact=None, ge=None, gt=None, le=None, lt=2).matches(2) is False

    with pytest.raises(TransformError, match="conflicts with"):
        parse_scalar_comparison(
            {"is": 1, "eq": 2},
            op_name="count",
            aliases={"eq": "is"},
        )

    with pytest.raises(TransformError, match="contradictory bounds"):
        parse_scalar_comparison({"gt": 5, "le": 5}, op_name="count")
    with pytest.raises(TransformError, match="contradictory bounds"):
        parse_scalar_comparison({"gt": 5, "le": 4}, op_name="count")

    with pytest.raises(TransformError, match="cannot be smaller"):
        parse_scalar_comparison({"is": 1, "ge": 2}, op_name="count")
    with pytest.raises(TransformError, match="must be greater"):
        parse_scalar_comparison({"is": 1, "gt": 1}, op_name="count")
    with pytest.raises(TransformError, match="cannot be larger"):
        parse_scalar_comparison({"is": 3, "le": 2}, op_name="count")
    with pytest.raises(TransformError, match="must be less"):
        parse_scalar_comparison({"is": 3, "lt": 3}, op_name="count")

    with pytest.raises(TransformError, match="non-empty string"):
        require_expr({"x": ""}, op_name="copy", key="x")
    with pytest.raises(TransformError, match="non-empty string"):
        require_expr({"x": ["a", ""]}, op_name="copy", key="x")
    with pytest.raises(TransformError, match="non-empty string"):
        require_expr({"x": 1}, op_name="copy", key="x")

    if torch.device("meta") != torch.device("cpu"):
        with pytest.raises(TransformError, match="device mismatch"):
            require_same_shape_dtype_device(
                torch.ones(2, device="cpu"),
                torch.ones(2, device="meta"),
                op_name="assign",
                left_name="a",
                right_name="b",
            )

        with pytest.raises(TransformError, match="device mismatch"):
            require_same_shape_dtype_device3(
                torch.ones(2, device="cpu"),
                torch.ones(2, device="meta"),
                torch.ones(2, device="cpu"),
                op_name="add",
                first_name="a",
                second_name="b",
                dest_name="dst",
                symbol="+",
            )


def test_core_matching_uncovered_paths() -> None:
    matcher = _StructuredPathMatcher()
    assert (
        matcher.match_and_rewrite(
            from_pattern=["x", "$i"],
            to_pattern=["y", "${i}"],
            name="x.1",
        )
        == "y.1"
    )

    assert matcher.match(["$x", "$x"], "a.a") == StructuredMatch(bindings={"x": "a"})
    assert matcher.match(["*tail", "x", "*tail"], "a.x.a") == StructuredMatch(
        bindings={"tail": ["a"]}
    )
    assert matcher.match(["~[0-9]+"], "7") == StructuredMatch(bindings={})

    with pytest.raises(_MatchError, match="invalid regex binding list"):
        matcher.match(["~x,,y::(a)(b)"], "ab")
    assert matcher.match(["~[0-9]+"], "x") is None
    assert matcher.match(["~x::x"], "x") == StructuredMatch(bindings={"x": "x"})
    with pytest.raises(_MatchError, match="binds 1 variable but regex has 2 capturing groups"):
        matcher.match(["~x::(a)(b)"], "ab")
    assert matcher.match(["~x,y::(a)(b)"], "ab") == StructuredMatch(bindings={"x": "a", "y": "b"})
    assert matcher.match(["~x,x::(a)(b)"], "ab") is None
    assert matcher.match(["a"], "a.b") is None
    assert matcher.match(["*x", "*x", "z"], "a.z") is None
    assert matcher.match(["a", "$x"], "a") is None

    with pytest.raises(_MatchError, match="cannot interpolate non-scalar"):
        matcher.rewrite(["${x}"], StructuredMatch(bindings={"x": ["a"]}))


def test_core_name_mapping_structured_collision_and_zero_match() -> None:
    provider = _Provider()
    provider.state_dicts["a"]["x.1"] = torch.ones(2, 2)

    with pytest.raises(TransformError, match="destination collision"):
        resolve_name_mappings(
            from_ref=TensorRef(model="a", expr=["x", "$i"]),
            to_ref=TensorRef(model="dst", expr=["same"]),
            provider=provider,
            op_name="copy",
        )

    with pytest.raises(TransformError, match="source matched zero tensors"):
        resolve_name_mappings(
            from_ref=TensorRef(model="a", expr=["missing", "$i"]),
            to_ref=TensorRef(model="dst", expr=["z", "${i}"]),
            provider=provider,
            op_name="copy",
        )


def test_core_resolver_slice_happy_path() -> None:
    provider = _Provider()
    provider.state_dicts["a"]["x.2"] = torch.arange(6).reshape(2, 3)
    tensor = _resolve_tensors(
        TensorRef(model="a", expr=r"x\.2", slice_spec="[:, :2]"),
        provider,
        op_name="assert",
        resolve_names=lambda ref, p, op_name: ["x.2"],
    )[0][1]
    assert tensor.shape == (2, 2)


def test_core_declarative_uncovered_paths() -> None:
    no_docs_u = _NoDocsUnary()
    no_docs_b = _NoDocsBinary()
    no_docs_t = _NoDocsTernary()
    assert no_docs_u.name == "nodocs_u"
    assert no_docs_b.name == "nodocs_b"
    assert no_docs_t.name == "nodocs_t"

    doc_u = _DocUnary()
    spec = doc_u.build_spec(TensorRef(model="a", expr="x"), {})
    assert spec == UnarySpec(target_ref=TensorRef(model="a", expr="x"))

    assert "may be created or overwritten" in _DocBinaryAny.help_text
    assert "may be created or overwritten" in _DocTernaryAny.help_text

    with pytest.raises(NotImplementedError):
        _DocBinaryAny().apply_mapping(object(), "x", "y", _Provider())  # type: ignore[arg-type]
    with pytest.raises(NotImplementedError):
        _DocTernaryAny().apply_mapping(object(), "a", "b", "c", _Provider())  # type: ignore[arg-type]


def test_core_ternary_validate_any_policy_returns_early() -> None:
    provider = _Provider()
    t = _Ternary()
    t.validate_resolved_items(
        _TernarySpec(
            from_a_ref=TensorRef(model="a", expr=r"x\.(\d+)"),
            from_b_ref=TensorRef(model="b", expr=r"y.\1"),
            to_ref=TensorRef(model="dst", expr=r"z.\1"),
        ),
        [("x.0", "y.0", "z.0")],
        provider,
    )
