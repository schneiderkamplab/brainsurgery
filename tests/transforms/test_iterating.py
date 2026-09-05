from importlib import import_module

_module = import_module("brainsurgery.core")
DestinationPolicy = _module.DestinationPolicy
IteratingTransform = _module.IteratingTransform
StateDictProvider = _module.StateDictProvider
TransformPayloadSchema = _module.TransformPayloadSchema


def test_destination_policy_values_are_stable() -> None:
    assert DestinationPolicy.ANY.value == "any"
    assert DestinationPolicy.MUST_EXIST.value == "must_exist"
    assert DestinationPolicy.MUST_NOT_EXIST.value == "must_not_exist"


def test_iterating_transform_default_validation_is_noop() -> None:
    class _Transform(IteratingTransform[int, int]):
        name = "dummy"
        spec_type = int

        def payload_schema(self) -> TransformPayloadSchema:
            return TransformPayloadSchema(
                mode_key=None,
                default_mode="default",
                common_required=set(),
                common_allowed=set(),
            )

        def compile(self, payload: dict, default_model: str | None) -> object:
            del default_model
            return int(payload["value"])

        def _infer_output_model(self, spec: object) -> str:
            del spec
            return "model"

        def resolve_items(self, spec: int, provider: StateDictProvider) -> list[int]:
            del provider
            return [spec]

        def apply_item(self, spec: int, item: int, provider: StateDictProvider) -> None:
            del spec, item, provider

    result = _Transform().apply(1, provider=None)  # type: ignore[arg-type]
    assert result.name == "dummy"
    assert result.count == 1
