from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import pytest

from brainsurgery.synapse import (
    lower_axon_module_to_synapse_spec,
    lower_axon_program_to_synapse_spec,
    parse_axon_module,
    parse_axon_program,
    parse_axon_program_from_path,
    synapse_spec_to_axon_module_text,
)


def _node_specs(graph: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for item in graph:
        assert isinstance(item, dict) and len(item) == 1
        _, node_spec = next(iter(item.items()))
        assert isinstance(node_spec, dict)
        out.append(node_spec)
    return out


def test_parse_axon_module_header_and_bindings() -> None:
    source = """
tiny :: Tensor -> ?Tensor -> Tensor
tiny x cache = do
  y <- x |> linear@proj |> _activations_gelu_new
  return y
"""
    module = parse_axon_module(source)
    assert module.name == "tiny"
    assert [param.name for param in module.params] == ["x", "cache"]
    assert [param.optional for param in module.params] == [False, True]
    assert module.returns == ()
    assert len(module.statements) == 2


def test_parse_axon_module_expression_definition_without_do() -> None:
    source = """
inc :: Int -> Int
inc x = x + 1
"""
    module = parse_axon_module(source)
    assert module.name == "inc"
    assert [param.name for param in module.params] == ["x"]
    assert len(module.statements) == 1
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    ops = [node["_op"] for node in node_specs]
    assert "add" in ops


def test_bind_rhs_do_expression_parses_and_lowers() -> None:
    source = """
main :: Tensor -> Tensor
main x = do
  y <- do
    z <- x + 1
    return z
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    ops = [node["_op"] for node in node_specs]
    assert "add" in ops
    assert spec["model"]["outputs"] == {"y": "y"}


def test_bind_rhs_inline_do_expression_parses_and_lowers() -> None:
    source = """
main :: Tensor -> Tensor
main x = do
  y <- do z <- x + 1; return z
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    ops = [node["_op"] for node in node_specs]
    assert "add" in ops
    assert spec["model"]["outputs"] == {"y": "y"}


def test_module_rhs_inline_do_expression_parses_and_lowers() -> None:
    source = """
main :: Tensor -> Tensor
main x = do y <- x + 1; k <- y * y; return k + y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    ops = [node["_op"] for node in node_specs]
    assert "add" in ops
    assert spec["model"]["outputs"] == {"out_0": "out_0"}


def test_module_rhs_do_expression_supports_mixed_inline_and_newline_statements() -> None:
    source = """
main :: Tensor -> Tensor
main x = do y <- x + 1;
  k <- y + 2
  return k
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    ops = [node["_op"] for node in node_specs]
    assert "add" in ops
    assert spec["model"]["outputs"] == {"out_0": "out_0"}


def test_bind_rhs_do_expression_supports_mixed_inline_and_newline_statements() -> None:
    source = """
main :: Tensor -> Tensor
main x = do
  y <- do z <- x + 1;
    k <- z + 2
    return k
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    ops = [node["_op"] for node in node_specs]
    assert "add" in ops
    assert spec["model"]["outputs"] == {"y": "y"}


def test_bind_rhs_do_expression_requires_return_in_block() -> None:
    source = """
main :: Tensor -> Tensor
main x = do
  y <- do
    z <- x + 1
  return y
"""
    with pytest.raises(ValueError, match=r"do expression requires a reachable return"):
        parse_axon_module(source)


def test_primitive_activation_alias_lowering() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  y <- _activations_gelu_pytorch_tanh x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "activations_gelu_pytorch_tanh"


def test_pointfree_definition_is_eta_expanded() -> None:
    source = """
silu :: Tensor[B,T,D] -> Tensor[B,T,D]
silu = _activations_silu

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- silu x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    blocks = spec["model"]["blocks"]
    assert "silu" in blocks
    silu_nodes = _node_specs(blocks["silu"]["graph"])
    assert silu_nodes[0]["_op"] == "activations_silu"


def test_lowering_reports_shape_mismatch_from_signature_on_block_call() -> None:
    source = """
blk :: Tensor[B,T,768] -> Tensor[B,T,768]
blk x = do
  return x

main :: Tensor[B,T,640] -> Tensor[B,T,768]
main x = do
  y <- blk x
  return y
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match=r"shape mismatch in call 'blk'"):
        lower_axon_program_to_synapse_spec(modules, main_module="main")


def test_namespaced_module_call_with_import() -> None:
    source = """
import Lib

Lib.swiglu :: Tensor[B,T,D] -> Tensor[B,T,D]
Lib.swiglu x = do
  return _activations_silu x * x

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- Lib.swiglu x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "call"
    assert node_specs[0]["_target"] == "Lib.swiglu"


def test_namespaced_module_call_requires_import() -> None:
    source = """
Lib.swiglu :: Tensor[B,T,D] -> Tensor[B,T,D]
Lib.swiglu x = do
  return _activations_silu x * x

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- Lib.swiglu x
  return y
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match=r"requires `import Lib`"):
        lower_axon_program_to_synapse_spec(modules, main_module="main")


def test_builtin_activations_import_resolves_from_builtin_file(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Activations

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- Activations.swiglu x
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    first = node_specs[0]
    if first["_op"] == "call":
        assert first["_target"] == "Activations.swiglu"
    else:
        assert first["_op"] == "activations_swiglu"


def test_relative_import_beats_builtin_file(tmp_path: Path) -> None:
    (tmp_path / "Activations.axon").write_text(
        """
swiglu :: Tensor[B,T,D] -> Tensor[B,T,D]
swiglu x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Activations

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- Activations.swiglu x
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    names = {module.name for module in modules}
    assert "Activations.swiglu" in names


def test_import_uses_axon_path_before_builtins(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    extra_dir = tmp_path / "axon_path"
    extra_dir.mkdir()
    (extra_dir / "Lib.axon").write_text(
        """
swiglu :: Tensor[B,T,D] -> Tensor[B,T,D]
swiglu x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AXON_PATH", str(extra_dir))
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Lib

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- Lib.swiglu x
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    names = {module.name for module in modules}
    assert "Lib.swiglu" in names


def test_builtin_xielu_path_binding_lowers_to_activation_params(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Activations xielu

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- xielu@act_fn x
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    first = node_specs[0]
    if first["_op"] == "call":
        assert first["_target"] == "Activations.xielu"
    else:
        assert first["_op"] == "activations_xielu"
        params = first.get("_params")
        assert isinstance(params, dict)
        assert params.get("alpha_p") == "act_fn.alpha_p"
        assert params.get("alpha_n") == "act_fn.alpha_n"
        assert params.get("beta") == "act_fn.beta"
        assert params.get("eps") == "act_fn.eps"


def test_builtin_position_import_resolves_relative_bias_alias(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Position

main :: Tensor[B,H,T,HD] -> Tensor[B,H,TK,HD] -> Tensor[1,H,T,TK]
main q k = do
  bias <- Position.relative_bias_t5 q k num_buckets=32 max_distance=128 bidirectional=true
  return bias
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    first = node_specs[0]
    if first["_op"] == "call":
        assert first["_target"] == "Position.relative_bias_t5"
    else:
        assert first["_op"] == "t5_relative_position_bias"
        assert first["num_buckets"] == 32
        assert first["max_distance"] == 128
        assert first["bidirectional"] is True


def test_builtin_position_member_import_resolves_disentangled_relative_bias(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Position (relative_bias_disentangled)

main :: Tensor[B,H,T,HD] -> Tensor[B,H,TK,HD] -> Tensor[B,H,T,TK]
main q k = do
  bias <- relative_bias_disentangled q k share_att_key=true c2p=true p2c=true
  return bias
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    first = node_specs[0]
    if first["_op"] == "call":
        assert first["_target"] == "Position.relative_bias_disentangled"
    else:
        assert first["_op"] == "disentangled_relative_bias"
        assert first["share_att_key"] is True
        assert first["c2p"] is True
        assert first["p2c"] is True


def test_builtin_cache_import_resolves_from_builtin_file(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Cache

main :: ?CacheLayer -> Tensor[B,H,T,D] -> Tensor[B,H,T,D] -> ?Bool -> ?Cache
main past k v use_cache = do
  k_ctx, v_ctx, present <- Cache.update past k v
  cache <- Cache.init
  cache <- use_cache ? Cache.append cache present : cache
  return cache
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    first = node_specs[0]
    if first["_op"] == "call":
        assert first["_target"] == "Cache.update"
    else:
        assert first["_op"] == "cache_update"
    second = node_specs[1]
    if second["_op"] == "call":
        assert second["_target"] == "Cache.init"
    else:
        assert second["_op"] == "list_init"
    third = node_specs[2]
    if third["_op"] == "select":
        assert third["cond"] == "use_cache"
        then_specs = _node_specs(third["_then"])
        assert len(then_specs) == 1
        then_spec = then_specs[0]
        if then_spec["_op"] == "call":
            assert then_spec["_target"] == "Cache.append"
        else:
            assert then_spec["_op"] == "list_append"
    elif third["_op"] == "call":
        assert third["_target"] == "Cache.append"
    else:
        assert third["_op"] == "list_append"


def test_import_resolution_prefers_local_file_over_builtins(tmp_path: Path) -> None:
    local_act = tmp_path / "Activations.axon"
    local_act.write_text(
        """
Activations.swiglu :: Tensor[B,T,D] -> Tensor[B,T,D]
Activations.swiglu x = do
  return _activations_relu x
""".strip()
        + "\n",
        encoding="utf-8",
    )
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Activations

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- Activations.swiglu x
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "activations_relu"


def test_selective_import_parenthesized_brings_member_into_scope(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Activations (gelu_new)

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- gelu_new x
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    first = node_specs[0]
    if first["_op"] == "call":
        assert first["_target"] == "Activations.gelu_new"
    else:
        assert first["_op"] == "activations_gelu_new"


def test_selective_import_shorthand_brings_member_into_scope(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Activations gelu_new

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- gelu_new x
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    first = node_specs[0]
    if first["_op"] == "call":
        assert first["_target"] == "Activations.gelu_new"
    else:
        assert first["_op"] == "activations_gelu_new"


def test_local_module_name_shadows_selective_import() -> None:
    source = """
import Activations (gelu_new)

gelu_new :: Tensor[B,T,D] -> Tensor[B,T,D]
gelu_new x = do
  return x

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- gelu_new x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "call"
    assert node_specs[0]["_target"] == "gelu_new"


def test_parse_program_from_path_loads_imported_axon_modules(tmp_path: Path) -> None:
    lib_path = tmp_path / "Lib.axon"
    lib_path.write_text(
        """
Lib.id :: Tensor[B,T,D] -> Tensor[B,T,D]
Lib.id x = do
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Lib

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- Lib.id x
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "call"
    assert node_specs[0]["_target"] == "Lib.id"


def test_prelude_is_implicitly_available_from_file_parse(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- gelu_new x
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    first = node_specs[0]
    if first["_op"] == "call":
        assert first["_target"] == "Prelude.gelu_new"
    else:
        assert first["_op"] == "activations_gelu_new"


def test_prelude_does_not_override_native_linear_op_semantics(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- linear@proj x dim=16 bias=true transpose=true
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "linear"
    assert node_specs[0]["dim"] == 16
    assert node_specs[0]["bias"] is True
    assert node_specs[0]["transpose"] is True


def test_cache_builtin_import_resolves_from_builtin_file(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Cache

main :: ?CacheLayer -> Tensor[B,H,T,D] -> Tensor[B,H,T,D] -> ?Bool -> (Tensor[B,H,T,D], Tensor[B,H,T,D], ?CacheLayer)
main past k v use_cache = do
  k_ctx, v_ctx, present <- Cache.update past k v
  return k_ctx, v_ctx, present
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    first = node_specs[0]
    if first["_op"] == "call":
        assert first["_target"] == "Cache.update"
    else:
        assert first["_op"] == "cache_update"


def test_moe_builtin_import_resolves_from_builtin_file(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import MoE

main :: Tensor[B,T,D] -> Tensor[B,T,K] -> Tensor[B,T,K] -> Int -> (Tensor[N,D], Tensor[N], Tensor[N], Tensor[N])
main hidden topk_scores topk_indices expert = do
  selected_hidden, token_idx, topk_pos, selected_scores <- MoE.select hidden topk_scores topk_indices expert
  return selected_hidden, token_idx, topk_pos, selected_scores
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    first = node_specs[0]
    if first["_op"] == "call":
        assert first["_target"] == "MoE.select"
    else:
        assert first["_op"] in {"moe_select", "_moe_select"}


def test_moe_grouped_ffn_builtin_import_resolves_from_builtin_file(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import MoE

main :: Tensor[B,T,D] -> Tensor[B,T,K] -> Tensor[B,T,K] -> Tensor[B,T,D]
main hidden topk_scores topk_indices = do
  out <- MoE.grouped_ffn hidden topk_scores topk_indices
  return out
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    first = node_specs[0]
    if first["_op"] == "call":
        assert first["_target"] == "MoE.grouped_ffn"
    else:
        assert first["_op"] in {"moe_grouped_ffn", "_moe_grouped_ffn"}


def test_config_builtin_import_resolves_and_lowers_default_kwarg(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Config

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  dim <- Config.int "hidden_size" default=640
  y <- linear@proj x dim=dim
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "config_int"
    assert node_specs[0]["_args"] == "hidden_size"
    assert node_specs[0]["default"] == 640
    assert node_specs[1]["_op"] == "linear"
    assert node_specs[1]["dim"] == "dim"


def test_config_builtin_import_resolves_and_lowers_value_primitive(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Config

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  factors <- Config.value "rope_scaling.long_factor" default=[]
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "config_value"
    assert node_specs[0]["_args"] == "rope_scaling.long_factor"
    assert node_specs[0]["default"] == []


def test_params_builtin_import_resolves_and_lowers_root_primitives(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Params

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  has_lm <- Params.has_root "language_model"
  root <- Params.root "language_model" default=""
  return x
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "params_has_root"
    assert node_specs[0]["_args"] == "language_model"
    assert node_specs[1]["_op"] == "params_root"
    assert node_specs[1]["_args"] == "language_model"
    assert node_specs[1]["default"] == ""


def test_imported_constant_includes_transitive_constant_dependencies(tmp_path: Path) -> None:
    cfg_path = tmp_path / "cfg.axon"
    cfg_path.write_text(
        """
import Config

CFG = (Config.has "text_config") ? "text_config" : ""
D = Config.int "hidden_size" root=CFG default=640
""".strip()
        + "\n",
        encoding="utf-8",
    )
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import cfg (D)

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- linear@proj x dim=D
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    cfg_nodes = [node for node in node_specs if node.get("_op") in {"config_has", "config_int"}]
    assert [node["_op"] for node in cfg_nodes] == ["config_has", "config_int"]
    assert cfg_nodes[1]["_args"] == "hidden_size"
    assert cfg_nodes[1]["root"] == "CFG"
    assert cfg_nodes[1]["default"] == 640


def test_multi_path_parameters_support_triple_at_call_syntax() -> None:
    source = """
expert_ffn :: @Path -> @Path -> @Path -> Tensor[B,T,D] -> Tensor[B,T,D]
expert_ffn@gate@up@down x = do
  g <- linear@gate x
  u <- linear@up x
  y <- g |> mul u |> linear@down
  return y

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- expert_ffn@mlp.gate_proj@mlp.up_proj@mlp.down_proj x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "call"
    assert node_specs[0]["_target"].startswith("expert_ffn__")
    assert "gate" not in node_specs[0]
    assert "up" not in node_specs[0]
    assert "down" not in node_specs[0]
    specialized_block_name = node_specs[0]["_target"]
    specialized_nodes = _node_specs(spec["model"]["blocks"][specialized_block_name]["graph"])
    assert specialized_nodes[0]["_params"]["weight"] == "mlp.gate_proj.weight"
    assert specialized_nodes[1]["_params"]["weight"] == "mlp.up_proj.weight"
    assert specialized_nodes[3]["_params"]["weight"] == "mlp.down_proj.weight"


def test_top_level_constant_stays_symbol_with_expression_module_definition() -> None:
    source = """
D = 7
inc :: Int -> Int
inc x = x + D
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    assert spec["model"]["symbols"]["D"] == 7


def test_top_level_constant_supports_inline_sqrt_call_without_parentheses() -> None:
    source = """
D = 640
E = sqrt D

main :: Int -> Int
main x = x
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    symbols = spec["model"]["symbols"]
    assert symbols["D"] == 640
    assert symbols["E"] == pytest.approx(25.298221281347036)


def test_top_level_constant_supports_composed_inline_expressions() -> None:
    source = """
D = 640
E = 1.0 / sqrt D
F = (D / 64) + 3

main :: Int -> Int
main x = x
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    symbols = spec["model"]["symbols"]
    assert symbols["D"] == 640
    assert symbols["E"] == pytest.approx(0.03952847075210474)
    assert symbols["F"] == 13


def test_top_level_constant_sqrt_negative_argument_is_not_resolved_as_symbol() -> None:
    source = """
N = -1
E = sqrt N

main :: Int -> Int
main x = x
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    symbols = spec["model"]["symbols"]
    assert symbols["N"] == -1
    assert "E" not in symbols


def test_parse_repeat_block_statements() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  for@loop i <- [0..3) do
    y <- add x x
  return y
"""
    module = parse_axon_module(source)
    assert len(module.statements) == 2


def test_parse_rejects_legacy_node_statement() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  node n1 = {"_op":"add","_args":["x","x"],"_bind":"y"}
  return y
"""
    with pytest.raises(ValueError, match="invalid Axon source syntax"):
        parse_axon_module(source)


def test_parse_rejects_legacy_meta_statement() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  meta symbols = {"D":768}
  return x
"""
    with pytest.raises(ValueError, match="invalid Axon source syntax"):
        parse_axon_module(source)


def test_parse_and_lower_scope_bind_expression() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  y <- scope@attn do
    h <- linear@proj x dim=4
    return _activations_gelu_new h
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    graph = spec["model"]["graph"]
    assert isinstance(graph, list)

    def collect_ops(items: list[dict[str, Any]]) -> list[str]:
        ops: list[str] = []
        for item in items:
            _, node_spec = next(iter(item.items()))
            if not isinstance(node_spec, dict):
                continue
            op = node_spec.get("_op")
            if isinstance(op, str):
                ops.append(op)
            nested = node_spec.get("graph")
            if isinstance(nested, list):
                ops.extend(collect_ops(nested))
        return ops

    ops = collect_ops(graph)
    assert "linear" in ops
    assert "activations_gelu_new" in ops
    assert "_ir_alias" in ops
    scoped_nodes = [
        node_spec
        for item in graph
        for _, node_spec in item.items()
        if isinstance(node_spec, dict) and isinstance(node_spec.get("_op"), str)
    ]
    assert any(node.get("_scope") == "attn" for node in scoped_nodes)


def test_lowering_preserves_nested_scope_paths_on_ops() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  y <- scope@outer do
    z <- scope@inner do
      h <- _activations_gelu_new x
      return h
    return z
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    assert any(node.get("_scope") == "outer.inner" for node in node_specs if isinstance(node, dict))


def test_parenthesized_expression_argument_is_lowered() -> None:
    source = """
tiny :: Tensor[B,T,D] -> Tensor[B,T,D]
tiny x = do
  y <- add (zeros_like x) x
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    ops = [node["_op"] for node in node_specs]
    assert "zeros_like" in ops
    assert "add" in ops


def test_at_path_is_scoped_inside_scope_bind() -> None:
    source = """
tiny :: TokenIds[B,T] -> Tensor[B,T,V]
tiny input_ids = do
  y <- scope@model do
    x <- embedding@embed_tokens input_ids dim=D
    return linear@lm_head x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    graph = spec["model"]["graph"]

    def _collect_param_paths(
        items: list[dict[str, Any]], prefix: str = ""
    ) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for item in items:
            name, node_spec = next(iter(item.items()))
            path = f"{prefix}.{name}" if prefix else name
            if not isinstance(node_spec, dict):
                continue
            op = node_spec.get("_op")
            if isinstance(op, str):
                out.append((path, op))
            nested = node_spec.get("graph")
            if isinstance(nested, list):
                out.extend(_collect_param_paths(nested, path))
        return out

    ops = _collect_param_paths(graph)
    assert not any(path == "model.embed_tokens" and op == "embedding" for path, op in ops)
    embedding_nodes = [
        node_spec
        for item in graph
        for _, node_spec in item.items()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "embedding"
    ]
    assert len(embedding_nodes) == 1
    assert embedding_nodes[0]["_params"]["weight"] == "embed_tokens.weight"
    assert embedding_nodes[0]["_scope"] == "model"
    assert "_abs_path" not in embedding_nodes[0]
    linear_nodes = [
        node_spec
        for item in graph
        for _, node_spec in item.items()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "linear"
    ]
    assert len(linear_nodes) == 1
    assert linear_nodes[0]["_params"]["weight"] == "lm_head.weight"
    assert linear_nodes[0]["_scope"] == "model"
    assert "_abs_path" not in linear_nodes[0]


def test_path_bound_block_call_keeps_single_scope_source_of_truth() -> None:
    source = """
D = 2
lin2 :: @Path -> Tensor[B,T,D] -> Tensor[B,T,D]
lin2@path x = do
  y <- linear@path x dim=D bias=true
  z <- add y y
  return z
tiny :: Tensor[B,T,D] -> Tensor[B,T,D]
tiny x = do
  y <- scope@attn do
    return lin2@proj x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    graph = spec["model"]["graph"]
    call_nodes = [
        node_spec
        for item in graph
        for _, node_spec in item.items()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "call"
    ]
    assert len(call_nodes) == 1
    call_node = call_nodes[0]
    assert call_node.get("_scope") == "attn"
    assert "path" not in call_node

    target = call_node.get("_target")
    assert isinstance(target, str)
    assert target != "lin2"

    blocks = spec["model"].get("blocks")
    assert isinstance(blocks, dict)
    assert target in blocks
    block_graph = blocks[target]["graph"]
    linear_nodes = [
        node_spec
        for item in block_graph
        for _, node_spec in item.items()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "linear"
    ]
    assert len(linear_nodes) == 1
    linear_node = linear_nodes[0]
    assert linear_node["_params"]["weight"] == "proj.weight"
    assert linear_node["_params"]["bias"] == "proj.bias"
    assert "param_base" not in linear_node


def test_block_call_scope_is_relative_inside_loop_scopes() -> None:
    source = """
D = 2
L = 2
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = linear@proj x dim=D bias=true
main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- scope@model do
    for@layers i <- [0..L) do
      x <- blk x
    return x
  return y
"""
    spec = lower_axon_program_to_synapse_spec(parse_axon_program(source))
    graph = spec["model"]["graph"]
    for_nodes = [
        node_spec
        for item in graph
        for _, node_spec in item.items()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "for"
    ]
    assert len(for_nodes) == 1
    for_node = for_nodes[0]
    assert for_node["_scope"] == "model.layers"
    body = for_node.get("_body")
    assert isinstance(body, list)
    body_calls = [
        node_spec
        for item in body
        for _, node_spec in item.items()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "call"
    ]
    assert len(body_calls) == 1
    assert body_calls[0].get("_scope") is None


def test_scope_root_candidates_are_applied_to_param_paths() -> None:
    source = """
tiny :: TokenIds[B,T] -> Tensor[B,T,D]
tiny input_ids = do
  y <- scope@model root=["", "language_model"] do
    x <- embedding@embed_tokens input_ids dim=D
    return x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    embedding_node = next(node for node in node_specs if node.get("_op") == "embedding")
    assert embedding_node["_params"]["weight"] == "embed_tokens.weight"
    assert embedding_node["_scope"] == "model"
    assert embedding_node["_param_root"] == ["", "language_model"]


def test_scope_single_root_prefix_is_applied_to_param_paths() -> None:
    source = """
tiny :: TokenIds[B,T] -> Tensor[B,T,D]
tiny input_ids = do
  y <- scope@model root="language_model" do
    x <- embedding@embed_tokens input_ids dim=D
    return x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    embedding_node = next(node for node in node_specs if node.get("_op") == "embedding")
    assert embedding_node["_params"]["weight"] == "embed_tokens.weight"
    assert embedding_node["_scope"] == "model"
    assert embedding_node["_param_root"] == "language_model"


def test_scope_dynamic_root_expression_emits_param_root(tmp_path: Path) -> None:
    main_path = tmp_path / "main.axon"
    main_path.write_text(
        """
import Params
ROOT = (Params.has_root "language_model") ? "language_model" : ""
tiny :: TokenIds[B,T] -> Tensor[B,T,D]
tiny input_ids = do
  y <- scope@model root=ROOT do
    x <- embedding@embed_tokens input_ids dim=D
    return x
  return y
""".strip()
        + "\n",
        encoding="utf-8",
    )
    modules = parse_axon_program_from_path(main_path)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    embedding_node = next(node for node in node_specs if node.get("_op") == "embedding")
    assert embedding_node["_params"]["weight"] == "embed_tokens.weight"
    assert embedding_node["_scope"] == "model"
    assert "_param_root" in embedding_node


def test_double_at_path_is_absolute_inside_scope_bind() -> None:
    source = """
tiny :: TokenIds[B,T] -> Tensor[B,T,V]
tiny input_ids = do
  y <- scope@model do
    x <- embedding@embed_tokens input_ids dim=D
    return linear@@lm_head x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    graph = spec["model"]["graph"]

    def _collect_param_paths(
        items: list[dict[str, Any]], prefix: str = ""
    ) -> list[tuple[str, str]]:
        out: list[tuple[str, str]] = []
        for item in items:
            name, node_spec = next(iter(item.items()))
            path = f"{prefix}.{name}" if prefix else name
            if not isinstance(node_spec, dict):
                continue
            op = node_spec.get("_op")
            if isinstance(op, str):
                out.append((path, op))
            nested = node_spec.get("graph")
            if isinstance(nested, list):
                out.extend(_collect_param_paths(nested, path))
        return out

    ops = _collect_param_paths(graph)
    assert not any(path == "model.embed_tokens" and op == "embedding" for path, op in ops)
    embedding_nodes = [
        node_spec
        for item in graph
        for _, node_spec in item.items()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "embedding"
    ]
    assert len(embedding_nodes) == 1
    assert embedding_nodes[0]["_params"]["weight"] == "embed_tokens.weight"
    assert embedding_nodes[0]["_scope"] == "model"
    assert "_abs_path" not in embedding_nodes[0]
    linear_nodes = [
        node_spec
        for item in graph
        for _, node_spec in item.items()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "linear"
    ]
    assert len(linear_nodes) == 1
    assert linear_nodes[0]["_params"]["weight"] == "weight"
    assert linear_nodes[0]["_abs_path"] == "lm_head"
    assert "_scope" not in linear_nodes[0]


def test_mamba_scan_param_overrides_are_scoped_inside_scope_bind() -> None:
    source = """
tiny :: Tensor[B,S,D] -> Tensor[B,S,D]
tiny x = do
  y <- scope@mixer do
    y <- mamba_scan x x x x A=A_log D=D a_is_log=true
    return y
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    assert len(node_specs) == 1
    assert node_specs[0]["_op"] == "mamba_scan"
    assert node_specs[0]["A"] == "mixer.A_log"
    assert node_specs[0]["D"] == "mixer.D"


def test_embedding_at_path_uses_neutral_node_name_and_explicit_weight_param() -> None:
    source = """
gpt2 :: TokenIds[B,T] -> Tensor[B,T,D]
gpt2 input_ids = do
  tok <- embedding@wte input_ids dim=D
  return tok
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    graph = spec["model"]["graph"]
    assert isinstance(graph, list) and len(graph) == 1
    node_name, node_spec = next(iter(graph[0].items()))
    assert node_name.startswith("n_op_")
    assert node_spec["_op"] == "embedding"
    assert node_spec["_params"]["weight"] == "wte.weight"


def test_embedding_paths_remain_distinct_for_wte_and_wpe() -> None:
    source = """
gpt2 :: TokenIds[B,T] -> Tensor[B,T,D]
gpt2 input_ids = do
  tok <- embedding@wte input_ids dim=D
  pos <- embedding@wpe input_ids dim=D
  return tok
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    weights = [node["_params"]["weight"] for node in node_specs if node.get("_op") == "embedding"]
    assert weights == ["wte.weight", "wpe.weight"]


def test_embedding_and_linear_can_share_wte_weight_path() -> None:
    source = """
gpt2 :: TokenIds[B,T] -> Tensor[B,T,V]
gpt2 input_ids = do
  h <- embedding@wte input_ids dim=D
  logits <- linear@wte h
  return logits
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "embedding"
    assert node_specs[0]["_params"]["weight"] == "wte.weight"
    assert node_specs[1]["_op"] == "linear"


def test_parse_rejects_scope_statement_form() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  scope@attn do
    y <- linear@proj x dim=4
    return y
  return x
"""
    with pytest.raises(ValueError, match="invalid Axon source syntax"):
        parse_axon_module(source)


def test_parse_for_at_range_loop_sugar() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  for@model.layers i <- [0..3] do
    y <- add x x
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    model = spec["model"]
    repeat_node = next(
        node_spec
        for item in model["graph"]
        for node_spec in item.values()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "for"
    )
    assert repeat_node["_op"] == "for"
    assert repeat_node["_scope"] == "model.layers"
    assert repeat_node["_var"] == "i"
    assert repeat_node["_to"] == {"_expr": "binary", "op": "+", "left": 3, "right": 1}


def test_parse_for_at_range_loop_sugar_with_nonzero_start() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  for@model.layers i <- [1..4] do
    y <- add x x
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    model = spec["model"]
    repeat_node = next(
        node_spec
        for item in model["graph"]
        for node_spec in item.values()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "for"
    )
    assert repeat_node["_op"] == "for"
    assert repeat_node["_scope"] == "model.layers"
    assert repeat_node["_var"] == "i"
    assert repeat_node["_from"] == 1
    assert repeat_node["_to"] == {"_expr": "binary", "op": "+", "left": 4, "right": 1}


def test_parse_for_at_range_loop_sugar_half_open_with_paren() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  for@model.layers i <- [1..4) do
    y <- add x x
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    model = spec["model"]
    repeat_node = next(
        node_spec
        for item in model["graph"]
        for node_spec in item.values()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "for"
    )
    assert repeat_node["_op"] == "for"
    assert repeat_node["_scope"] == "model.layers"
    assert repeat_node["_var"] == "i"
    assert repeat_node["_from"] == 1
    assert repeat_node["_to"] == 4


def test_parse_for_at_range_loop_sugar_left_open_right_closed() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  for@model.layers i <- (0..4] do
    y <- add x x
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    model = spec["model"]
    repeat_node = next(
        node_spec
        for item in model["graph"]
        for node_spec in item.values()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "for"
    )
    assert repeat_node["_op"] == "for"
    assert repeat_node["_scope"] == "model.layers"
    assert repeat_node["_var"] == "i"
    assert repeat_node["_from"] == {"_expr": "binary", "op": "+", "left": 0, "right": 1}
    assert repeat_node["_to"] == {"_expr": "binary", "op": "+", "left": 4, "right": 1}


def test_parse_for_at_range_loop_sugar_with_step() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  for@model.layers i <- [1..8) step=2 do
    y <- add x x
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    model = spec["model"]
    repeat_node = next(
        node_spec
        for item in model["graph"]
        for node_spec in item.values()
        if isinstance(node_spec, dict) and node_spec.get("_op") == "for"
    )
    assert repeat_node["_op"] == "for"
    assert repeat_node["_scope"] == "model.layers"
    assert repeat_node["_var"] == "i"
    assert repeat_node["_from"] == 1
    assert repeat_node["_to"] == 8
    assert repeat_node["_step"] == 2


def test_parse_top_level_haskell_constants_across_modules() -> None:
    source = """
D = 768

id_block :: Tensor -> Tensor
id_block x = do
  return x

eps = 1e-05

main :: Tensor -> Tensor
main x = do
  y <- layernorm x dim=D eps=eps
  return y
"""
    modules = parse_axon_program(source)
    assert [m.name for m in modules] == ["id_block", "main"]
    spec = lower_axon_program_to_synapse_spec(modules)
    symbols = spec["model"].get("symbols")
    assert symbols == {"D": 768, "eps": 1e-05}
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "layernorm"
    assert node_specs[0]["dim"] == "D"
    assert node_specs[0]["eps"] == "eps"


def test_type_shape_annotations_expose_symbols_and_infer_layernorm_dim() -> None:
    source = """
gpt2_block :: Tensor[B,T,D] -> Tensor[B,T,D]
gpt2_block x = do
  y <- layernorm@ln_1 x eps=1e-05
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    symbols = spec["model"].get("symbols")
    assert symbols == {"B": None, "T": None, "D": None}
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "layernorm"
    assert node_specs[0]["_args"] == "x"
    assert node_specs[0]["dim"] == "D"


def test_layernorm_accepts_bias_false_and_null_literals() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  a <- layernorm@ln_a x eps=1e-05 bias=false
  b <- layernorm@ln_b a eps=1e-05 bias=null
  return b
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "layernorm"
    assert node_specs[0]["bias"] is False
    assert node_specs[1]["_op"] == "layernorm"
    assert node_specs[1]["bias"] is None


def test_type_shape_annotations_infer_rmsnorm_dim() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y <- rmsnorm@n x eps=1e-06
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "rmsnorm"
    assert node_specs[0]["_args"] == "x"
    assert node_specs[0]["dim"] == "D"


def test_infer_split_sizes_from_known_last_dim() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  qkv <- linear x dim=3*D
  q, k, v <- split qkv
  return q
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[1]["_op"] == "split"
    assert node_specs[1]["sizes"] == [
        {"_expr": "name", "id": "D"},
        {"_expr": "name", "id": "D"},
        {"_expr": "name", "id": "D"},
    ]


def test_split_rejects_parts_and_sizes_together() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  q, k <- split x parts=2 sizes=[4,4]
  return q
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match="split accepts either parts or sizes, not both"):
        lower_axon_program_to_synapse_spec(modules)


def test_split_rejects_output_arity_mismatch_for_parts() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  q, k <- split x parts=3
  return q
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            r"split parts=3 requires 3 outputs, got 2|"
            r"tuple bind arity mismatch: 2 target\(s\), 3 value\(s\)"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_split_requires_tuple_binding_outputs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  q <- split x parts=1
  return q
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match="split requires tuple/list binding outputs"):
        lower_axon_program_to_synapse_spec(modules)


def test_topk_requires_k_kwarg() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  vals, idx <- topk x
  return vals
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match=r"topk missing required kwargs: k"):
        lower_axon_program_to_synapse_spec(modules)


def test_topk_requires_two_outputs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  vals <- topk x k=8
  return vals
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "topk requires exactly two outputs: values, indices|"
            "cannot bind multi-value expression to a single target"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_moe_select_requires_expert_kwarg() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,EPT] -> Tensor[B,T,EPT] -> Tensor[B,T,D]
blk x scores idx = do
  x_sel, token_idx, topk_pos, sel_scores <- moe_select x scores idx
  return x_sel
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match=r"moe_select missing required kwargs: expert"):
        lower_axon_program_to_synapse_spec(modules)


def test_moe_select_requires_four_outputs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,EPT] -> Tensor[B,T,EPT] -> Tensor[B,T,D]
blk x scores idx = do
  x_sel <- moe_select x scores idx expert=0
  return x_sel
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "moe_select requires exactly four outputs: "
            "selected_hidden, token_idx, topk_pos, selected_scores|"
            "cannot bind multi-value expression to a single target"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_moe_select_rejects_positional_expert_compat() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,EPT] -> Tensor[B,T,EPT] -> Tensor[B,T,D]
blk x scores idx = do
  x_sel, token_idx, topk_pos, sel_scores <- moe_select x scores idx 0
  return x_sel
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match=r"moe_select expects 3 positional args, got 4"):
        lower_axon_program_to_synapse_spec(modules)


def test_moe_scatter_add_requires_single_output() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[N] -> Tensor[N,D] -> Tensor[N] -> Tensor[B,T,D]
blk m idx upd scores = do
  a, b <- moe_scatter_add m idx upd scores
  return a
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "moe_scatter_add requires a single scalar output binding|"
            "multi-target bind requires a tuple-valued expression"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_attention_requires_single_output() -> None:
    source = """
blk :: Tensor[B,H,T,D] -> Tensor[B,H,T,D] -> Tensor[B,H,T,D] -> Tensor[B,H,T,D]
blk q k v = do
  y, z <- attention q k v
  return y
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "attention requires a single scalar output binding|"
            "multi-target bind requires a tuple-valued expression"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_attention_rejects_unknown_kwarg() -> None:
    source = """
blk :: Tensor[B,H,T,D] -> Tensor[B,H,T,D] -> Tensor[B,H,T,D] -> Tensor[B,H,T,D]
blk q k v = do
  y <- attention q k v foo=1
  return y
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            r"attention unsupported kwargs: foo; allowed: causal, eager, "
            r"float_mask_additive, float_mask_floor_keep, mask, padding_mask, "
            r"scale, sink, sink_path"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_cache_update_requires_three_outputs() -> None:
    source = """
blk :: ?Cache -> Tensor[B,H,T,D] -> Tensor[B,H,T,D] -> Tensor[B,H,T,D]
blk past k v = do
  k_ctx, v_ctx <- _cache_update past k v
  return k_ctx
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "cache_update requires exactly three outputs: k_ctx, v_ctx, present|"
            r"tuple bind arity mismatch: 2 target\(s\), 3 value\(s\)"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_cache_update_rejects_unknown_kwarg() -> None:
    source = """
blk :: ?Cache -> Tensor[B,H,T,D] -> Tensor[B,H,T,D] -> Tensor[B,H,T,D]
blk past k v = do
  k_ctx, v_ctx, present <- _cache_update past k v foo=1
  return k_ctx
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match=r"cache_update unsupported kwargs: foo"):
        lower_axon_program_to_synapse_spec(modules)


def test_list_append_requires_single_output() -> None:
    source = """
blk :: ?Cache -> ?CacheLayer -> ?Cache
blk cache present = do
  a, b <- _list_append cache present
  return cache
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "list_append requires a single scalar output binding|"
            "multi-target bind requires a tuple-valued expression"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_list_append_rejects_unknown_kwarg() -> None:
    source = """
blk :: ?Cache -> ?Cache -> ?Cache
blk cache present = do
  out <- _list_append cache present foo=1
  return out
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match=r"list_append unsupported kwargs: foo"):
        lower_axon_program_to_synapse_spec(modules)


def test_cache_seq_len_requires_single_output() -> None:
    source = """
blk :: ?Cache -> Int
blk kv = do
  a, b <- _cache_seq_len kv
  return a
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "cache_seq_len requires a single scalar output binding|"
            "multi-target bind requires a tuple-valued expression"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_causal_mask_requires_single_output() -> None:
    source = """
blk :: Tensor[B,H,T,D] -> Tensor[B,H,T,D] -> Tensor[B,H,T,T]
blk q k = do
  m, n <- causal_mask q k
  return m
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "causal_mask requires a single scalar output binding|"
            "multi-target bind requires a tuple-valued expression"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_linear_position_bias_requires_single_output() -> None:
    source = """
blk :: Tensor[B,T] -> Tensor[B,H,1,T]
blk attention_mask = do
  a, b <- linear_position_bias attention_mask heads=H
  return a
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "linear_position_bias requires a single scalar output binding|"
            "multi-target bind requires a tuple-valued expression"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_list_index_requires_single_output() -> None:
    source = """
blk :: ?Cache -> ?Cache
blk cache = do
  a, b <- _list_index cache 0
  return a
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "list_index requires a single scalar output binding|"
            "multi-target bind requires a tuple-valued expression"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_list_init_requires_single_output() -> None:
    source = """
blk :: ?Cache
blk = do
  a, b <- _list_init
  return a
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "list_init requires a single scalar output binding|"
            "multi-target bind requires a tuple-valued expression"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_position_ids_requires_single_output() -> None:
    source = """
blk :: Tensor[B,T] -> ?Tensor[B,T] -> Tensor[B,T]
blk input_ids attn_mask = do
  a, b <- position_ids input_ids attn_mask
  return a
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "position_ids requires a single scalar output binding|"
            "multi-target bind requires a tuple-valued expression"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_topk_accepts_largest_and_sorted_kwargs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  vals, idx <- topk x k=8 dim=-1 largest=false sorted=false
  return vals
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "topk"
    assert node_specs[0]["largest"] is False
    assert node_specs[0]["sorted"] is False


def test_softmax_rejects_tuple_outputs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y, z <- softmax x dim=-1
  return y
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "softmax requires a single scalar output binding|"
            "multi-target bind requires a tuple-valued expression"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_softmax_rejects_unsupported_dtype() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y <- softmax x dim=-1 dtype=float64
  return y
"""
    modules = parse_axon_program(source)
    with pytest.raises(ValueError, match=r"Unsupported softmax dtype: float64"):
        lower_axon_program_to_synapse_spec(modules)


def test_softmax_accepts_supported_dtype_and_default_dim() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y <- softmax x dtype=bfloat16
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "softmax"
    assert node_specs[0]["dtype"] == "bfloat16"
    assert "dim" not in node_specs[0]


def test_zeros_like_rejects_tuple_outputs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y, z <- zeros_like x
  return y
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "zeros_like requires a single scalar output binding|"
            "multi-target bind requires a tuple-valued expression"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_add_rejects_tuple_outputs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D] -> Tensor[B,T,D]
blk x y = do
  a, b <- add x y
  return a
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "add requires a single scalar output binding|"
            "multi-target bind requires a tuple-valued expression"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_mul_rejects_tuple_outputs() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D] -> Tensor[B,T,D]
blk x y = do
  a, b <- mul x y
  return a
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError,
        match=(
            "mul requires a single scalar output binding|"
            "multi-target bind requires a tuple-valued expression"
        ),
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_infer_linear_dim_from_return_shape() -> None:
    source = """
gpt2 :: TokenIds[B,T] -> Tensor[B,T,V]
gpt2 input_ids = do
  h <- embedding@wte input_ids dim=D
  logits <- linear@wte h
  return logits
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[1]["_op"] == "linear"
    assert node_specs[1]["_args"] == "h"
    assert node_specs[1]["_bind"] == "logits"
    assert node_specs[1]["dim"] == "V"


def test_infer_embedding_dim_from_typed_output_shape() -> None:
    source = """
emb :: TokenIds[B,T] -> Tensor[B,T,D]
emb ids = do
  x <- embedding ids
  return x
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "embedding"
    assert node_specs[0]["dim"] == "D"


def test_add_unifies_symbolic_last_dim_for_following_ops() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor -> Tensor[B,T,D]
blk tok pos = do
  x <- tok + pos
  y <- layernorm pos
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[1]["_op"] == "layernorm"
    assert node_specs[1]["dim"] == "D"


def test_embedding_accepts_dim_kwarg() -> None:
    source = """
emb :: TokenIds[B,T] -> Tensor[B,T,D]
emb ids = do
  x <- embedding ids dim=D
  return x
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "embedding"
    assert node_specs[0]["dim"] == "D"


def test_embedding_rejects_embedding_dim_kwarg() -> None:
    source = """
emb :: TokenIds[B,T] -> Tensor[B,T,D]
emb ids = do
  x <- embedding ids embedding_dim=D
  return x
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError, match="embedding unsupported kwargs: embedding_dim; allowed: dim, scale"
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_embedding_rejects_num_embeddings_kwarg() -> None:
    source = """
emb :: TokenIds[B,T] -> Tensor[B,T,D]
emb ids = do
  x <- embedding ids dim=D num_embeddings=V
  return x
"""
    modules = parse_axon_program(source)
    with pytest.raises(
        ValueError, match="embedding unsupported kwargs: num_embeddings; allowed: dim, scale"
    ):
        lower_axon_program_to_synapse_spec(modules)


def test_path_parameterized_block_call_binds_param_base() -> None:
    source = """
lin_bt :: @Path -> Tensor -> Int -> Tensor
lin_bt@path x d = do
  return linear@path x dim=d bias=true transpose=true

blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y <- lin_bt@attn.c_proj x D
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    model_nodes = _node_specs(spec["model"]["graph"])
    assert model_nodes[0]["_op"] == "linear"
    assert "param_base" not in model_nodes[0]
    assert model_nodes[0]["_params"]["weight"] == "attn.c_proj.weight"
    assert model_nodes[0]["_params"]["bias"] == "attn.c_proj.bias"


def test_path_parameter_annotation_rejects_non_path_type() -> None:
    source = """
lin_bt :: @ParamPath -> Tensor -> Int -> Tensor
lin_bt@path x d = do
  return linear@path x dim=d bias=true transpose=true

blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y <- lin_bt@attn.c_proj x D
  return y
"""
    with pytest.raises(ValueError, match=r"path signature type must be Path"):
        parse_axon_program(source)


def test_linear_accepts_transpose_flag() -> None:
    source = """
blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y <- linear x dim=D transpose=true bias=false
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "linear"
    assert node_specs[0]["transpose"] is True


def test_block_signature_propagates_output_last_dim_from_tensor_shape() -> None:
    source = """
rms :: @Path -> Tensor[B,T,D] -> Tensor[B,T,D]
rms@path x = rmsnorm@path x

blk :: Tensor[B,T,D] -> Tensor[B,T,D]
blk x = do
  y <- rms@norm x
  z <- layernorm y
  return z
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[1]["_op"] == "layernorm"
    assert node_specs[1]["dim"] == "D"


def test_block_signature_propagates_output_last_dim_from_scalar_param() -> None:
    source = """
lin :: @Path -> Tensor[B,T,Din] -> Int -> Tensor[B,T,dim]
lin@path x dim = linear@path x dim=dim bias=true transpose=true

blk :: Tensor[B,T,D] -> Tensor[B,T,16]
blk x = do
  y <- lin@proj x 16
  z <- layernorm y
  return z
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[1]["_op"] == "layernorm"
    assert node_specs[1]["dim"] == 16


def test_infer_repeat_repeats_from_typed_shapes() -> None:
    source = """
rk :: Tensor[B,Kh,T,Hd] -> Tensor[B,H,T,Hd]
rk k = do
  k_ctx <- repeat k
  return k_ctx
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "repeat"
    assert node_specs[0]["repeats"] == "(H // Kh)"


def test_parse_axon_ignores_haskell_style_comments() -> None:
    source = """
-- leading comment
tiny :: Tensor -> ?Tensor -> Tensor -- signature comment
tiny x cache = do -- def comment
  -- statement comment
  y <- x |> linear@proj dim=4 bias=false -- inline comment
  return y -- trailing comment
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    assert len(node_specs) == 1
    assert node_specs[0]["_op"] == "linear"
    assert node_specs[0]["_args"] == "x"
    assert node_specs[0]["_bind"] == "y"


def test_parse_and_lower_pipeline_with_trailing_operator_continuations() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  qkv <- x |>
    layernorm@ln_1 dim=768 eps=1e-05 |>
    linear@attn.c_attn dim=2304 transpose=true bias=true
  return qkv
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    assert len(node_specs) == 2
    assert node_specs[0]["_op"] == "layernorm"
    assert node_specs[0]["_bind"] == "pipe_1"
    assert node_specs[1]["_op"] == "linear"
    assert node_specs[1]["_args"] == "pipe_1"
    assert node_specs[1]["_bind"] == "qkv"
    assert node_specs[1]["_params"]["weight"] == "attn.c_attn.weight"
    assert node_specs[1]["_params"]["bias"] == "attn.c_attn.bias"


def test_lower_pipeline_axon_to_synapse_spec() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  y <- x |> linear@proj |> _activations_gelu_new
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)

    model = spec["model"]
    assert model["inputs"] == {"x": {"optional": False}}
    assert model["outputs"] == {"y": "y"}

    node_specs = _node_specs(model["graph"])
    assert node_specs[0] == {
        "_op": "linear",
        "_args": "x",
        "_bind": "pipe_1",
        "_params": {"weight": "proj.weight"},
    }
    assert node_specs[1] == {"_op": "activations_gelu_new", "_args": "pipe_1", "_bind": "y"}


def test_lower_axon_module_to_synapse_spec_persists_block_io_types() -> None:
    source = """
tiny :: Tensor[B,S,D] -> Tensor[B,S,D]
tiny x = do
  return x
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    block_io = spec["model"]["types"]["block_io"]
    main_output_name = next(iter(spec["model"]["outputs"]))
    assert block_io["main"]["inputs"]["x"] == "Tensor[B,S,D]"
    assert block_io["main"]["outputs"][main_output_name] == "Tensor[B,S,D]"


def test_lower_axon_program_to_synapse_spec_persists_block_io_types_for_blocks() -> None:
    source = """
blk :: Tensor[B,S,D] -> Tensor[B,S,D]
blk x = do
  return x

main :: Tensor[B,S,D] -> Tensor[B,S,D]
main x = do
  y <- blk x
  return y
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    block_io = spec["model"]["types"]["block_io"]
    main_output_name = next(iter(spec["model"]["outputs"]))
    blk_output_name = next(iter(spec["model"]["blocks"]["blk"]["outputs"]))
    assert block_io["main"]["inputs"]["x"] == "Tensor[B,S,D]"
    assert block_io["main"]["outputs"][main_output_name] == "Tensor[B,S,D]"
    assert block_io["blk"]["inputs"]["x"] == "Tensor[B,S,D]"
    assert block_io["blk"]["outputs"][blk_output_name] == "Tensor[B,S,D]"


def test_lower_return_pipeline_expression_to_named_output() -> None:
    source = """
tiny :: Tensor -> Tensor -> Tensor
tiny x wte = do
  return layernorm@ln_f x dim=768 eps=1e-05 |> linear@wte dim=50257
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    assert len(node_specs) == 2
    assert node_specs[0] == {
        "_op": "layernorm",
        "_args": "x",
        "_bind": "pipe_1",
        "dim": 768,
        "eps": "1e-05",
        "_params": {"weight": "ln_f.weight", "bias": "ln_f.bias"},
    }
    assert node_specs[1] == {
        "_op": "linear",
        "_args": "pipe_1",
        "_bind": "out_0",
        "dim": 50257,
        "_params": {"weight": "wte.weight"},
    }
    assert spec["model"]["outputs"] == {"out_0": "out_0"}


def test_lower_bind_operator_to_synapse_spec() -> None:
    source = """
tiny :: Tensor -> Tensor
tiny x = do
  y <- linear@p1 x >>= \\z -> _activations_gelu_new z
  return y
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0] == {
        "_op": "linear",
        "_args": "x",
        "_bind": "bind_1",
        "_params": {"weight": "p1.weight"},
    }
    assert node_specs[1] == {"_op": "activations_gelu_new", "_args": "bind_1", "_bind": "y"}


def test_nested_call_expression_in_kwarg_lowers_via_temp_binding() -> None:
    source = """
tiny :: TokenIds[B,S] -> ?Tensor[B,S] -> ?Cache -> Tensor[B,S]
tiny input_ids attn_mask past_key_values = do
  pos <- position_ids input_ids attn_mask past_length=(_cache_seq_len (_list_index past_key_values 0))
  return pos
"""
    modules = parse_axon_program(source)
    spec = lower_axon_program_to_synapse_spec(modules)
    node_specs = _node_specs(spec["model"]["graph"])
    assert node_specs[0]["_op"] == "list_index"
    assert node_specs[1]["_op"] == "cache_seq_len"
    assert node_specs[2]["_op"] == "position_ids"
    assert node_specs[2]["past_length"] == node_specs[1]["_bind"]


def test_reshape_heads_triplet_is_rejected_as_obsolete_compat() -> None:
    source = """
tiny :: Tensor -> Tensor -> Tensor -> Tensor -> Tensor
tiny q k v bias = do
  ctx_heads <- reshape_heads_triplet q k v heads=12 head_dim=64 |>
    attention mask=bias
  return ctx_heads
"""
    module = parse_axon_module(source)
    with pytest.raises(ValueError, match="obsolete compatibility call 'reshape_heads_triplet'"):
        lower_axon_module_to_synapse_spec(module)


def test_lower_ternary_to_lazy_select_op() -> None:
    source = """
tiny :: Tensor -> ?Tensor -> (Tensor, Tensor)
tiny x use_cache = do
  k, v <- use_cache ? _cache_update past k0 v0 : k0, v0
  return k, v
"""
    module = parse_axon_module(source)
    spec = lower_axon_module_to_synapse_spec(module)
    node_specs = _node_specs(spec["model"]["graph"])
    assert len(node_specs) == 1
    select_spec = node_specs[0]
    assert select_spec["_op"] == "select"
    assert select_spec["cond"] == "use_cache"
    assert select_spec["_bind"] == ["k", "v"]
    assert isinstance(select_spec["_then"], list)
    assert isinstance(select_spec["_else"], list)


def test_lower_if_then_else_matches_ternary_lowering() -> None:
    ternary_source = """
tiny :: Tensor -> ?Tensor -> (Tensor, Tensor)
tiny x use_cache = do
  k, v <- use_cache ? _cache_update past k0 v0 : k0, v0
  return k, v
"""
    if_source = """
tiny :: Tensor -> ?Tensor -> (Tensor, Tensor)
tiny x use_cache = do
  k, v <- if use_cache then _cache_update past k0 v0 else k0, v0
  return k, v
"""

    ternary_spec = lower_axon_module_to_synapse_spec(parse_axon_module(ternary_source))
    if_spec = lower_axon_module_to_synapse_spec(parse_axon_module(if_source))

    assert _node_specs(if_spec["model"]["graph"]) == _node_specs(ternary_spec["model"]["graph"])


def test_synapse_to_axon_roundtrip_equivalence_for_subset() -> None:
    spec: dict[str, Any] = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {"optional": False}},
            "graph": [
                {
                    "n1": {
                        "_op": "linear",
                        "_args": "x",
                        "_bind": "h",
                        "_params": {"weight": "proj.weight", "bias": "proj.bias"},
                    }
                },
                {
                    "n2": {
                        "_op": "activations_gelu_new",
                        "_args": "h",
                        "_bind": "y",
                    }
                },
            ],
            "outputs": {"y": "y"},
        },
    }

    axon = synapse_spec_to_axon_module_text(spec, module_name="tiny")
    reparsed = parse_axon_module(axon)
    spec2 = lower_axon_module_to_synapse_spec(reparsed)

    assert spec2["model"]["inputs"] == spec["model"]["inputs"]
    assert spec2["model"]["outputs"] == spec["model"]["outputs"]
    assert _node_specs(spec2["model"]["graph"])[0]["_op"] == "linear"
    assert _node_specs(spec2["model"]["graph"])[0]["_args"] == "x"
    assert _node_specs(spec2["model"]["graph"])[0]["_bind"] == "h"
    assert _node_specs(spec2["model"]["graph"])[1] == {
        "_op": "activations_gelu_new",
        "_args": "h",
        "_bind": "y",
    }


def test_synapse_to_axon_roundtrip_with_meta_and_control_nodes() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "name": "Tiny",
            "symbols": {"L": 2},
            "inputs": {"x": {"optional": False}},
            "graph": [
                {
                    "n1": {
                        "_op": "for",
                        "_scope": "n1",
                        "_var": "i",
                        "_to": "L",
                        "_body": [{"a": {"_op": "add", "_args": ["x", "x"], "_bind": "x"}}],
                    }
                },
                {"n2": {"_op": "call", "_target": "block", "_args": "x", "_bind": "x"}},
                {
                    "n3": {
                        "_op": "layernorm",
                        "_args": "x",
                        "_bind": "y",
                        "dim": 4,
                        "eps": 1e-5,
                    }
                },
            ],
            "outputs": {"logits": "y"},
            "blocks": {"block": {"inputs": {"x": {}}, "graph": [], "outputs": {"y": "x"}}},
        },
    }

    axon = synapse_spec_to_axon_module_text(spec, module_name="tiny")
    reparsed = parse_axon_program(axon)
    spec2 = lower_axon_program_to_synapse_spec(reparsed)
    assert spec2["synapse"] == 1
    assert "block" in spec2["model"]["blocks"]


def test_synapse_to_axon_readable_omits_meta_lines() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "name": "Tiny",
            "symbols": {"D": 4},
            "inputs": {"x": {"optional": False}},
            "graph": [{"n1": {"_op": "activations_gelu_new", "_args": "x", "_bind": "y"}}],
            "outputs": {"y": "y"},
        },
    }
    axon = synapse_spec_to_axon_module_text(spec, module_name="tiny")
    assert "meta " not in axon
    assert "y <- _activations_gelu_new x" in axon


def test_synapse_to_axon_readable_blocks_lower_back_via_program() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "symbols": {"L": 2},
            "blocks": {
                "blk": {
                    "inputs": {"x": {"optional": False}},
                    "graph": [{"n": {"_op": "activations_gelu_new", "_args": "x", "_bind": "y"}}],
                    "outputs": {"y": "y"},
                }
            },
            "inputs": {"x": {"optional": False}},
            "graph": [
                {
                    "loop": {
                        "_op": "for",
                        "_scope": "loop",
                        "_var": "i",
                        "_to": 2,
                        "_body": [
                            {"u": {"_op": "call", "_target": "blk", "_args": "x", "_bind": "x"}}
                        ],
                    }
                }
            ],
            "outputs": {"y": "x"},
        },
    }
    axon = synapse_spec_to_axon_module_text(spec, module_name="main")
    modules = parse_axon_program(axon)
    spec2 = lower_axon_program_to_synapse_spec(modules)
    assert spec2["model"]["outputs"] == {"x": "x"}
    assert "blocks" in spec2["model"]
    assert "blk" in spec2["model"]["blocks"]
    assert "for@loop i <- [0..2) do" in axon


def test_parse_axon_padding_side_pragma_is_preserved() -> None:
    source = """
{-# PADDING_SIDE "right" #-}
tiny :: Tensor[B,T,D] -> Tensor[B,T,D]
tiny x = do
  return x
"""
    module = parse_axon_module(source)
    assert module.pragmas == {"padding_side": "right"}
    spec = lower_axon_module_to_synapse_spec(module)
    assert spec["model"]["meta"] == {"padding_side": "right"}


def test_parse_program_top_level_padding_side_pragma_applies_to_main_module() -> None:
    source = """
{-# PADDING_SIDE "left" #-}
helper :: Tensor[B,T,D] -> Tensor[B,T,D]
helper x = do
  return x

main :: Tensor[B,T,D] -> Tensor[B,T,D]
main x = do
  y <- helper x
  return y
"""
    modules = parse_axon_program(source)
    assert [module.pragmas for module in modules] == [
        {"padding_side": "left"},
        {"padding_side": "left"},
    ]
    spec = lower_axon_program_to_synapse_spec(modules, main_module="main")
    assert spec["model"]["meta"] == {"padding_side": "left"}


def test_parse_rejects_conflicting_padding_side_pragmas() -> None:
    source = """
{-# PADDING_SIDE "left" #-}
{-# PADDING_SIDE "right" #-}
tiny :: Tensor[B,T,D] -> Tensor[B,T,D]
tiny x = do
  return x
"""
    with pytest.raises(ValueError, match="conflicting PADDING_SIDE pragmas"):
        parse_axon_module(source)
