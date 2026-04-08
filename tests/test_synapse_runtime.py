from __future__ import annotations

from pathlib import Path

import pytest
import torch

from brainsurgery.engine.state_dicts import _InMemoryStateDict
from brainsurgery.synapse import lower_axon_program_to_synapse_spec, parse_axon_program
from brainsurgery.synapse.runtime import SynapseProgramModel


def _tiny_linear_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "symbols": {"B": None, "T": None, "V": 8, "D": 4},
            "inputs": {"input_ids": {"shape": ["B", "T"], "dtype": "int64"}},
            "graph": [
                {
                    "embed_tokens": {
                        "_op": "embedding",
                        "_args": "input_ids",
                        "_bind": "x",
                        "dim": "D",
                    }
                },
                {
                    "lm_head": {
                        "_op": "linear",
                        "_args": "x",
                        "_bind": "logits",
                        "dim": "V",
                        "bias": False,
                        "weight": "embed_tokens.weight",
                    }
                },
            ],
            "outputs": {"logits": "logits"},
        },
    }


def _reshape_three_heads_spec(
    *, heads: int | None = None, head_dim: int | None = None
) -> dict[str, object]:
    q_node: dict[str, object] = {"_op": "reshape_heads", "_args": "q", "_bind": "qh"}
    k_node: dict[str, object] = {"_op": "reshape_heads", "_args": "k", "_bind": "kh"}
    v_node: dict[str, object] = {"_op": "reshape_heads", "_args": "v", "_bind": "vh"}
    if heads is not None:
        q_node["heads"] = heads
        k_node["heads"] = heads
        v_node["heads"] = heads
    if head_dim is not None:
        q_node["head_dim"] = head_dim
        k_node["head_dim"] = head_dim
        v_node["head_dim"] = head_dim
    return {
        "synapse": 1,
        "model": {
            "inputs": {"q": {}, "k": {}, "v": {}},
            "graph": [{"q": q_node}, {"k": k_node}, {"v": v_node}],
            "outputs": {"qh": "qh", "kh": "kh", "vh": "vh"},
        },
    }


def _reshape_heads_spec(
    *, heads: int | None = None, head_dim: int | None = None
) -> dict[str, object]:
    node: dict[str, object] = {
        "_op": "reshape_heads",
        "_args": "x",
        "_bind": "xh",
    }
    if heads is not None:
        node["heads"] = heads
    if head_dim is not None:
        node["head_dim"] = head_dim
    return {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [{"r": node}],
            "outputs": {"xh": "xh"},
        },
    }


def _split_qkv_grouped_spec(*, heads: int, kv_heads: int) -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [
                {
                    "split": {
                        "_op": "split_qkv_grouped",
                        "_args": "x",
                        "_bind": ["q", "k", "v"],
                        "heads": heads,
                        "kv_heads": kv_heads,
                    }
                }
            ],
            "outputs": {"q": "q", "k": "k", "v": "v"},
        },
    }


def _causal_mask_with_padding_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"q": {}, "k": {}, "padding_mask": {}},
            "graph": [
                {
                    "m": {
                        "_op": "causal_mask",
                        "_args": ["q", "k"],
                        "padding_mask": "padding_mask",
                        "window": 8,
                        "_bind": "mask",
                    }
                }
            ],
            "outputs": {"mask": "mask"},
        },
    }


def _causal_mask_early_exit_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"q": {}, "k": {}, "padding_mask": {"optional": True}},
            "graph": [
                {
                    "m": {
                        "_op": "causal_mask",
                        "_args": ["q", "k"],
                        "padding_mask": "padding_mask",
                        "early_exit": True,
                        "_bind": "mask",
                    }
                }
            ],
            "outputs": {"mask": "mask"},
        },
    }


def _blocksparse_mask_spec(
    *, block_size: int = 2, local_blocks: int = 1, vert_stride: int = 2, homo_head: bool = False
) -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"q": {}, "k": {}, "padding_mask": {"optional": True}},
            "graph": [
                {
                    "m": {
                        "_op": "blocksparse_mask",
                        "_args": ["q", "k"],
                        "padding_mask": "padding_mask",
                        "block_size": block_size,
                        "local_blocks": local_blocks,
                        "vert_stride": vert_stride,
                        "homo_head": homo_head,
                        "_bind": "mask",
                    }
                }
            ],
            "outputs": {"mask": "mask"},
        },
    }


def _arange_positions_with_mask_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"input_ids": {}, "attention_mask": {"optional": True}},
            "graph": [
                {
                    "p": {
                        "_op": "position_ids",
                        "_args": ["input_ids", "attention_mask"],
                        "_bind": "pos",
                    }
                }
            ],
            "outputs": {"pos": "pos"},
        },
    }


def _arange_positions_ignore_mask_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"input_ids": {}, "attention_mask": {"optional": True}},
            "graph": [
                {
                    "p": {
                        "_op": "position_ids",
                        "_args": ["input_ids", "attention_mask"],
                        "past_length": 3,
                        "use_attention_mask": False,
                        "_bind": "pos",
                    }
                }
            ],
            "outputs": {"pos": "pos"},
        },
    }


def _rope_pair_spec(*, inv_freq_dtype: str | None = None) -> dict[str, object]:
    node: dict[str, object] = {
        "_op": "rope_pair",
        "_args": ["q", "k"],
        "_bind": ["q_rot", "k_rot"],
        "position_ids": "position_ids",
        "theta": 1_000_000.0,
    }
    if inv_freq_dtype is not None:
        node["inv_freq_dtype"] = inv_freq_dtype
    return {
        "synapse": 1,
        "model": {
            "inputs": {"q": {}, "k": {}, "position_ids": {}},
            "graph": [{"rope": node}],
            "outputs": {"q_rot": "q_rot", "k_rot": "k_rot"},
        },
    }


def _rope_pair_partial_spec(*, partial_rotary_factor: float) -> dict[str, object]:
    node: dict[str, object] = {
        "_op": "rope_pair",
        "_args": ["q", "k"],
        "_bind": ["q_rot", "k_rot"],
        "position_ids": "position_ids",
        "theta": 1_000_000.0,
        "partial_rotary_factor": partial_rotary_factor,
    }
    return {
        "synapse": 1,
        "model": {
            "inputs": {"q": {}, "k": {}, "position_ids": {}},
            "graph": [{"rope": node}],
            "outputs": {"q_rot": "q_rot", "k_rot": "k_rot"},
        },
    }


def _rope_pair_proportional_spec(
    *, partial_rotary_factor: float, scale_factor: float = 1.0
) -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"q": {}, "k": {}, "position_ids": {}},
            "graph": [
                {
                    "rope": {
                        "_op": "rope_pair",
                        "_args": ["q", "k"],
                        "_bind": ["q_rot", "k_rot"],
                        "position_ids": "position_ids",
                        "theta": 1_000_000.0,
                        "rope_mode": "proportional",
                        "partial_rotary_factor": partial_rotary_factor,
                        "scale_factor": scale_factor,
                    }
                }
            ],
            "outputs": {"q_rot": "q_rot", "k_rot": "k_rot"},
        },
    }


def _rmsnorm_no_scale_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [
                {
                    "norm": {
                        "_op": "rmsnorm",
                        "_args": "x",
                        "_bind": "y",
                        "with_scale": False,
                        "eps": 1.0e-6,
                    }
                }
            ],
            "outputs": {"y": "y"},
        },
    }


def _select_spec(*, else_uses_missing_ref: bool = False) -> dict[str, object]:
    else_graph: list[dict[str, object]]
    if else_uses_missing_ref:
        else_graph = [{"else_ref": {"_op": "_ir_alias", "_args": "missing", "_bind": "else_v"}}]
    else:
        else_graph = [{"else_const": {"_op": "_ir_expr", "value": 2, "_bind": "else_v"}}]
    return {
        "synapse": 1,
        "model": {
            "inputs": {"flag": {}},
            "graph": [
                {
                    "pick": {
                        "_op": "select",
                        "cond": "flag",
                        "_bind": "y",
                        "_then_bind": "then_v",
                        "_else_bind": "else_v",
                        "_then": [
                            {"then_const": {"_op": "_ir_expr", "value": 1, "_bind": "then_v"}}
                        ],
                        "_else": else_graph,
                    }
                }
            ],
            "outputs": {"y": "y"},
        },
    }


def _moe_select_spec(*, expert: object = 1) -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}, "scores": {}, "idx": {}},
            "graph": [
                {
                    "sel": {
                        "_op": "moe_select",
                        "_args": ["x", "scores", "idx"],
                        "_bind": ["x_sel", "token_idx", "topk_pos", "sel_scores"],
                        "expert": expert,
                    }
                }
            ],
            "outputs": {
                "x_sel": "x_sel",
                "token_idx": "token_idx",
                "topk_pos": "topk_pos",
                "sel_scores": "sel_scores",
            },
        },
    }


def test_runtime_select_returns_then_or_else_value() -> None:
    model = SynapseProgramModel.from_spec(_select_spec())
    out_true = model.forward(flag=True)
    out_false = model.forward(flag=False)
    assert isinstance(out_true, dict)
    assert isinstance(out_false, dict)
    assert out_true["y"] == 1
    assert out_false["y"] == 2


def test_runtime_select_is_lazy_and_skips_non_selected_branch() -> None:
    model = SynapseProgramModel.from_spec(_select_spec(else_uses_missing_ref=True))
    out = model.forward(flag=True)
    assert isinstance(out, dict)
    assert out["y"] == 1


def test_runtime_rope_pair_proportional_matches_manual_reference() -> None:
    model = SynapseProgramModel.from_spec(
        _rope_pair_proportional_spec(partial_rotary_factor=0.5, scale_factor=4.0)
    )
    q = torch.randn(1, 2, 3, 16, dtype=torch.float32)
    k = torch.randn(1, 2, 3, 16, dtype=torch.float32)
    position_ids = torch.tensor([[0, 3, 7]], dtype=torch.int64)

    out = model.forward(q=q, k=k, position_ids=position_ids)
    q_rot = out["q_rot"]
    k_rot = out["k_rot"]

    head_dim = int(q.shape[-1])
    rotary_dim = int(head_dim * 0.5)
    rope_angles = rotary_dim // 2
    half = head_dim // 2
    theta = 1_000_000.0
    inv_freq_rotated = 1.0 / (
        theta ** (torch.arange(0, 2 * rope_angles, 2, dtype=torch.float32) / float(head_dim))
    )
    noop_angles = half - rope_angles
    inv_freq = (
        torch.cat([inv_freq_rotated, torch.zeros(noop_angles, dtype=torch.float32)], dim=0)
        if noop_angles > 0
        else inv_freq_rotated
    )
    inv_freq = inv_freq / 4.0
    ang = position_ids.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
    emb = torch.cat((ang, ang), dim=-1)
    cos = torch.cos(emb).unsqueeze(1)
    sin = torch.sin(emb).unsqueeze(1)

    q_half = torch.cat((-q[..., half:], q[..., :half]), dim=-1)
    k_half = torch.cat((-k[..., half:], k[..., :half]), dim=-1)
    q_expected = q * cos + q_half * sin
    k_expected = k * cos + k_half * sin

    assert torch.allclose(q_rot, q_expected, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(k_rot, k_expected, atol=1.0e-6, rtol=1.0e-6)


def test_runtime_rmsnorm_without_scale_matches_reference() -> None:
    model = SynapseProgramModel.from_spec(_rmsnorm_no_scale_spec())
    x = torch.randn(2, 3, 5, dtype=torch.float32)

    out = model.forward(x=x)
    y = out["y"]
    expected = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + 1.0e-6)

    assert torch.allclose(y, expected, atol=1.0e-6, rtol=1.0e-6)


def _moe_scatter_add_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"m": {}, "token_idx": {}, "upd": {}, "scores": {}},
            "graph": [
                {
                    "scatter": {
                        "_op": "moe_scatter_add",
                        "_args": ["m", "token_idx", "upd", "scores"],
                        "_bind": "m_out",
                    }
                }
            ],
            "outputs": {"m_out": "m_out"},
        },
    }


def _moe_grouped_ffn_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}, "scores": {}, "idx": {}},
            "graph": [
                {
                    "ffn": {
                        "_op": "moe_grouped_ffn",
                        "_args": ["x", "scores", "idx"],
                        "_bind": "m_out",
                        "gate_up_weight": "mlp.experts.gate_up_proj.weight",
                        "gate_up_bias": "mlp.experts.gate_up_proj.bias",
                        "down_weight": "mlp.experts.down_proj.weight",
                        "down_bias": "mlp.experts.down_proj.bias",
                        "alpha": 1.702,
                        "limit": 7.0,
                    }
                }
            ],
            "outputs": {"m_out": "m_out"},
        },
    }


def _attention_with_sink_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"q": {}, "k": {}, "v": {}, "sink": {}},
            "graph": [
                {
                    "attn": {
                        "_op": "attention",
                        "_args": ["q", "k", "v"],
                        "_bind": "out",
                        "sink": "sink",
                    }
                }
            ],
            "outputs": {"out": "out"},
        },
    }


def _attention_with_padding_mask_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"q": {}, "k": {}, "v": {}, "padding_mask": {}},
            "graph": [
                {
                    "attn": {
                        "_op": "attention",
                        "_args": ["q", "k", "v"],
                        "_bind": "out",
                        "mask": "padding_mask",
                        "padding_mask": True,
                        "causal": False,
                    }
                }
            ],
            "outputs": {"out": "out"},
        },
    }


def _concat_spec(*, dim: int = -1) -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}, "y": {}},
            "graph": [
                {
                    "cat": {
                        "_op": "concat",
                        "_args": ["x", "y"],
                        "_bind": "z",
                        "dim": dim,
                    }
                }
            ],
            "outputs": {"z": "z"},
        },
    }


def _linear_expert_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [
                {
                    "n": {
                        "_op": "linear",
                        "_args": "x",
                        "_bind": "y",
                        "bias": True,
                        "expert": 1,
                        "transpose": True,
                    }
                }
            ],
            "outputs": {"y": "y"},
        },
    }


def _mamba_scan_spec(*, with_state_input: bool, with_state_output: bool) -> dict[str, object]:
    inputs: dict[str, object] = {
        "u": {},
        "delta": {},
        "A": {},
        "B": {},
        "C": {},
        "D": {},
    }
    args: list[object] = ["u", "delta", "A", "B", "C", "D"]
    if with_state_input:
        inputs["state"] = {"optional": True}
        args.append("state")
    bind: object = ["y", "final_state"] if with_state_output else "y"
    outputs: dict[str, object] = {"y": "y"}
    if with_state_output:
        outputs["final_state"] = "final_state"
    return {
        "synapse": 1,
        "model": {
            "inputs": inputs,
            "graph": [{"scan": {"_op": "mamba_scan", "_args": args, "_bind": bind}}],
            "outputs": outputs,
        },
    }


def _mamba_scan_reference(
    *,
    u: torch.Tensor,
    delta: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    c: torch.Tensor,
    d: torch.Tensor,
    state: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    delta_sp = torch.nn.functional.softplus(delta)
    batch, seq, dim = u.shape
    state_dim = a.shape[1]
    cur = torch.zeros((batch, dim, state_dim), dtype=u.dtype) if state is None else state.clone()
    ys: list[torch.Tensor] = []
    for t in range(seq):
        u_t = u[:, t, :]
        delta_t = delta_sp[:, t, :]
        b_t = b[:, t, :]
        c_t = c[:, t, :]
        a_t = torch.exp(delta_t.unsqueeze(-1) * a.unsqueeze(0))
        bu_t = (delta_t * u_t).unsqueeze(-1) * b_t.unsqueeze(1)
        cur = a_t * cur + bu_t
        y_t = (cur * c_t.unsqueeze(1)).sum(dim=-1) + u_t * d.unsqueeze(0)
        ys.append(y_t)
    return torch.stack(ys, dim=1), cur


def _cache_state_generate_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "symbols": {"V": 8, "D": 4},
            "inputs": {
                "input_ids": {"shape": ["B", "T"], "dtype": "int64"},
                "cache_state": {"optional": True},
                "use_cache": {"optional": True},
            },
            "graph": [
                {"tok": {"_op": "embedding", "_args": "input_ids", "_bind": "x", "dim": "D"}},
                {
                    "head": {
                        "_op": "linear",
                        "_args": "x",
                        "_bind": "logits",
                        "dim": "V",
                        "bias": False,
                        "weight": "tok.weight",
                    }
                },
            ],
            "outputs": {"logits": "logits", "cache_state": "input_ids"},
        },
    }


def _split_interleave_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [
                {
                    "s": {
                        "_op": "split",
                        "_args": "x",
                        "_bind": ["even", "odd"],
                        "parts": 2,
                        "interleave": True,
                    }
                }
            ],
            "outputs": {"even": "even", "odd": "odd"},
        },
    }


def _clamp_sigmoid_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [
                {"c": {"_op": "clamp", "_args": "x", "_bind": "xc", "min": -1.0, "max": 1.0}},
                {"s": {"_op": "activations_sigmoid", "_args": "xc", "_bind": "y"}},
            ],
            "outputs": {"y": "y"},
        },
    }


def _xielu_spec() -> dict[str, object]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [
                {
                    "xielu": {
                        "_op": "activations_xielu",
                        "_args": "x",
                        "_bind": "y",
                        "_params": {
                            "alpha_p": "act.alpha_p",
                            "alpha_n": "act.alpha_n",
                            "beta": "act.beta",
                            "eps": "act.eps",
                        },
                    }
                }
            ],
            "outputs": {"y": "y"},
        },
    }


def _mxfp4_linear_state_dict() -> dict[str, torch.Tensor]:
    blocks = torch.tensor(
        [
            [
                [[0x00, 0x00]],
                [[0x00, 0x00]],
            ],
            [
                [[0x21, 0x43]],
                [[0x65, 0x87]],
            ],
        ],
        dtype=torch.uint8,
    )
    scales = torch.full((2, 2, 1), 127, dtype=torch.uint8)
    bias = torch.tensor(
        [
            [0.0, 0.0],
            [0.25, -0.75],
        ],
        dtype=torch.float32,
    )
    return {
        "n_blocks": blocks,
        "n_scales": scales,
        "n_bias": bias,
    }


def test_runtime_infer_param_path_uses_scope_for_undotted_explicit_param_name() -> None:
    model = SynapseProgramModel.from_spec({"synapse": 1, "model": {"graph": [], "outputs": {}}})
    node_spec = {"A": "A_log", "D": "D"}
    node_path = "backbone.layers.0.mixer.n_op_6"
    assert model._infer_param_path(node_spec, node_path=node_path, param_name="A") == (
        "backbone.layers.0.mixer.A_log"
    )
    assert model._infer_param_path(node_spec, node_path=node_path, param_name="D") == (
        "backbone.layers.0.mixer.D"
    )


def test_runtime_infer_param_path_keeps_double_at_absolute_under_param_root() -> None:
    model = SynapseProgramModel.from_spec({"synapse": 1, "model": {"graph": [], "outputs": {}}})
    model.load_state_dict_tensors(
        {"mlp.experts.0.gate_proj.weight": torch.ones(1, 1, dtype=torch.float32)}
    )
    model._param_roots_stack.append(["mlp.experts.0"])

    assert (
        model._infer_param_path(
            {"_params": {"weight": "@@gate_proj.weight"}},
            node_path="layers.0.mlp.experts.0.n_op_0",
            param_name="weight",
        )
        == "gate_proj.weight"
    )


def test_runtime_infer_param_path_keeps_double_at_absolute_under_scope() -> None:
    model = SynapseProgramModel.from_spec({"synapse": 1, "model": {"graph": [], "outputs": {}}})
    model.load_state_dict_tensors(
        {"mlp.experts.0.gate_proj.weight": torch.ones(1, 1, dtype=torch.float32)}
    )

    assert (
        model._infer_param_path(
            {"_params": {"weight": "@@gate_proj.weight"}},
            node_path="mlp.experts.0.n_op_0",
            param_name="weight",
        )
        == "gate_proj.weight"
    )


def test_runtime_scope_path_bound_calls_do_not_double_apply_local_scope() -> None:
    modules = parse_axon_program(
        """
D = 2
lin :: @Path -> Tensor[B,T,D] -> Tensor[B,T,D]
lin@path x = linear@path x dim=D bias=true
tiny :: Tensor[B,T,D] -> Tensor[B,T,D]
tiny x = do
  y <- scope@attn do
    return lin@proj x
  return y
"""
    )
    spec = lower_axon_program_to_synapse_spec(modules)
    model = SynapseProgramModel.from_spec(spec)
    model.load_state_dict_tensors(
        {
            "attn.proj.weight": torch.eye(2, dtype=torch.float32),
            "attn.proj.bias": torch.zeros(2, dtype=torch.float32),
        }
    )
    out = model(x=torch.tensor([[[1.0, 2.0]]], dtype=torch.float32))
    assert torch.equal(out["y"], torch.tensor([[[1.0, 2.0]]], dtype=torch.float32))


def test_runtime_layernorm_allows_explicit_weight_and_bias_names() -> None:
    modules = parse_axon_program(
        """
D = 2
tiny :: Tensor[B,T,D] -> Tensor[B,T,D]
tiny x = do
  y <- layernorm@norm x eps=1e-05 weight=gamma bias=beta
  return y
"""
    )
    spec = lower_axon_program_to_synapse_spec(modules)
    model = SynapseProgramModel.from_spec(spec)
    gamma = torch.tensor([1.5, 0.5], dtype=torch.float32)
    beta = torch.tensor([-0.25, 0.75], dtype=torch.float32)
    model.load_state_dict_tensors(
        {
            "norm.gamma": gamma,
            "norm.beta": beta,
        }
    )
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    out = model(x=x)["y"]
    expected = torch.nn.functional.layer_norm(x, (2,), weight=gamma, bias=beta, eps=1.0e-5)
    assert torch.allclose(out, expected, atol=1.0e-6, rtol=0.0)


@pytest.mark.parametrize("bias_literal", ["false", "null"])
def test_runtime_layernorm_allows_disabling_bias_via_literal(bias_literal: str) -> None:
    modules = parse_axon_program(
        f"""
D = 2
tiny :: Tensor[B,T,D] -> Tensor[B,T,D]
tiny x = do
  y <- layernorm@norm x eps=1e-05 weight=gamma bias={bias_literal}
  return y
"""
    )
    spec = lower_axon_program_to_synapse_spec(modules)
    model = SynapseProgramModel.from_spec(spec)
    gamma = torch.tensor([1.5, 0.5], dtype=torch.float32)
    model.load_state_dict_tensors({"norm.gamma": gamma})
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]], dtype=torch.float32)
    out = model(x=x)["y"]
    expected = torch.nn.functional.layer_norm(x, (2,), weight=gamma, bias=None, eps=1.0e-5)
    assert torch.allclose(out, expected, atol=1.0e-6, rtol=0.0)


def test_runtime_from_spec_and_from_yaml(tmp_path: Path) -> None:
    spec = _tiny_linear_spec()
    model = SynapseProgramModel.from_spec(spec)

    state_dict = {
        "embed_tokens.weight": torch.randn(8, 4),
    }
    model.load_state_dict_tensors(state_dict)

    input_ids = torch.randint(low=0, high=8, size=(2, 3), dtype=torch.long)
    logits = model(input_ids)
    assert logits.shape == (2, 3, 8)

    spec_path = tmp_path / "spec.yaml"
    spec_path.write_text(
        """synapse: 1
model:
  symbols:
    B: null
    T: null
    V: 8
    D: 4
  graph:
    - embed_tokens:
        _op: embedding
        _args: input_ids
        _bind: x
        dim: D
    - lm_head:
        _op: linear
        _args: x
        _bind: logits
        dim: V
        bias: false
        weight: embed_tokens.weight
  outputs:
    logits: logits
""",
        encoding="utf-8",
    )
    model_from_yaml = SynapseProgramModel.from_yaml(spec_path, state_dict=state_dict)
    logits_yaml = model_from_yaml(input_ids)
    assert logits_yaml.shape == (2, 3, 8)


def test_runtime_three_reshape_heads_infers_head_dim_from_heads() -> None:
    spec = _reshape_three_heads_spec(heads=12)
    model = SynapseProgramModel.from_spec(spec)
    q = torch.randn(2, 5, 768)
    out = model(q=q, k=q, v=q)
    assert out["qh"].shape == (2, 12, 5, 64)
    assert out["kh"].shape == (2, 12, 5, 64)
    assert out["vh"].shape == (2, 12, 5, 64)


def test_runtime_concat_last_dim_matches_torch_cat() -> None:
    spec = _concat_spec(dim=-1)
    model = SynapseProgramModel.from_spec(spec)
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 3, 5)
    out = model(x=x, y=y)
    assert torch.equal(out["z"], torch.cat([x, y], dim=-1))


def test_runtime_three_reshape_heads_infers_heads_from_head_dim() -> None:
    spec = _reshape_three_heads_spec(head_dim=64)
    model = SynapseProgramModel.from_spec(spec)
    q = torch.randn(2, 5, 768)
    out = model(q=q, k=q, v=q)
    assert out["qh"].shape == (2, 12, 5, 64)
    assert out["kh"].shape == (2, 12, 5, 64)
    assert out["vh"].shape == (2, 12, 5, 64)


def test_runtime_three_reshape_heads_requires_heads_or_head_dim() -> None:
    spec = _reshape_three_heads_spec()
    model = SynapseProgramModel.from_spec(spec)
    q = torch.randn(2, 5, 768)
    with pytest.raises(ValueError, match="requires heads or head_dim"):
        model(q=q, k=q, v=q)


def test_runtime_reshape_heads_infers_head_dim_from_heads() -> None:
    spec = _reshape_heads_spec(heads=12)
    model = SynapseProgramModel.from_spec(spec)
    x = torch.randn(2, 5, 768)
    out = model(x=x)
    assert out["xh"].shape == (2, 12, 5, 64)


def test_runtime_reshape_heads_infers_heads_from_head_dim() -> None:
    spec = _reshape_heads_spec(head_dim=64)
    model = SynapseProgramModel.from_spec(spec)
    x = torch.randn(2, 5, 768)
    out = model(x=x)
    assert out["xh"].shape == (2, 12, 5, 64)


def test_runtime_reshape_heads_requires_heads_or_head_dim() -> None:
    spec = _reshape_heads_spec()
    model = SynapseProgramModel.from_spec(spec)
    x = torch.randn(2, 5, 768)
    with pytest.raises(ValueError, match="requires heads or head_dim"):
        model(x=x)


def test_runtime_split_qkv_grouped_matches_reference_layout() -> None:
    spec = _split_qkv_grouped_spec(heads=4, kv_heads=2)
    model = SynapseProgramModel.from_spec(spec)
    x = torch.arange(1 * 2 * 24, dtype=torch.float32).view(1, 2, 24)
    out = model(x=x)

    grouped = x.view(1, 2, 2, 4, 3)
    q_ref = grouped[:, :, :, :2, :].reshape(1, 2, 4, 3).permute(0, 2, 1, 3)
    k_ref = grouped[:, :, :, -2, :].reshape(1, 2, 2, 3).permute(0, 2, 1, 3)
    v_ref = grouped[:, :, :, -1, :].reshape(1, 2, 2, 3).permute(0, 2, 1, 3)

    assert out["q"].shape == (1, 4, 2, 3)
    assert out["k"].shape == (1, 2, 2, 3)
    assert out["v"].shape == (1, 2, 2, 3)
    assert torch.equal(out["q"], q_ref)
    assert torch.equal(out["k"], k_ref)
    assert torch.equal(out["v"], v_ref)


def test_runtime_causal_mask_combines_padding_mask() -> None:
    spec = _causal_mask_with_padding_spec()
    model = SynapseProgramModel.from_spec(spec)
    q = torch.randn(2, 12, 4, 8)
    k = torch.randn(2, 12, 4, 8)
    padding_mask = torch.tensor(
        [[1, 1, 1, 1], [0, 0, 1, 1]],
        dtype=torch.long,
    )
    out = model(q=q, k=k, padding_mask=padding_mask)
    mask = out["mask"]
    assert mask.shape == (2, 1, 4, 4)
    assert torch.isfinite(mask[0]).all()
    assert torch.isfinite(mask[1, :, :, 2:]).all()
    assert torch.all(mask[1, :, :, :2] < -1.0e20)


def test_runtime_causal_mask_early_exit_for_trivial_full_causal_mask() -> None:
    spec = _causal_mask_early_exit_spec()
    model = SynapseProgramModel.from_spec(spec)
    q = torch.randn(1, 8, 4, 16)
    k = torch.randn(1, 8, 4, 16)

    out = model(q=q, k=k, padding_mask=torch.ones(1, 4, dtype=torch.long))
    assert out["mask"] is None

    out_nontrivial = model(q=q, k=k, padding_mask=torch.tensor([[1, 1, 0, 1]], dtype=torch.long))
    mask = out_nontrivial["mask"]
    assert isinstance(mask, torch.Tensor)
    assert mask.shape == (1, 1, 4, 4)


def test_runtime_blocksparse_mask_combines_pattern_with_padding_mask() -> None:
    spec = _blocksparse_mask_spec(block_size=2, local_blocks=1, vert_stride=2, homo_head=False)
    model = SynapseProgramModel.from_spec(spec)
    q = torch.randn(1, 4, 8, 16)
    k = torch.randn(1, 4, 8, 16)
    padding_mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 1]], dtype=torch.long)
    out = model(q=q, k=k, padding_mask=padding_mask)
    mask = out["mask"]
    assert mask.shape == (1, 4, 8, 8)
    assert torch.all(mask[..., 0] < -1.0e20)
    assert torch.isfinite(mask[..., 1:]).any()


def test_runtime_rope_pair_inv_freq_dtype_bfloat16_matches_quantized_reference() -> None:
    q = torch.randn(1, 2, 12, 8, dtype=torch.float32)
    k = torch.randn(1, 2, 12, 8, dtype=torch.float32)
    pos_ids = torch.arange(12, dtype=torch.long).unsqueeze(0)

    model_fp32 = SynapseProgramModel.from_spec(_rope_pair_spec(inv_freq_dtype=None))
    out_fp32 = model_fp32(q=q, k=k, position_ids=pos_ids)

    model_bf16 = SynapseProgramModel.from_spec(_rope_pair_spec(inv_freq_dtype="bfloat16"))
    out_bf16 = model_bf16(q=q, k=k, position_ids=pos_ids)

    half = q.shape[-1] // 2
    theta = 1_000_000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / float(half)))
    inv_freq = inv_freq.to(torch.bfloat16).to(torch.float32)
    pos = pos_ids.to(torch.float32)
    ang = pos.unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
    cos = torch.cos(ang).to(q.dtype).unsqueeze(1)
    sin = torch.sin(ang).to(q.dtype).unsqueeze(1)
    q1, q2 = q[..., :half], q[..., half:]
    q_ref = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos], dim=-1)

    assert torch.allclose(out_bf16["q_rot"], q_ref, atol=1.0e-6, rtol=1.0e-6)
    assert float((out_bf16["q_rot"] - out_fp32["q_rot"]).abs().max()) > 0.0


def test_runtime_rope_pair_partial_rotary_factor_preserves_suffix() -> None:
    q = torch.randn(1, 2, 5, 8, dtype=torch.float32)
    k = torch.randn(1, 2, 5, 8, dtype=torch.float32)
    pos_ids = torch.arange(5, dtype=torch.long).unsqueeze(0)

    model = SynapseProgramModel.from_spec(_rope_pair_partial_spec(partial_rotary_factor=0.5))
    out = model(q=q, k=k, position_ids=pos_ids)

    rotary_dim = 4
    half = rotary_dim // 2
    theta = 1_000_000.0
    inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32) / float(half)))
    ang = pos_ids.to(torch.float32).unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
    cos = torch.cos(ang).to(q.dtype).unsqueeze(1)
    sin = torch.sin(ang).to(q.dtype).unsqueeze(1)
    q1, q2 = q[..., :half], q[..., half:rotary_dim]
    k1, k2 = k[..., :half], k[..., half:rotary_dim]
    q_ref = torch.cat([q1 * cos - q2 * sin, q1 * sin + q2 * cos, q[..., rotary_dim:]], dim=-1)
    k_ref = torch.cat([k1 * cos - k2 * sin, k1 * sin + k2 * cos, k[..., rotary_dim:]], dim=-1)

    assert torch.allclose(out["q_rot"], q_ref, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(out["k_rot"], k_ref, atol=1.0e-6, rtol=1.0e-6)


def test_runtime_arange_positions_uses_attention_mask_for_left_padding() -> None:
    spec = _arange_positions_with_mask_spec()
    model = SynapseProgramModel.from_spec(spec)
    input_ids = torch.tensor([[10, 11, 12, 13], [0, 0, 20, 21]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]], dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    pos = out["pos"]
    assert torch.equal(pos[0], torch.tensor([0, 1, 2, 3], dtype=torch.long))
    assert torch.equal(pos[1], torch.tensor([0, 0, 0, 1], dtype=torch.long))


def test_runtime_arange_positions_without_mask_preserves_batch_dimension() -> None:
    spec = _arange_positions_with_mask_spec()
    model = SynapseProgramModel.from_spec(spec)
    input_ids = torch.tensor([[10, 11, 12, 13], [20, 21, 22, 23]], dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=None)
    pos = out["pos"]
    expected = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
    assert pos.shape == (2, 4)
    assert torch.equal(pos, expected)


def test_runtime_arange_positions_can_ignore_attention_mask() -> None:
    spec = _arange_positions_ignore_mask_spec()
    model = SynapseProgramModel.from_spec(spec)
    input_ids = torch.tensor([[10, 11, 12, 13], [0, 0, 20, 21]], dtype=torch.long)
    attention_mask = torch.tensor([[1, 1, 1, 1], [0, 0, 1, 1]], dtype=torch.long)
    out = model(input_ids=input_ids, attention_mask=attention_mask)
    expected = torch.tensor([[3, 4, 5, 6], [3, 4, 5, 6]], dtype=torch.long)
    assert torch.equal(out["pos"], expected)


def test_runtime_moe_select_selects_routed_rows() -> None:
    spec = _moe_select_spec(expert=1)
    model = SynapseProgramModel.from_spec(spec)
    x = torch.tensor([[[10.0, 11.0], [20.0, 21.0], [30.0, 31.0]]])
    scores = torch.tensor([[[0.7, 0.3], [0.2, 0.8], [0.6, 0.4]]])
    idx = torch.tensor([[[1, 0], [2, 1], [1, 2]]], dtype=torch.long)

    out = model(x=x, scores=scores, idx=idx)
    assert torch.equal(out["token_idx"], torch.tensor([0, 2, 1], dtype=torch.long))
    assert torch.equal(out["topk_pos"], torch.tensor([0, 0, 1], dtype=torch.long))
    assert torch.equal(out["x_sel"], torch.tensor([[10.0, 11.0], [30.0, 31.0], [20.0, 21.0]]))
    assert torch.allclose(out["sel_scores"], torch.tensor([0.7, 0.6, 0.8], dtype=torch.float32))


def test_runtime_moe_select_allows_empty_selection() -> None:
    spec = _moe_select_spec(expert=9)
    model = SynapseProgramModel.from_spec(spec)
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    scores = torch.tensor([[[0.6, 0.4], [0.2, 0.8]]])
    idx = torch.tensor([[[1, 2], [2, 1]]], dtype=torch.long)

    out = model(x=x, scores=scores, idx=idx)
    assert out["x_sel"].shape == (0, 2)
    assert out["token_idx"].numel() == 0
    assert out["topk_pos"].numel() == 0
    assert out["sel_scores"].numel() == 0


def test_runtime_moe_select_validates_flattened_token_alignment() -> None:
    spec = _moe_select_spec(expert=1)
    model = SynapseProgramModel.from_spec(spec)
    x = torch.randn(1, 3, 4)
    scores = torch.randn(1, 2, 2)
    idx = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.long)
    with pytest.raises(
        ValueError,
        match="moe_select hidden and topk tensors must align on flattened token count",
    ):
        model(x=x, scores=scores, idx=idx)


def test_runtime_moe_select_validates_index_dtype() -> None:
    spec = _moe_select_spec(expert=1)
    model = SynapseProgramModel.from_spec(spec)
    x = torch.randn(1, 2, 4)
    scores = torch.randn(1, 2, 2)
    idx = torch.randn(1, 2, 2)
    with pytest.raises(ValueError, match=r"moe_select topk_indices must be an integer tensor"):
        model(x=x, scores=scores, idx=idx)


def test_runtime_moe_select_requires_integral_expert() -> None:
    spec = _moe_select_spec(expert=1.5)
    model = SynapseProgramModel.from_spec(spec)
    x = torch.randn(1, 2, 4)
    scores = torch.randn(1, 2, 2)
    idx = torch.tensor([[[1, 0], [0, 1]]], dtype=torch.long)
    with pytest.raises(ValueError, match=r"moe_select expert must evaluate to an integer"):
        model(x=x, scores=scores, idx=idx)


def test_runtime_moe_scatter_add_accumulates_weighted_updates() -> None:
    spec = _moe_scatter_add_spec()
    model = SynapseProgramModel.from_spec(spec)
    m = torch.zeros(1, 3, 2)
    token_idx = torch.tensor([0, 2, 2], dtype=torch.long)
    upd = torch.tensor([[1.0, 2.0], [10.0, 20.0], [30.0, 40.0]])
    scores = torch.tensor([1.0, 0.5, 0.25])

    out = model(m=m.clone(), token_idx=token_idx, upd=upd, scores=scores)
    expected = m.reshape(-1, 2)
    expected[0] += torch.tensor([1.0, 2.0])
    expected[2] += torch.tensor([5.0, 10.0])
    expected[2] += torch.tensor([7.5, 10.0])
    assert torch.allclose(out["m_out"].reshape(-1, 2), expected)


def test_runtime_moe_scatter_add_empty_indices_is_noop() -> None:
    spec = _moe_scatter_add_spec()
    model = SynapseProgramModel.from_spec(spec)
    m = torch.randn(1, 2, 3)
    token_idx = torch.zeros((0,), dtype=torch.long)
    upd = torch.zeros((0, 3))
    scores = torch.zeros((0,))
    out = model(m=m.clone(), token_idx=token_idx, upd=upd, scores=scores)
    assert torch.equal(out["m_out"], m)


def test_runtime_moe_scatter_add_validates_alignment_and_dtypes() -> None:
    spec = _moe_scatter_add_spec()
    model = SynapseProgramModel.from_spec(spec)
    m = torch.zeros(1, 2, 4)
    with pytest.raises(ValueError, match=r"moe_scatter_add token_idx must be an integer tensor"):
        model(
            m=m.clone(),
            token_idx=torch.tensor([0.0, 1.0]),
            upd=torch.randn(2, 4),
            scores=torch.ones(2),
        )
    with pytest.raises(
        ValueError,
        match="moe_scatter_add token_idx, updates, and scores must align on row count",
    ):
        model(
            m=m.clone(),
            token_idx=torch.tensor([0, 1], dtype=torch.long),
            upd=torch.randn(3, 4),
            scores=torch.ones(2),
        )


def test_runtime_moe_scatter_add_validates_token_index_bounds() -> None:
    spec = _moe_scatter_add_spec()
    model = SynapseProgramModel.from_spec(spec)
    with pytest.raises(ValueError, match="moe_scatter_add token_idx contains out-of-range values"):
        model(
            m=torch.zeros(1, 2, 3),
            token_idx=torch.tensor([3], dtype=torch.long),
            upd=torch.randn(1, 3),
            scores=torch.ones(1),
        )


def test_runtime_moe_grouped_ffn_matches_grouped_reference() -> None:
    spec = _moe_grouped_ffn_spec()
    model = SynapseProgramModel.from_spec(spec)
    x = torch.tensor(
        [[[0.25, -0.5], [1.0, 0.75], [-0.25, 0.5]]],
        dtype=torch.bfloat16,
    )
    scores = torch.tensor(
        [[[0.7, 0.3], [0.1, 0.9], [0.4, 0.6]]],
        dtype=torch.bfloat16,
    )
    idx = torch.tensor(
        [[[0, 1], [1, 0], [0, 1]]],
        dtype=torch.long,
    )
    gate_up_weight = torch.tensor(
        [
            [[0.4, -0.3, 0.2, 0.1], [0.1, 0.2, -0.4, 0.5]],
            [[-0.2, 0.6, 0.3, -0.1], [0.5, -0.4, 0.2, 0.3]],
        ],
        dtype=torch.bfloat16,
    )
    gate_up_bias = torch.tensor(
        [[0.1, -0.2, 0.05, 0.2], [-0.1, 0.05, 0.15, -0.05]],
        dtype=torch.bfloat16,
    )
    down_weight = torch.tensor(
        [
            [[0.3, -0.1], [0.2, 0.4]],
            [[-0.2, 0.5], [0.6, -0.3]],
        ],
        dtype=torch.bfloat16,
    )
    down_bias = torch.tensor([[0.05, -0.1], [0.02, 0.03]], dtype=torch.bfloat16)
    state = {
        "mlp.experts.gate_up_proj.weight": gate_up_weight,
        "mlp.experts.gate_up_proj.bias": gate_up_bias,
        "mlp.experts.down_proj.weight": down_weight,
        "mlp.experts.down_proj.bias": down_bias,
    }
    model.load_state_dict_tensors(state)

    out = model(x=x, scores=scores, idx=idx)["m_out"]

    hidden_flat = x.reshape(-1, x.shape[-1])
    scores_flat = scores.reshape(-1, scores.shape[-1])
    idx_flat = idx.reshape(-1, idx.shape[-1])
    num_tokens = int(hidden_flat.shape[0])
    num_topk = int(idx_flat.shape[-1])
    token_idx = (
        torch.arange(num_tokens, device=hidden_flat.device)
        .unsqueeze(1)
        .expand(-1, num_topk)
        .reshape(-1)
    )
    sample_weights = scores_flat.reshape(-1)
    expert_ids = idx_flat.reshape(-1)
    selected_hidden = hidden_flat[token_idx]
    selected_gate_up_weight = gate_up_weight[expert_ids]
    selected_gate_up_bias = gate_up_bias[expert_ids]
    gate_up = (
        torch.bmm(selected_hidden.unsqueeze(1), selected_gate_up_weight).squeeze(1)
        + selected_gate_up_bias
    )
    gate = gate_up[..., ::2].clamp(max=7.0)
    up = gate_up[..., 1::2].clamp(min=-7.0, max=7.0)
    ff = (up + 1.0) * (gate * torch.sigmoid(gate * 1.702))
    selected_down_weight = down_weight[expert_ids]
    selected_down_bias = down_bias[expert_ids]
    down = torch.bmm(ff.unsqueeze(1), selected_down_weight).squeeze(1) + selected_down_bias
    weighted = down * sample_weights.unsqueeze(-1)
    expected = weighted.view(num_tokens, num_topk, hidden_flat.shape[-1]).sum(dim=1)
    expected = expected.to(dtype=x.dtype).reshape_as(x)

    assert torch.allclose(out, expected)


def test_runtime_moe_grouped_ffn_validates_input_shapes_and_types() -> None:
    spec = _moe_grouped_ffn_spec()
    model = SynapseProgramModel.from_spec(spec)
    model.load_state_dict_tensors(
        {
            "mlp.experts.gate_up_proj.weight": torch.randn(2, 2, 4),
            "mlp.experts.gate_up_proj.bias": torch.randn(2, 4),
            "mlp.experts.down_proj.weight": torch.randn(2, 2, 2),
            "mlp.experts.down_proj.bias": torch.randn(2, 2),
        }
    )
    with pytest.raises(ValueError, match="moe_grouped_ffn topk_indices must be integer"):
        model(
            x=torch.randn(1, 3, 2),
            scores=torch.randn(1, 3, 2),
            idx=torch.randn(1, 3, 2),
        )
    with pytest.raises(
        ValueError,
        match="moe_grouped_ffn hidden and topk tensors must align on flattened token count",
    ):
        model(
            x=torch.randn(1, 4, 2),
            scores=torch.randn(1, 3, 2),
            idx=torch.zeros(1, 3, 2, dtype=torch.long),
        )


def test_runtime_validates_input_rank_from_shape_spec() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "symbols": {},
            "inputs": {"x": {"shape": ["B", "T", "D"]}},
            "graph": [],
            "outputs": {"x": "x"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    with pytest.raises(ValueError, match="rank mismatch"):
        model(x=torch.randn(2, 3))


def test_runtime_validates_symbolic_shape_consistency_across_inputs() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "symbols": {},
            "inputs": {
                "x": {"shape": ["B", "T", "D"]},
                "mask": {"shape": ["B", "T"]},
            },
            "graph": [],
            "outputs": {"x": "x"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    with pytest.raises(ValueError, match="symbol T was previously bound"):
        model(x=torch.randn(2, 3, 4), mask=torch.ones(2, 5, dtype=torch.long))


def test_runtime_validates_numeric_symbol_dims_in_input_specs() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "symbols": {"D": 4},
            "inputs": {"x": {"shape": ["B", "T", "D"]}},
            "graph": [],
            "outputs": {"x": "x"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    with pytest.raises(ValueError, match="expected symbol D=4"):
        model(x=torch.randn(2, 3, 5))


def test_runtime_linear_handles_empty_batch_without_kernel_work() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [{"n": {"_op": "linear", "_args": "x", "_bind": "y", "bias": False}}],
            "outputs": {"y": "y"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    model.load_state_dict_tensors({"n.weight": torch.randn(8, 4)})
    x = torch.empty((0, 4), dtype=torch.float32)
    out = model(x=x)
    assert out["y"].shape == (0, 8)


def test_runtime_attention_supports_sink_logits_path() -> None:
    spec = _attention_with_sink_spec()
    model = SynapseProgramModel.from_spec(spec)
    q = torch.randn(1, 2, 3, 4)
    k = torch.randn(1, 2, 3, 4)
    v = torch.randn(1, 2, 3, 4)
    sink = torch.randn(2)
    out = model(q=q, k=k, v=v, sink=sink)
    assert out["out"].shape == (1, 2, 3, 4)
    assert torch.isfinite(out["out"]).all()


def test_runtime_attention_padding_mask_blocks_masked_keys() -> None:
    spec = _attention_with_padding_mask_spec()
    model = SynapseProgramModel.from_spec(spec)
    q = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    k = torch.zeros((1, 1, 2, 2), dtype=torch.float32)
    v = torch.tensor([[[[1.0, 0.0], [10.0, 0.0]]]], dtype=torch.float32)
    padding_mask = torch.tensor([[1, 0]], dtype=torch.long)
    out = model(q=q, k=k, v=v, padding_mask=padding_mask)
    expected = torch.tensor([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=torch.float32)
    assert torch.allclose(out["out"], expected, atol=1e-6, rtol=0.0)


def test_runtime_linear_expert_materializes_mxfp4_aliases() -> None:
    spec = _linear_expert_spec()
    model = SynapseProgramModel.from_spec(spec)
    model.load_state_dict_tensors(_mxfp4_linear_state_dict())
    x = torch.tensor([[1.0, 2.0, 3.0, 4.0]], dtype=torch.float32)
    out = model(x=x)
    expected = torch.tensor([[15.25, 28.25]], dtype=torch.float32)
    assert torch.allclose(out["y"], expected, atol=1e-6, rtol=0.0)


def test_runtime_split_supports_interleave_mode() -> None:
    spec = _split_interleave_spec()
    model = SynapseProgramModel.from_spec(spec)
    x = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
    out = model(x=x)
    assert torch.equal(out["even"], torch.tensor([[0.0, 2.0]], dtype=torch.float32))
    assert torch.equal(out["odd"], torch.tensor([[1.0, 3.0]], dtype=torch.float32))


def test_runtime_clamp_and_sigmoid_ops() -> None:
    spec = _clamp_sigmoid_spec()
    model = SynapseProgramModel.from_spec(spec)
    x = torch.tensor([[-2.0, 0.0, 2.0]], dtype=torch.float32)
    out = model(x=x)
    expected = torch.sigmoid(torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float32))
    assert torch.allclose(out["y"], expected, atol=1e-6, rtol=0.0)


def test_runtime_xielu_reads_learned_activation_parameters() -> None:
    spec = _xielu_spec()
    model = SynapseProgramModel.from_spec(spec)
    state_dict = {
        "act.alpha_p": torch.tensor([0.2], dtype=torch.float32),
        "act.alpha_n": torch.tensor([-0.4], dtype=torch.float32),
        "act.beta": torch.tensor([0.6], dtype=torch.float32),
        "act.eps": torch.tensor([-1.0e-4], dtype=torch.float32),
    }
    model.load_state_dict_tensors(state_dict)

    x = torch.tensor([[-1.25, -0.25, 0.5, 1.75]], dtype=torch.float32)
    out = model(x=x)

    alpha_p = torch.nn.functional.softplus(state_dict["act.alpha_p"])
    beta = state_dict["act.beta"]
    alpha_n = beta + torch.nn.functional.softplus(state_dict["act.alpha_n"])
    eps = state_dict["act.eps"]
    expected = torch.where(
        x > 0,
        alpha_p * x * x + beta * x,
        (torch.expm1(torch.minimum(x, eps)) - x) * alpha_n + beta * x,
    )
    assert torch.allclose(out["y"], expected, atol=1e-6, rtol=0.0)


def test_runtime_mamba_scan_matches_reference_without_state_io() -> None:
    spec = _mamba_scan_spec(with_state_input=False, with_state_output=False)
    model = SynapseProgramModel.from_spec(spec)
    u = torch.tensor(
        [[[0.2, -0.1], [0.4, 0.3], [0.7, -0.2]]],
        dtype=torch.float32,
    )
    delta = torch.tensor(
        [[[0.1, 0.5], [0.2, -0.3], [0.6, 0.4]]],
        dtype=torch.float32,
    )
    a = torch.tensor([[0.2, -0.1], [-0.3, 0.4]], dtype=torch.float32)
    b = torch.tensor([[[0.7, 0.2], [0.6, -0.2], [0.1, 0.3]]], dtype=torch.float32)
    c = torch.tensor([[[0.8, -0.5], [0.2, 0.4], [-0.1, 0.6]]], dtype=torch.float32)
    d = torch.tensor([0.9, -0.3], dtype=torch.float32)

    out = model(u=u, delta=delta, A=a, B=b, C=c, D=d)
    expected_y, _ = _mamba_scan_reference(u=u, delta=delta, a=a, b=b, c=c, d=d, state=None)
    assert torch.allclose(out["y"], expected_y, atol=1.0e-6, rtol=1.0e-6)


def test_runtime_mamba_scan_returns_final_state_with_state_input() -> None:
    spec = _mamba_scan_spec(with_state_input=True, with_state_output=True)
    model = SynapseProgramModel.from_spec(spec)
    u = torch.tensor(
        [[[0.3, 0.1], [0.5, -0.4]]],
        dtype=torch.float32,
    )
    delta = torch.tensor(
        [[[0.2, 0.7], [0.1, -0.2]]],
        dtype=torch.float32,
    )
    a = torch.tensor([[0.3, -0.2], [0.1, 0.2]], dtype=torch.float32)
    b = torch.tensor([[[0.4, 0.6], [0.9, -0.1]]], dtype=torch.float32)
    c = torch.tensor([[[0.5, -0.4], [0.8, 0.2]]], dtype=torch.float32)
    d = torch.tensor([0.7, -0.5], dtype=torch.float32)
    state = torch.tensor([[[0.1, -0.3], [0.2, 0.5]]], dtype=torch.float32)

    out = model(u=u, delta=delta, A=a, B=b, C=c, D=d, state=state)
    expected_y, expected_state = _mamba_scan_reference(
        u=u,
        delta=delta,
        a=a,
        b=b,
        c=c,
        d=d,
        state=state,
    )
    assert torch.allclose(out["y"], expected_y, atol=1.0e-6, rtol=1.0e-6)
    assert torch.allclose(out["final_state"], expected_state, atol=1.0e-6, rtol=1.0e-6)


def test_runtime_generate_uses_generic_cache_state_contract() -> None:
    spec = _cache_state_generate_spec()
    model = SynapseProgramModel.from_spec(spec)
    model.load_state_dict_tensors({"tok.weight": torch.randn(8, 4)})

    seen_lengths: list[int] = []
    original_forward = model.forward

    def _wrapped_forward(input_ids: torch.Tensor | None = None, **inputs: object):
        assert input_ids is not None
        seen_lengths.append(int(input_ids.shape[1]))
        return original_forward(input_ids, **inputs)

    model.forward = _wrapped_forward  # type: ignore[method-assign]
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generated = model.generate(input_ids=input_ids, eos_token_id=7, max_len=6)
    assert generated.shape[1] <= 6
    assert seen_lengths[0] == 3
    assert all(length == 1 for length in seen_lengths[1:])


def test_runtime_generate_accepts_singleton_non_logits_output_dict() -> None:
    spec = _tiny_linear_spec()
    model_spec = spec["model"]
    assert isinstance(model_spec, dict)
    model_spec["outputs"] = {"out_0": "logits"}
    model = SynapseProgramModel.from_spec(spec)
    model.load_state_dict_tensors({"embed_tokens.weight": torch.randn(8, 4)})
    input_ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    generated = model.generate(input_ids=input_ids, eos_token_id=7, max_len=6)
    assert generated.ndim == 2
    assert generated.shape[0] == 1
    assert generated.shape[1] <= 6


def test_runtime_records_intermediate_tensors_to_runtime_state_dict() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}, "y": {}},
            "graph": [{"sum_xy": {"_op": "add", "_args": ["x", "y"], "_bind": "z"}}],
            "outputs": {"z": "z"},
        },
    }
    runtime_state_dict = _InMemoryStateDict()
    model = SynapseProgramModel.from_spec(spec, runtime_state_dict=runtime_state_dict)
    out = model(x=torch.tensor([1.0, 2.0]), y=torch.tensor([3.0, 4.0]))
    expected = torch.tensor([4.0, 6.0])
    assert torch.equal(out["z"], expected)
    assert "sum_xy::z" in runtime_state_dict
    assert torch.equal(runtime_state_dict["sum_xy::z"], expected)


def test_runtime_config_primitives_resolve_config_values_and_defaults() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "config": {
                "hidden_size": 640,
                "name": "gemma3",
                "text_config": {"sliding_window": 512},
            },
            "graph": [
                {"k": {"_op": "_ir_expr", "_bind": "k", "value": "'text_config.sliding_window'"}},
                {"h": {"_op": "config_has", "_args": "k", "_bind": "has_key"}},
                {"s": {"_op": "config_int", "_args": "k", "_bind": "window"}},
                {
                    "d": {
                        "_op": "config_int",
                        "_args": "missing.value",
                        "_bind": "defaulted",
                        "default": 7,
                    }
                },
                {"f": {"_op": "config_float", "_args": "hidden_size", "_bind": "hidden_f"}},
                {"n": {"_op": "config_str", "_args": "name", "_bind": "name"}},
            ],
            "outputs": {
                "has_key": "has_key",
                "window": "window",
                "defaulted": "defaulted",
                "hidden_f": "hidden_f",
                "name": "name",
            },
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    out = model()
    assert out["has_key"] is True
    assert out["window"] == 512
    assert out["defaulted"] == 7
    assert out["hidden_f"] == 640.0
    assert out["name"] == "gemma3"


def test_runtime_config_value_and_expression_default() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "config": {
                "hidden_size": 640,
                "rope_scaling": {"long_factor": [1.0, 2.0, 3.0]},
            },
            "graph": [
                {
                    "v": {
                        "_op": "config_value",
                        "_args": "rope_scaling.long_factor",
                        "_bind": "vals",
                    }
                },
                {
                    "d": {
                        "_op": "config_int",
                        "_args": "missing.hidden_size",
                        "_bind": "fallback_hidden",
                        "default": {
                            "_expr": "call",
                            "callee": "Config.int",
                            "args": ["hidden_size"],
                            "kwargs": {},
                        },
                    }
                },
            ],
            "outputs": {"vals": "vals", "fallback_hidden": "fallback_hidden"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    out = model()
    assert out["vals"] == [1.0, 2.0, 3.0]
    assert out["fallback_hidden"] == 640


def test_runtime_config_primitives_support_root_kwarg() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "config": {"text_config": {"hidden_size": 4096}},
            "graph": [
                {
                    "h": {
                        "_op": "config_has",
                        "_args": "hidden_size",
                        "_bind": "has_h",
                        "root": "text_config",
                    }
                },
                {
                    "i": {
                        "_op": "config_int",
                        "_args": "hidden_size",
                        "_bind": "h",
                        "root": "text_config",
                    }
                },
                {
                    "d": {
                        "_op": "config_int",
                        "_args": "missing",
                        "_bind": "d",
                        "root": "text_config",
                        "default": 7,
                    }
                },
            ],
            "outputs": {"has_h": "has_h", "h": "h", "d": "d"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    out = model()
    assert out["has_h"] is True
    assert out["h"] == 4096
    assert out["d"] == 7


def test_runtime_config_expression_root_symbol_string() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "symbols": {"CFG": "text_config"},
            "config": {"text_config": {"hidden_size": 4096}},
            "graph": [
                {
                    "h": {
                        "_op": "_ir_expr",
                        "_bind": "h",
                        "value": {
                            "_expr": "call",
                            "callee": "Config.int",
                            "args": ["hidden_size"],
                            "kwargs": {"root": {"_expr": "name", "id": "CFG"}},
                        },
                    }
                }
            ],
            "outputs": {"h": "h"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    out = model()
    assert out["h"] == 4096


def test_runtime_config_int_missing_key_without_default_raises() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "config": {},
            "graph": [{"n": {"_op": "config_int", "_args": "hidden_size", "_bind": "h"}}],
            "outputs": {"h": "h"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    with pytest.raises(KeyError, match="config_int missing required config key"):
        model()


def test_runtime_params_primitives_detect_and_select_param_root() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {
                    "h": {
                        "_op": "params_has_root",
                        "_args": "language_model",
                        "_bind": "has_lm",
                    }
                },
                {
                    "r": {
                        "_op": "params_root",
                        "_args": "language_model",
                        "_bind": "root",
                        "default": "",
                    }
                },
            ],
            "outputs": {"has_lm": "has_lm", "root": "root"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    model.load_state_dict_tensors({"language_model.embed_tokens.weight": torch.randn(8, 4)})
    out = model()
    assert out["has_lm"] is True
    assert out["root"] == "language_model"


def test_runtime_params_primitives_fallback_when_root_missing() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {"h": {"_op": "params_has_root", "_args": "language_model", "_bind": "has_lm"}},
                {
                    "r": {
                        "_op": "params_root",
                        "_args": "language_model",
                        "_bind": "root",
                        "default": "",
                    }
                },
            ],
            "outputs": {"has_lm": "has_lm", "root": "root"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    model.load_state_dict_tensors({"embed_tokens.weight": torch.randn(8, 4)})
    out = model()
    assert out["has_lm"] is False
    assert out["root"] == ""


def test_runtime_param_root_guides_parameter_resolution() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [
                {"r": {"_op": "_ir_expr", "_bind": "root", "value": "'language_model'"}},
                {
                    "n": {
                        "_op": "linear",
                        "_args": "x",
                        "_bind": "y",
                        "_params": {"weight": "proj.weight", "bias": "proj.bias"},
                        "_param_root": {"_expr": "name", "id": "root"},
                    }
                },
            ],
            "outputs": {"y": "y"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    model.load_state_dict_tensors(
        {
            "language_model.proj.weight": torch.eye(2, dtype=torch.float32),
            "language_model.proj.bias": torch.zeros(2, dtype=torch.float32),
        }
    )
    out = model(x=torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    assert torch.equal(out["y"], torch.tensor([[1.0, 2.0]], dtype=torch.float32))


def test_runtime_prefers_existing_param_candidate_path() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "inputs": {"x": {}},
            "graph": [
                {
                    "n": {
                        "_op": "linear",
                        "_args": "x",
                        "_bind": "y",
                        "_params": {
                            "weight": ["missing.weight", "live.weight"],
                            "bias": ["missing.bias", "live.bias"],
                        },
                    }
                }
            ],
            "outputs": {"y": "y"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    model.load_state_dict_tensors(
        {
            "live.weight": torch.eye(2, dtype=torch.float32),
            "live.bias": torch.zeros(2, dtype=torch.float32),
        }
    )
    out = model(x=torch.tensor([[1.0, 2.0]], dtype=torch.float32))
    assert torch.equal(out["y"], torch.tensor([[1.0, 2.0]], dtype=torch.float32))


def test_runtime_sqrt_promotes_int_scalar_to_float() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {"d": {"_op": "_ir_expr", "_bind": "d", "value": 16}},
                {"s": {"_op": "sqrt", "_args": "d", "_bind": "s"}},
            ],
            "outputs": {"s": "s"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    out = model()
    assert isinstance(out["s"], float)
    assert out["s"] == 4.0


def test_runtime_log_promotes_int_scalar_to_float() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {"d": {"_op": "_ir_expr", "_bind": "d", "value": 16}},
                {"s": {"_op": "log", "_args": "d", "_bind": "s"}},
            ],
            "outputs": {"s": "s"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    out = model()
    assert isinstance(out["s"], float)
    assert out["s"] == pytest.approx(math.log(16.0))


def test_runtime_floor_promotes_int_scalar_to_int() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {"d": {"_op": "_ir_expr", "_bind": "d", "value": 16}},
                {"s": {"_op": "floor", "_args": "d", "_bind": "s"}},
            ],
            "outputs": {"s": "s"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    out = model()
    assert isinstance(out["s"], int)
    assert out["s"] == 16


def test_runtime_div_floor_log_support_tensor_pipeline() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {"half": {"_op": "_ir_expr", "_bind": "half", "value": 2.0}},
                {"one": {"_op": "_ir_expr", "_bind": "one", "value": 1.0}},
                {"scaled": {"_op": "div", "_args": ["x", "half"], "_bind": "scaled"}},
                {"bucket": {"_op": "floor", "_args": "scaled", "_bind": "bucket"}},
                {"shifted": {"_op": "add", "_args": ["bucket", "one"], "_bind": "shifted"}},
                {"y": {"_op": "log", "_args": "shifted", "_bind": "y"}},
            ],
            "inputs": {"x": {}},
            "outputs": {"y": "y"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    x = torch.tensor([[0.0, 1.0, 2.0, 5.0]], dtype=torch.float32)
    out = model(x=x)
    expected = torch.log(1.0 + torch.floor(x / 2.0))
    assert torch.allclose(out["y"], expected)


def test_runtime_reshape_supports_rank_change() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {
                    "y": {
                        "_op": "reshape",
                        "_args": "x",
                        "_bind": "y",
                        "shape": [2, 1, 3, 1],
                    }
                }
            ],
            "inputs": {"x": {}},
            "outputs": {"y": "y"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = model(x=x)
    assert tuple(out["y"].shape) == (2, 1, 3, 1)
    assert torch.equal(out["y"].reshape(2, 3), x)


def test_runtime_unsqueeze_supports_rank_change() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {"y": {"_op": "unsqueeze", "_args": "x", "_bind": "y", "dim": 1}},
                {"z": {"_op": "unsqueeze", "_args": "y", "_bind": "z", "dim": 3}},
            ],
            "inputs": {"x": {}},
            "outputs": {"z": "z"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    x = torch.arange(6, dtype=torch.float32).reshape(2, 3)
    out = model(x=x)
    assert tuple(out["z"].shape) == (2, 1, 3, 1)
    assert torch.equal(out["z"].reshape(2, 3), x)


def test_runtime_ir_expr_supports_inline_sqrt_with_config_call() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "config": {"text_config": {"query_pre_attn_scalar": 256}},
            "graph": [
                {
                    "a": {
                        "_op": "_ir_expr",
                        "_bind": "attn_scale",
                        "value": '1.0 / sqrt (Config.float "query_pre_attn_scalar" root="text_config" default=256.0)',
                    }
                }
            ],
            "outputs": {"attn_scale": "attn_scale"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    out = model()
    assert out["attn_scale"] == pytest.approx(0.0625)


def test_runtime_ir_expr_supports_inline_params_root_call() -> None:
    spec = {
        "synapse": 1,
        "model": {
            "graph": [
                {
                    "x": {
                        "_op": "_ir_expr",
                        "_bind": "chosen",
                        "value": 'Params.root "language_model" default=""',
                    }
                },
                {
                    "h": {
                        "_op": "_ir_expr",
                        "_bind": "has",
                        "value": 'Params.has_root "language_model"',
                    }
                },
            ],
            "outputs": {"chosen": "chosen", "has": "has"},
        },
    }
    model = SynapseProgramModel.from_spec(spec)
    model.load_state_dict_tensors({"language_model.embed_tokens.weight": torch.randn(8, 4)})
    out = model()
    assert out["chosen"] == "language_model"
    assert out["has"] is True
