from __future__ import annotations

from typing import Any

import torch
from torch.nn import functional as F

from brainsurgery.synapse import SynapseProgramModel
from tests.synapse_test_utils import build_codegen_model


def _make_log_bucket_position(
    relative_pos: torch.Tensor, *, bucket_size: int, max_position: int
) -> torch.Tensor:
    sign = torch.sign(relative_pos)
    mid = bucket_size // 2
    abs_pos = torch.where(
        (relative_pos < mid) & (relative_pos > -mid),
        torch.tensor(mid - 1, device=relative_pos.device, dtype=relative_pos.dtype),
        torch.abs(relative_pos),
    )
    log_pos = torch.ceil(
        torch.log(abs_pos.to(torch.float32) / float(mid))
        / torch.log(
            torch.tensor(
                (float(max_position) - 1.0) / float(mid),
                device=relative_pos.device,
                dtype=torch.float32,
            )
        )
        * float(mid - 1)
    ) + float(mid)
    return torch.where(
        abs_pos <= mid,
        relative_pos.to(dtype=log_pos.dtype),
        log_pos * sign.to(dtype=log_pos.dtype),
    )


def _build_relative_position(
    q_layer: torch.Tensor, k_layer: torch.Tensor, *, bucket_size: int, max_position: int
) -> torch.Tensor:
    q_size = int(q_layer.size(-2))
    k_size = int(k_layer.size(-2))
    q_ids = torch.arange(q_size, dtype=torch.long, device=q_layer.device)
    k_ids = torch.arange(k_size, dtype=torch.long, device=k_layer.device)
    rel_pos = q_ids[:, None] - k_ids[None, :]
    if bucket_size > 0 and max_position > 0:
        rel_pos = _make_log_bucket_position(
            rel_pos,
            bucket_size=bucket_size,
            max_position=max_position,
        )
    return rel_pos.to(torch.long).unsqueeze(0)


def _build_rpos(
    q_layer: torch.Tensor,
    k_layer: torch.Tensor,
    relative_pos: torch.Tensor,
    *,
    position_buckets: int,
    max_relative_positions: int,
) -> torch.Tensor:
    if int(k_layer.size(-2)) != int(q_layer.size(-2)):
        return _build_relative_position(
            k_layer,
            k_layer,
            bucket_size=position_buckets,
            max_position=max_relative_positions,
        )
    return relative_pos


def _transpose_for_scores(x: torch.Tensor, *, heads: int) -> torch.Tensor:
    new_shape = x.size()[:-1] + (heads, -1)
    x = x.view(new_shape)
    return x.permute(0, 2, 1, 3).contiguous().view(-1, x.size(1), x.size(-1))


def _reference_bias(
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    state: dict[str, torch.Tensor],
    share_att_key: bool,
    c2p: bool,
    p2c: bool,
    apply_rel_layernorm: bool,
    position_buckets: int,
    max_relative_positions: int,
) -> torch.Tensor:
    batch = int(q.shape[0])
    heads = int(q.shape[1])
    q_len = int(q.shape[-2])
    k_len = int(k.shape[-2])
    head_dim = int(q.shape[-1])

    rel_embeddings = state["attn.rel_embeddings.weight"][: (position_buckets * 2), :]
    if apply_rel_layernorm:
        rel_embeddings = F.layer_norm(
            rel_embeddings,
            (int(rel_embeddings.shape[-1]),),
            weight=state["attn.rel_norm.weight"],
            bias=state["attn.rel_norm.bias"],
            eps=1e-7,
        )
    rel_embeddings = rel_embeddings.to(dtype=q.dtype, device=q.device).unsqueeze(0)

    query_layer = q.contiguous().view(batch * heads, q_len, head_dim)
    key_layer = k.contiguous().view(batch * heads, k_len, head_dim)

    if share_att_key:
        pos_query_layer = _transpose_for_scores(
            F.linear(
                rel_embeddings,
                state["attn.query_proj.weight"].to(dtype=q.dtype, device=q.device),
                state["attn.query_proj.bias"].to(dtype=q.dtype, device=q.device),
            ),
            heads=heads,
        ).repeat(batch, 1, 1)
        pos_key_layer = _transpose_for_scores(
            F.linear(
                rel_embeddings,
                state["attn.key_proj.weight"].to(dtype=q.dtype, device=q.device),
                state["attn.key_proj.bias"].to(dtype=q.dtype, device=q.device),
            ),
            heads=heads,
        ).repeat(batch, 1, 1)
    else:
        pos_query_layer = None
        pos_key_layer = None
        if p2c:
            pos_query_layer = _transpose_for_scores(
                F.linear(
                    rel_embeddings,
                    state["attn.pos_query_proj.weight"].to(dtype=q.dtype, device=q.device),
                    state["attn.pos_query_proj.bias"].to(dtype=q.dtype, device=q.device),
                ),
                heads=heads,
            ).repeat(batch, 1, 1)
        if c2p:
            pos_key_layer = _transpose_for_scores(
                F.linear(
                    rel_embeddings,
                    state["attn.pos_key_proj.weight"].to(dtype=q.dtype, device=q.device),
                    state["attn.pos_key_proj.bias"].to(dtype=q.dtype, device=q.device),
                ),
                heads=heads,
            ).repeat(batch, 1, 1)

    relative_pos = _build_relative_position(
        query_layer,
        key_layer,
        bucket_size=position_buckets,
        max_position=max_relative_positions,
    ).unsqueeze(1)
    att_span = position_buckets
    scale_factor = 1 + int(c2p) + int(p2c)
    score = torch.zeros((batch * heads, q_len, k_len), dtype=q.dtype, device=q.device)

    if c2p:
        assert pos_key_layer is not None
        scale = torch.sqrt(
            torch.tensor(float(pos_key_layer.size(-1)), dtype=torch.float32, device=q.device)
            * float(scale_factor)
        )
        c2p_att = torch.bmm(query_layer, pos_key_layer.transpose(-1, -2))
        c2p_pos = torch.clamp(relative_pos + att_span, 0, att_span * 2 - 1)
        c2p_att = torch.gather(
            c2p_att,
            dim=-1,
            index=c2p_pos.squeeze(0).expand(
                [query_layer.size(0), query_layer.size(1), relative_pos.size(-1)]
            ),
        )
        score = score + (c2p_att / scale.to(dtype=c2p_att.dtype))

    if p2c:
        assert pos_query_layer is not None
        scale = torch.sqrt(
            torch.tensor(float(pos_query_layer.size(-1)), dtype=torch.float32, device=q.device)
            * float(scale_factor)
        )
        # Keep the same argument order as HF DeBERTa v2.
        r_pos = _build_rpos(
            query_layer,
            key_layer,
            relative_pos,
            position_buckets=max_relative_positions,
            max_relative_positions=position_buckets,
        )
        p2c_pos = torch.clamp(-r_pos + att_span, 0, att_span * 2 - 1)
        p2c_att = torch.bmm(key_layer, pos_query_layer.transpose(-1, -2))
        p2c_att = torch.gather(
            p2c_att,
            dim=-1,
            index=p2c_pos.squeeze(0).expand(
                [query_layer.size(0), key_layer.size(-2), key_layer.size(-2)]
            ),
        ).transpose(-1, -2)
        score = score + (p2c_att / scale.to(dtype=p2c_att.dtype))

    return score.view(batch, heads, q_len, k_len)


def _spec(
    *, share_att_key: bool, c2p: bool, p2c: bool, apply_rel_layernorm: bool
) -> dict[str, Any]:
    return {
        "synapse": 1,
        "model": {
            "inputs": {"q": {}, "k": {}},
            "graph": [
                {
                    "bias": {
                        "_op": "disentangled_relative_bias",
                        "_scope": "attn",
                        "_args": ["q", "k"],
                        "_bind": "bias",
                        "rel_embeddings": "rel_embeddings.weight",
                        "position_buckets": 16,
                        "max_relative_positions": 32,
                        "share_att_key": share_att_key,
                        "c2p": c2p,
                        "p2c": p2c,
                        "apply_rel_layernorm": apply_rel_layernorm,
                        "rel_norm_weight": "rel_norm.weight",
                        "rel_norm_bias": "rel_norm.bias",
                    }
                }
            ],
            "outputs": {"bias": "bias"},
        },
    }


@torch.inference_mode()
def test_disentangled_relative_bias_matches_reference_share_att_key_true() -> None:
    torch.manual_seed(0)
    spec = _spec(share_att_key=True, c2p=True, p2c=True, apply_rel_layernorm=True)
    state = {
        "attn.rel_embeddings.weight": torch.randn(64, 24, dtype=torch.float32),
        "attn.rel_norm.weight": torch.randn(24, dtype=torch.float32),
        "attn.rel_norm.bias": torch.randn(24, dtype=torch.float32),
        "attn.query_proj.weight": torch.randn(12, 24, dtype=torch.float32),
        "attn.query_proj.bias": torch.randn(12, dtype=torch.float32),
        "attn.key_proj.weight": torch.randn(12, 24, dtype=torch.float32),
        "attn.key_proj.bias": torch.randn(12, dtype=torch.float32),
    }

    runtime = SynapseProgramModel.from_spec(spec, state_dict=state).eval()
    codegen = build_codegen_model(spec, "DisentangledRelativeBiasShareGenerated", state).eval()

    q = torch.randn(2, 3, 5, 4, dtype=torch.float32)
    k = torch.randn(2, 3, 5, 4, dtype=torch.float32)

    expected = _reference_bias(
        q=q,
        k=k,
        state=state,
        share_att_key=True,
        c2p=True,
        p2c=True,
        apply_rel_layernorm=True,
        position_buckets=16,
        max_relative_positions=32,
    )
    runtime_bias = runtime(q=q, k=k)["bias"]
    codegen_bias = codegen(q=q, k=k)["bias"]

    assert runtime_bias.shape == (2, 3, 5, 5)
    assert torch.allclose(runtime_bias, expected, atol=0.0, rtol=0.0)
    assert torch.allclose(codegen_bias, expected, atol=0.0, rtol=0.0)


@torch.inference_mode()
def test_disentangled_relative_bias_matches_reference_separate_pos_projections() -> None:
    torch.manual_seed(0)
    spec = _spec(share_att_key=False, c2p=True, p2c=True, apply_rel_layernorm=False)
    state = {
        "attn.rel_embeddings.weight": torch.randn(64, 24, dtype=torch.float32),
        "attn.pos_query_proj.weight": torch.randn(12, 24, dtype=torch.float32),
        "attn.pos_query_proj.bias": torch.randn(12, dtype=torch.float32),
        "attn.pos_key_proj.weight": torch.randn(12, 24, dtype=torch.float32),
        "attn.pos_key_proj.bias": torch.randn(12, dtype=torch.float32),
    }

    runtime = SynapseProgramModel.from_spec(spec, state_dict=state).eval()
    codegen = build_codegen_model(spec, "DisentangledRelativeBiasSplitGenerated", state).eval()

    q = torch.randn(1, 3, 6, 4, dtype=torch.float32)
    k = torch.randn(1, 3, 6, 4, dtype=torch.float32)

    expected = _reference_bias(
        q=q,
        k=k,
        state=state,
        share_att_key=False,
        c2p=True,
        p2c=True,
        apply_rel_layernorm=False,
        position_buckets=16,
        max_relative_positions=32,
    )
    runtime_bias = runtime(q=q, k=k)["bias"]
    codegen_bias = codegen(q=q, k=k)["bias"]

    assert runtime_bias.shape == (1, 3, 6, 6)
    assert torch.allclose(runtime_bias, expected, atol=0.0, rtol=0.0)
    assert torch.allclose(codegen_bias, expected, atol=0.0, rtol=0.0)
