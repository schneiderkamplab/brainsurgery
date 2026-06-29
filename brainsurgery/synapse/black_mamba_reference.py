from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class BlackMambaConfig:
    num_layers: int
    hidden_size: int
    state_size: int
    expansion_factor: int
    ffn_hidden_size: int
    num_experts: int
    eps: float = 1.0e-5
    top_k: int = 2


def is_black_mamba_config_dir(model_dir: Path) -> bool:
    config_path = model_dir / "config.json"
    if not config_path.exists():
        return False
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    return "mamba_moe_layers" in payload and "hidden_size" in payload and "num_layers" in payload


def _load_black_mamba_config(model_dir: Path) -> BlackMambaConfig:
    payload = json.loads((model_dir / "config.json").read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("BlackMamba config.json must be a mapping")

    mamba_moe_layers = payload.get("mamba_moe_layers")
    if not isinstance(mamba_moe_layers, list) or not mamba_moe_layers:
        raise ValueError("BlackMamba config missing mamba_moe_layers")
    first_expert_layer = next((x for x in mamba_moe_layers if str(x) != "r"), None)
    if first_expert_layer is None:
        raise ValueError("BlackMamba config has no MoE layer marker")
    num_experts = int(first_expert_layer)

    return BlackMambaConfig(
        num_layers=int(payload["num_layers"]),
        hidden_size=int(payload["hidden_size"]),
        state_size=int(payload["state_size"]),
        expansion_factor=int(payload["expansion_factor"]),
        ffn_hidden_size=int(payload["ffn_hidden_size"]),
        num_experts=num_experts,
    )


def _rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    x_float = x.float()
    y = x_float * torch.rsqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + float(eps))
    return (y * weight.float()).to(dtype=x.dtype)


class BlackMambaReferenceModel(nn.Module):
    def __init__(self, *, config: BlackMambaConfig, state_dict: dict[str, torch.Tensor]) -> None:
        super().__init__()
        self.config = config
        self._state = state_dict
        self._inner = config.hidden_size * config.expansion_factor

    @classmethod
    def from_state_dict(
        cls,
        *,
        model_dir: Path,
        state_dict: dict[str, torch.Tensor],
    ) -> "BlackMambaReferenceModel":
        return cls(config=_load_black_mamba_config(model_dir), state_dict=state_dict)

    def _linear(
        self, x: torch.Tensor, weight_key: str, bias_key: str | None = None
    ) -> torch.Tensor:
        weight = self._state[weight_key]
        bias = self._state.get(bias_key) if bias_key is not None else None
        if x.is_floating_point() and weight.is_floating_point() and x.dtype != weight.dtype:
            weight = weight.to(dtype=x.dtype)
            if bias is not None and bias.is_floating_point() and bias.dtype != x.dtype:
                bias = bias.to(dtype=x.dtype)
        return F.linear(x, weight, bias)

    def _mamba_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        pref = f"decoder.layers.{layer_idx}.mixer"
        norm_w = self._state[f"decoder.layers.{layer_idx}.norm.weight"]
        x_norm = _rms_norm(x, norm_w, self.config.eps)

        in_proj = self._linear(x_norm, f"{pref}.in_proj.weight")
        u, gate = torch.split(in_proj, [self._inner, self._inner], dim=-1)

        conv_w = self._state[f"{pref}.conv1d.weight"]
        conv_b = self._state[f"{pref}.conv1d.bias"]
        u_t = u.transpose(1, 2).contiguous()
        u_conv_t = F.conv1d(
            u_t,
            conv_w.to(dtype=u.dtype),
            bias=conv_b.to(dtype=u.dtype),
            stride=1,
            padding=int(conv_w.shape[-1]) - 1,
            groups=self._inner,
        )[..., : int(u_t.shape[-1])]
        u_conv = F.silu(u_conv_t).transpose(1, 2).contiguous()

        x_proj = self._linear(u_conv, f"{pref}.x_proj.weight")
        dt_rank = int(self._state[f"{pref}.dt_proj.weight"].shape[1])
        dt, b, c = torch.split(
            x_proj, [dt_rank, self.config.state_size, self.config.state_size], dim=-1
        )
        delta = self._linear(dt, f"{pref}.dt_proj.weight", f"{pref}.dt_proj.bias")

        a_log = self._state[f"{pref}.A_log"]
        d = self._state[f"{pref}.D"]
        batch, seq, inner = int(u_conv.shape[0]), int(u_conv.shape[1]), int(u_conv.shape[2])
        state_dim = int(a_log.shape[1])
        work_dtype = torch.float32
        u_work = u_conv.to(dtype=work_dtype)
        delta_work = F.softplus(delta.to(dtype=work_dtype))
        a_work = -torch.exp(a_log.to(dtype=work_dtype))
        b_work = b.to(dtype=work_dtype)
        c_work = c.to(dtype=work_dtype)
        d_work = d.to(dtype=work_dtype)
        state = torch.zeros((batch, inner, state_dim), device=u_conv.device, dtype=work_dtype)
        outs: list[torch.Tensor] = []
        for t in range(seq):
            u_t = u_work[:, t, :]
            delta_t = delta_work[:, t, :]
            b_t = b_work[:, t, :]
            c_t = c_work[:, t, :]
            a_t = torch.exp(delta_t.unsqueeze(-1) * a_work.unsqueeze(0))
            bu_t = (delta_t * u_t).unsqueeze(-1) * b_t.unsqueeze(1)
            state = a_t * state + bu_t
            y_t = (state * c_t.unsqueeze(1)).sum(dim=-1) + u_t * d_work.unsqueeze(0)
            outs.append(y_t)
        y_scan = torch.stack(outs, dim=1).to(dtype=u_conv.dtype)
        y = self._linear(y_scan * F.silu(gate), f"{pref}.out_proj.weight")
        return x + y

    def _moe_layer(self, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
        pref = f"decoder.layers.{layer_idx}.mixer"
        norm_w = self._state[f"decoder.layers.{layer_idx}.norm.weight"]
        x_norm = _rms_norm(x, norm_w, self.config.eps)

        router_logits = self._linear(x_norm, f"{pref}.router.weight", f"{pref}.router.bias")
        router_probs = F.softmax(router_logits.float(), dim=-1).to(dtype=x.dtype)
        topk_scores, topk_idx = torch.topk(router_probs, k=self.config.top_k, dim=-1)

        out = torch.zeros_like(x_norm)
        for expert_idx in range(self.config.num_experts):
            fc1 = self._linear(x_norm, f"{pref}.local_experts.{expert_idx}.linear_fc1.weight")
            gate, up = torch.split(
                fc1, [self.config.ffn_hidden_size, self.config.ffn_hidden_size], dim=-1
            )
            hidden = F.silu(gate) * up
            expert_out = self._linear(
                hidden, f"{pref}.local_experts.{expert_idx}.linear_fc2.weight"
            )
            score = torch.zeros_like(topk_scores[..., 0])
            for k in range(self.config.top_k):
                score = score + torch.where(
                    topk_idx[..., k] == expert_idx,
                    topk_scores[..., k],
                    torch.zeros_like(score),
                )
            out = out + expert_out * score.unsqueeze(-1)
        return x + out

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        use_cache: bool = False,
        **_: Any,
    ) -> Any:
        del attention_mask, use_cache
        x = F.embedding(input_ids, self._state["embedding.word_embeddings.weight"])
        for i in range(self.config.num_layers):
            mamba_key = f"decoder.layers.{i}.mixer.A_log"
            router_key = f"decoder.layers.{i}.mixer.router.weight"
            if mamba_key in self._state:
                x = self._mamba_layer(x, i)
            elif router_key in self._state:
                x = self._moe_layer(x, i)
            else:
                raise ValueError(f"Unsupported BlackMamba layer layout at index {i}")

        final_w = self._state["decoder.final_layernorm.weight"]
        x = _rms_norm(x, final_w, self.config.eps)
        lm_head = self._state.get("lm_head.weight", self._state["embedding.word_embeddings.weight"])
        logits = F.linear(x, lm_head.to(dtype=x.dtype))
        return SimpleNamespace(logits=logits)

    def generate(
        self,
        input_ids: torch.Tensor,
        *,
        max_new_tokens: int,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        generated = input_ids.clone()
        for _ in range(max_new_tokens):
            logits = self.forward(generated).logits
            next_tok = torch.argmax(logits[:, -1, :], dim=-1)
            generated = torch.cat([generated, next_tok.unsqueeze(-1)], dim=-1)
            if eos_token_id is not None:
                if bool((next_tok == eos_token_id).all()):
                    break
        if pad_token_id is not None and eos_token_id is not None:
            generated = torch.where(
                generated == eos_token_id,
                torch.full_like(generated, eos_token_id),
                generated,
            )
        return generated


__all__ = ["BlackMambaReferenceModel", "is_black_mamba_config_dir"]
