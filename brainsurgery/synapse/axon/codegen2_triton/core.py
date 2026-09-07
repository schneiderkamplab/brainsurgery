from __future__ import annotations

from typing import Any

from ..codegen2_torch.core import _DirectTorchEmitter
from ..graph_ir import GraphProgram, validate_graph_program


def triton_op_table_markdown(_graph: GraphProgram) -> str:
    """Diagnostic placeholder for backend-specific Triton lowering coverage."""

    return "| Op | Count | Reason |\n|---|---:|---|\n"


class _DirectTritonEmitter(_DirectTorchEmitter):
    """Triton backend scaffold.

    The initial backend keeps the existing Torch tensor ABI and generated model
    contract. Triton kernels should be introduced here behind `__triton_*`
    Graph IR intrinsics, not by adding model-specific codegen branches.
    """

    def _emit_common(self, lines: list[str]) -> None:
        super()._emit_common(lines)
        add = self._add
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _rmsnorm_noscale(cls, x, eps=1e-6, dim=None, cast_float=False):")
        add(lines, 8, "if dim is not None and int(dim) != -1:")
        add(lines, 12, "raise ValueError('__triton_rmsnorm_noscale only supports dim=None/-1')")
        add(lines, 8, "if triton is None or _axon_triton_rmsnorm_noscale_kernel is None or not torch.is_tensor(x) or not x.is_cuda:")
        add(lines, 12, "raise RuntimeError('__triton_rmsnorm_noscale requires Triton and a CUDA tensor')")
        add(lines, 8, "last_dim = int(x.shape[-1])")
        add(lines, 8, "if last_dim <= 0:")
        add(lines, 12, "return torch.empty_like(x)")
        add(lines, 8, "x_in = x if x.is_contiguous() else x.contiguous()")
        add(lines, 8, "out = torch.empty_like(x_in)")
        add(lines, 8, "rows = out.numel() // last_dim")
        add(lines, 8, "if rows == 0:")
        add(lines, 12, "return out.reshape(x.shape)")
        add(lines, 8, "block = triton.next_power_of_2(last_dim)")
        add(lines, 8, "if block > 32768:")
        add(lines, 12, "raise ValueError('__triton_rmsnorm_noscale last dimension is too large for the Triton kernel')")
        add(lines, 8, "_axon_triton_debug_count('rmsnorm_noscale')")
        add(lines, 8, "_axon_triton_rmsnorm_noscale_kernel[(rows,)](x_in, out, rows, last_dim, float(eps), BLOCK=block)")
        add(lines, 8, "return out.reshape(x.shape)")
        add(lines, 4, "")
        add(lines, 4, "def _rmsnorm_scaled(self, x, scale, eps=1e-6, dim=None, cast_float=False):")
        add(lines, 8, "if not torch.is_tensor(scale):")
        add(lines, 12, "scale = self._param(scale)")
        add(lines, 8, "if torch.is_tensor(x):")
        add(lines, 12, "scale = self._move_to(scale, x.device)")
        add(lines, 8, "if dim is not None and int(dim) != -1:")
        add(lines, 12, "raise ValueError('__triton_rmsnorm_scaled only supports dim=None/-1')")
        add(lines, 8, "if triton is None or _axon_triton_rmsnorm_scaled_kernel is None or not torch.is_tensor(x) or not torch.is_tensor(scale) or not x.is_cuda or not scale.is_cuda:")
        add(lines, 12, "raise RuntimeError('__triton_rmsnorm_scaled requires Triton and CUDA tensors')")
        add(lines, 8, "last_dim = int(x.shape[-1])")
        add(lines, 8, "if last_dim <= 0:")
        add(lines, 12, "return torch.empty_like(x)")
        add(lines, 8, "if int(scale.numel()) != last_dim:")
        add(lines, 12, "raise ValueError('__triton_rmsnorm_scaled requires scale.numel() == x.shape[-1]')")
        add(lines, 8, "x_in = x if x.is_contiguous() else x.contiguous()")
        add(lines, 8, "scale_in = scale.reshape(-1)")
        add(lines, 8, "scale_in = scale_in if scale_in.is_contiguous() else scale_in.contiguous()")
        add(lines, 8, "out = torch.empty_like(x_in)")
        add(lines, 8, "rows = out.numel() // last_dim")
        add(lines, 8, "if rows == 0:")
        add(lines, 12, "return out.reshape(x.shape)")
        add(lines, 8, "block = triton.next_power_of_2(last_dim)")
        add(lines, 8, "if block > 32768:")
        add(lines, 12, "raise ValueError('__triton_rmsnorm_scaled last dimension is too large for the Triton kernel')")
        add(lines, 8, "_axon_triton_debug_count('rmsnorm_scaled')")
        add(lines, 8, "_axon_triton_rmsnorm_scaled_kernel[(rows,)](x_in, scale_in, out, rows, last_dim, float(eps), BLOCK=block)")
        add(lines, 8, "return out.reshape(x.shape)")
        add(lines, 4, "")
        add(lines, 4, "def _rmsnorm_unit_offset_scaled(self, x, scale, eps=1e-6, dim=None, cast_float=False):")
        add(lines, 8, "if not torch.is_tensor(scale):")
        add(lines, 12, "scale = self._param(scale)")
        add(lines, 8, "if torch.is_tensor(x):")
        add(lines, 12, "scale = self._move_to(scale, x.device)")
        add(lines, 8, "if dim is not None and int(dim) != -1:")
        add(lines, 12, "raise ValueError('__triton_rmsnorm_unit_offset_scaled only supports dim=None/-1')")
        add(lines, 8, "if triton is None or _axon_triton_rmsnorm_unit_offset_scaled_kernel is None or not torch.is_tensor(x) or not torch.is_tensor(scale) or not x.is_cuda or not scale.is_cuda:")
        add(lines, 12, "raise RuntimeError('__triton_rmsnorm_unit_offset_scaled requires Triton and CUDA tensors')")
        add(lines, 8, "last_dim = int(x.shape[-1])")
        add(lines, 8, "if last_dim <= 0:")
        add(lines, 12, "return torch.empty_like(x)")
        add(lines, 8, "if int(scale.numel()) != last_dim:")
        add(lines, 12, "raise ValueError('__triton_rmsnorm_unit_offset_scaled requires scale.numel() == x.shape[-1]')")
        add(lines, 8, "x_in = x if x.is_contiguous() else x.contiguous()")
        add(lines, 8, "scale_in = scale.reshape(-1)")
        add(lines, 8, "scale_in = scale_in if scale_in.is_contiguous() else scale_in.contiguous()")
        add(lines, 8, "out = torch.empty_like(x_in)")
        add(lines, 8, "rows = out.numel() // last_dim")
        add(lines, 8, "if rows == 0:")
        add(lines, 12, "return out.reshape(x.shape)")
        add(lines, 8, "block = triton.next_power_of_2(last_dim)")
        add(lines, 8, "if block > 32768:")
        add(lines, 12, "raise ValueError('__triton_rmsnorm_unit_offset_scaled last dimension is too large for the Triton kernel')")
        add(lines, 8, "_axon_triton_debug_count('rmsnorm_unit_offset_scaled')")
        add(lines, 8, "_axon_triton_rmsnorm_unit_offset_scaled_kernel[(rows,)](x_in, scale_in, out, rows, last_dim, float(eps), BLOCK=block)")
        add(lines, 8, "return out.reshape(x.shape)")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _rope_apply_factors(cls, x, sin, cos, interleaved=False):")
        add(lines, 8, "if interleaved:")
        add(lines, 12, "raise NotImplementedError('__torch_rope_apply_factors only supports non-interleaved RoPE')")
        add(lines, 8, "if torch.is_tensor(x):")
        add(lines, 12, "sin = cls._move_to(sin, x.device)")
        add(lines, 12, "cos = cls._move_to(cos, x.device)")
        add(lines, 8, "half = x.shape[-1] // 2")
        add(lines, 8, "if half * 2 != x.shape[-1]:")
        add(lines, 12, "raise ValueError('non-interleaved RoPE requires an even last dimension')")
        add(lines, 8, "if triton is None or _axon_triton_rope_apply_kernel is None or not torch.is_tensor(x) or not x.is_cuda:")
        add(lines, 12, "raise RuntimeError('__triton_rope_apply_factors requires Triton and a CUDA tensor')")
        add(lines, 8, "try:")
        add(lines, 12, "sin_expanded = sin.expand_as(x)")
        add(lines, 12, "cos_expanded = cos.expand_as(x)")
        add(lines, 8, "except RuntimeError:")
        add(lines, 12, "raise ValueError('__triton_rope_apply_factors requires sin/cos to expand to x')")
        add(lines, 8, "x_in = x if x.is_contiguous() else x.contiguous()")
        add(lines, 8, "sin_in = sin_expanded if sin_expanded.is_contiguous() else sin_expanded.contiguous()")
        add(lines, 8, "cos_in = cos_expanded if cos_expanded.is_contiguous() else cos_expanded.contiguous()")
        add(lines, 8, "out = torch.empty_like(x_in)")
        add(lines, 8, "n_elements = out.numel()")
        add(lines, 8, "if n_elements == 0:")
        add(lines, 12, "return out.reshape(x.shape)")
        add(lines, 8, "block = 1024")
        add(lines, 8, "grid = (triton.cdiv(n_elements, block),)")
        add(lines, 8, "_axon_triton_debug_count('rope_apply_factors')")
        add(
            lines,
            8,
            "_axon_triton_rope_apply_kernel[grid](x_in, sin_in, cos_in, out, n_elements, int(x_in.shape[-1]), BLOCK=block)",
        )
        add(lines, 8, "return out.reshape(x.shape)")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _rope_pair_apply_factors(cls, q, k, sin, cos, interleaved=False):")
        add(lines, 8, "if interleaved:")
        add(lines, 12, "raise NotImplementedError('__torch_rope_pair_apply_factors only supports non-interleaved RoPE')")
        add(lines, 8, "if torch.is_tensor(q):")
        add(lines, 12, "sin = cls._move_to(sin, q.device)")
        add(lines, 12, "cos = cls._move_to(cos, q.device)")
        add(lines, 8, "if triton is None or _axon_triton_rope_pair_apply_kernel is None or not torch.is_tensor(q) or not torch.is_tensor(k) or not q.is_cuda or not k.is_cuda:")
        add(lines, 12, "raise RuntimeError('__triton_rope_pair_apply_factors requires Triton and CUDA tensors')")
        add(lines, 8, "if q.shape[-1] != k.shape[-1]:")
        add(lines, 12, "raise ValueError('__triton_rope_pair_apply_factors requires q and k to have equal last dimensions')")
        add(lines, 8, "half = q.shape[-1] // 2")
        add(lines, 8, "if half * 2 != q.shape[-1]:")
        add(lines, 12, "raise ValueError('non-interleaved RoPE requires an even last dimension')")
        add(lines, 8, "try:")
        add(lines, 12, "sin_q = sin.expand_as(q)")
        add(lines, 12, "cos_q = cos.expand_as(q)")
        add(lines, 12, "sin_k = sin.expand_as(k)")
        add(lines, 12, "cos_k = cos.expand_as(k)")
        add(lines, 8, "except RuntimeError:")
        add(lines, 12, "raise ValueError('__triton_rope_pair_apply_factors requires sin/cos to expand to q and k')")
        add(lines, 8, "q_in = q if q.is_contiguous() else q.contiguous()")
        add(lines, 8, "k_in = k if k.is_contiguous() else k.contiguous()")
        add(lines, 8, "sin_q_in = sin_q if sin_q.is_contiguous() else sin_q.contiguous()")
        add(lines, 8, "cos_q_in = cos_q if cos_q.is_contiguous() else cos_q.contiguous()")
        add(lines, 8, "sin_k_in = sin_k if sin_k.is_contiguous() else sin_k.contiguous()")
        add(lines, 8, "cos_k_in = cos_k if cos_k.is_contiguous() else cos_k.contiguous()")
        add(lines, 8, "q_out = torch.empty_like(q_in)")
        add(lines, 8, "k_out = torch.empty_like(k_in)")
        add(lines, 8, "q_elements = q_out.numel()")
        add(lines, 8, "k_elements = k_out.numel()")
        add(lines, 8, "if q_elements == 0 and k_elements == 0:")
        add(lines, 12, "return q_out.reshape(q.shape), k_out.reshape(k.shape)")
        add(lines, 8, "block = 1024")
        add(lines, 8, "grid = (triton.cdiv(max(q_elements, k_elements), block),)")
        add(lines, 8, "_axon_triton_debug_count('rope_pair_apply_factors')")
        add(
            lines,
            8,
            "_axon_triton_rope_pair_apply_kernel[grid](q_in, k_in, sin_q_in, cos_q_in, sin_k_in, cos_k_in, q_out, k_out, q_elements, k_elements, int(q_in.shape[-1]), BLOCK=block)",
        )
        add(lines, 8, "return q_out.reshape(q.shape), k_out.reshape(k.shape)")
        add(lines, 4, "")
        add(lines, 4, "def _swiglu_ffn(self, x, gate_weight_path, up_weight_path, down_weight_path, gate_bias_path='bias', up_bias_path='bias', down_bias_path='bias'):")
        add(lines, 8, "gate, up = self._gate_up_linear_pair(x, gate_weight_path, up_weight_path, gate_bias_path=gate_bias_path, up_bias_path=up_bias_path, bias=False, transpose=False)")
        add(lines, 8, "hidden = self._swiglu_activation(gate, up)")
        add(lines, 8, "down_weight = self._param(down_weight_path)")
        add(lines, 8, "hidden = self._move_to(hidden, down_weight.device)")
        add(lines, 8, "return F.linear(hidden, down_weight, None)")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _swiglu_activation(cls, gate, up):")
        add(lines, 8, "if triton is None or _axon_triton_swiglu_kernel is None or not torch.is_tensor(gate) or not torch.is_tensor(up) or not gate.is_cuda or not up.is_cuda:")
        add(lines, 12, "raise RuntimeError('__triton_swiglu_activation requires Triton and CUDA tensors')")
        add(lines, 8, "if gate.shape != up.shape:")
        add(lines, 12, "raise ValueError('__triton_swiglu_activation requires equal gate/up shapes')")
        add(lines, 8, "gate_in = gate if gate.is_contiguous() else gate.contiguous()")
        add(lines, 8, "up_in = up if up.is_contiguous() else up.contiguous()")
        add(lines, 8, "out = torch.empty_like(gate_in)")
        add(lines, 8, "n_elements = out.numel()")
        add(lines, 8, "if n_elements == 0:")
        add(lines, 12, "return out.reshape(gate.shape)")
        add(lines, 8, "block = 1024")
        add(lines, 8, "grid = (triton.cdiv(n_elements, block),)")
        add(lines, 8, "_axon_triton_debug_count('swiglu_activation')")
        add(lines, 8, "_axon_triton_swiglu_kernel[grid](gate_in, up_in, out, n_elements, BLOCK=block)")
        add(lines, 8, "return out.reshape(gate.shape)")
        add(lines, 4, "")
        add(lines, 4, "@classmethod")
        add(lines, 4, "def _geglu_tanh_activation(cls, gate, up):")
        add(lines, 8, "if triton is None or _axon_triton_geglu_tanh_kernel is None or not torch.is_tensor(gate) or not torch.is_tensor(up) or not gate.is_cuda or not up.is_cuda:")
        add(lines, 12, "raise RuntimeError('__triton_geglu_tanh_activation requires Triton and CUDA tensors')")
        add(lines, 8, "if gate.shape != up.shape:")
        add(lines, 12, "raise ValueError('__triton_geglu_tanh_activation requires equal gate/up shapes')")
        add(lines, 8, "gate_in = gate if gate.is_contiguous() else gate.contiguous()")
        add(lines, 8, "up_in = up if up.is_contiguous() else up.contiguous()")
        add(lines, 8, "out = torch.empty_like(gate_in)")
        add(lines, 8, "n_elements = out.numel()")
        add(lines, 8, "if n_elements == 0:")
        add(lines, 12, "return out.reshape(gate.shape)")
        add(lines, 8, "block = 1024")
        add(lines, 8, "grid = (triton.cdiv(n_elements, block),)")
        add(lines, 8, "_axon_triton_debug_count('geglu_tanh_activation')")
        add(lines, 8, "_axon_triton_geglu_tanh_kernel[grid](gate_in, up_in, out, n_elements, BLOCK=block)")
        add(lines, 8, "return out.reshape(gate.shape)")
        add(lines, 4, "")
        add(lines, 4, "# Deprecated experimental path: real Qwen3-MoE-30B measurements showed this")
        add(lines, 4, "# Triton grouped matmul is much slower than the Torch grouped_mm path.")
        add(lines, 4, "# Keep it unreachable unless a future explicit experiment reworks the kernel.")
        add(lines, 4, "def _triton_expert_linear_weight(self, x, expert_idx, weight_path, bias_value=None, transpose=False):")
        add(lines, 8, "if triton is None or _axon_triton_grouped_mm_kernel is None:")
        add(lines, 12, "raise RuntimeError('__triton_selected_expert_packed_swiglu_ffn requires Triton grouped expert matmul')")
        add(lines, 8, "weight = self._param(weight_path)")
        add(lines, 8, "if not torch.is_tensor(x) or not torch.is_tensor(expert_idx) or not torch.is_tensor(weight) or not x.is_cuda or not expert_idx.is_cuda or not weight.is_cuda:")
        add(lines, 12, "raise RuntimeError('__triton_selected_expert_packed_swiglu_ffn requires CUDA tensors')")
        add(lines, 8, "if weight.ndim != 3:")
        add(lines, 12, "raise ValueError('__triton_selected_expert_packed_swiglu_ffn expert weight must have shape [E, O, I] or [E, I, O]')")
        add(lines, 8, "expert_idx = expert_idx.to(device=weight.device, dtype=torch.long)")
        add(lines, 8, "x = self._move_to(x, weight.device)")
        add(lines, 8, "out_dim = int(weight.shape[-1] if transpose else weight.shape[-2])")
        add(lines, 8, "out = x.new_empty((*x.shape[:-1], out_dim))")
        add(lines, 8, "if x.numel() == 0:")
        add(lines, 12, "return out")
        add(lines, 8, "flat_x = x.reshape(-1, x.shape[-1])")
        add(lines, 8, "flat_idx = expert_idx.reshape(-1)")
        add(lines, 8, "if flat_idx.numel() != flat_x.shape[0]:")
        add(lines, 12, "raise ValueError(f'expert_idx shape {tuple(expert_idx.shape)} is incompatible with input shape {tuple(x.shape)}')")
        add(lines, 8, "grouped_weight = weight if transpose else weight.transpose(-2, -1)")
        add(lines, 8, "grouped_weight = grouped_weight if grouped_weight.is_contiguous() else grouped_weight.contiguous()")
        add(lines, 8, "expert_ids_g, perm = torch.sort(flat_idx)")
        add(lines, 8, "x_g = flat_x.index_select(0, perm)")
        add(lines, 8, "x_run = x_g.to(dtype=grouped_weight.dtype) if x_g.is_floating_point() and grouped_weight.is_floating_point() and x_g.dtype != grouped_weight.dtype else x_g")
        add(lines, 8, "x_run = x_run if x_run.is_contiguous() else x_run.contiguous()")
        add(lines, 8, "tokens_per_expert = torch.histc(expert_ids_g.int(), bins=int(weight.shape[0]), min=0, max=int(weight.shape[0]) - 1)")
        add(lines, 8, "offsets = torch.cumsum(tokens_per_expert, dim=0, dtype=torch.int32)")
        add(lines, 8, "max_tokens = int(tokens_per_expert.max().item()) if tokens_per_expert.numel() else 0")
        add(lines, 8, "if max_tokens <= 0:")
        add(lines, 12, "return out")
        add(lines, 8, "y_g = x_run.new_empty((x_run.shape[0], out_dim))")
        add(lines, 8, "block_m = 16")
        add(lines, 8, "block_n = 32")
        add(lines, 8, "block_k = 32")
        add(lines, 8, "grid = (int(weight.shape[0]), triton.cdiv(max_tokens, block_m), triton.cdiv(out_dim, block_n))")
        add(lines, 8, "_axon_triton_debug_count('grouped_expert_linear')")
        add(lines, 8, "_axon_triton_grouped_mm_kernel[grid](x_run, grouped_weight, y_g, offsets, IN_DIM=int(x_run.shape[-1]), OUT_DIM=out_dim, BLOCK_M=block_m, BLOCK_N=block_n, BLOCK_K=block_k)")
        add(lines, 8, "y_g = y_g.to(dtype=x.dtype) if x.is_floating_point() and y_g.is_floating_point() and y_g.dtype != x.dtype else y_g")
        add(lines, 8, "if bias_value is not None:")
        add(lines, 12, "bias_g = self._move_to(bias_value, weight.device).index_select(0, expert_ids_g)")
        add(lines, 12, "bias_g = bias_g.to(dtype=x.dtype) if x.is_floating_point() and bias_g.is_floating_point() and bias_g.dtype != x.dtype else bias_g")
        add(lines, 12, "y_g = y_g + bias_g")
        add(lines, 8, "inv_perm = torch.empty_like(perm)")
        add(lines, 8, "inv_perm[perm] = torch.arange(perm.numel(), device=perm.device)")
        add(lines, 8, "return y_g.index_select(0, inv_perm).reshape_as(out)")
        add(lines, 4, "")
        add(lines, 4, "def _triton_selected_expert_packed_swiglu_ffn(self, x, topk_scores, topk_indices, gate_up_weight_path, down_weight_path, transpose=False):")
        add(lines, 8, "topk_indices = topk_indices.long()")
        add(lines, 8, "expanded = torch.unsqueeze(x, 2).expand((*topk_indices.shape, x.shape[-1]))")
        add(lines, 8, "gate_up = self._expert_linear_weight(expanded, topk_indices, gate_up_weight_path, transpose=transpose)")
        add(lines, 8, "gate, up = torch.chunk(gate_up, 2, dim=-1)")
        add(lines, 8, "hidden = self._swiglu_activation(gate, up)")
        add(lines, 8, "values = self._expert_linear_weight(hidden, topk_indices, down_weight_path, transpose=transpose)")
        add(lines, 8, "weights = torch.unsqueeze(topk_scores.to(device=values.device, dtype=values.dtype), -1)")
        add(lines, 8, "_axon_triton_debug_count('selected_expert_packed_swiglu_ffn')")
        add(lines, 8, "return torch.sum(values * weights, dim=2, keepdim=False)")

    def _primitive_expr(self, primitive: str, node: Any, *, local: set[str], symbols_dict: str) -> str:
        if primitive == "_triton_sdpa":
            return super()._primitive_expr("_torch_sdpa", node, local=local, symbols_dict=symbols_dict)
        if primitive == "_triton_rmsnorm_noscale":
            args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
            if len(args) < 4:
                raise ValueError("__triton_rmsnorm_noscale expects x, eps, dim, cast_float")
            return f"self._rmsnorm_noscale({args[0]}, {args[1]}, {args[2]}, {args[3]})"
        if primitive == "_triton_rmsnorm_scaled":
            args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
            if len(args) < 5:
                raise ValueError("__triton_rmsnorm_scaled expects x, scale, eps, dim, cast_float")
            return f"self._rmsnorm_scaled({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]})"
        if primitive == "_triton_rmsnorm_unit_offset_scaled":
            args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
            if len(args) < 5:
                raise ValueError("__triton_rmsnorm_unit_offset_scaled expects x, scale, eps, dim, cast_float")
            return f"self._rmsnorm_unit_offset_scaled({args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]})"
        if primitive == "_triton_swiglu_activation":
            args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
            if len(args) < 2:
                raise ValueError("__triton_swiglu_activation expects gate and up")
            return f"self._swiglu_activation({args[0]}, {args[1]})"
        if primitive == "_triton_geglu_tanh_activation":
            args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
            if len(args) < 2:
                raise ValueError("__triton_geglu_tanh_activation expects gate and up")
            return f"self._geglu_tanh_activation({args[0]}, {args[1]})"
        if primitive == "_triton_selected_expert_packed_swiglu_ffn":
            args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
            if len(args) < 6:
                raise ValueError("__triton_selected_expert_packed_swiglu_ffn expects input, top-k scores/indices, gate-up/down weight paths, and transpose")
            return (
                f"self._triton_selected_expert_packed_swiglu_ffn("
                f"{args[0]}, {args[1]}, {args[2]}, {args[3]}, {args[4]}, transpose=bool({args[5]}))"
            )
        if primitive == "_torch_rope_pair_apply_factors":
            args = [self._operand_expr(x, local=local, symbols_dict=symbols_dict) for x in node.inputs]
            if len(args) < 5:
                raise ValueError("__torch_rope_pair_apply_factors expects q, k, sin, cos, interleaved")
            interleaved = "False"
            if len(node.inputs) >= 5:
                value = getattr(node.inputs[4], "value", None)
                if isinstance(value, bool):
                    interleaved = "True" if value else "False"
                else:
                    interleaved = args[4]
            return f"self._rope_pair_apply_factors({args[0]}, {args[1]}, {args[2]}, {args[3]}, {interleaved})"
        return super()._primitive_expr(primitive, node, local=local, symbols_dict=symbols_dict)


def emit_model_code_from_graph_ir(
    program: GraphProgram,
    *,
    class_name: str = "GeneratedAxonModel",
    model_config: dict[str, Any] | None = None,
    profile: bool = False,
    align_devices: bool = False,
) -> str:
    """Emit direct Python/Torch model code from Graph IR with Triton hooks."""

    validate_graph_program(program)
    emitter = _DirectTritonEmitter(
        program=program,
        class_name=class_name,
        profile=profile,
        align_devices=align_devices,
    )
    body = emitter.emit()
    header = [
        "from __future__ import annotations",
        "",
    ]
    if profile:
        header.append("import time")
    header.extend(
        [
            "import atexit",
            "import json",
            "import os",
            "import torch",
            "from torch import nn",
            "from torch.nn import functional as F",
            "_AXON_TRITON_DEBUG_INTRINSICS = bool(int(os.environ.get('AXON_TRITON_DEBUG_INTRINSICS', '0')))",
            "_AXON_TRITON_DEBUG_COUNTS = {}",
            "def _axon_triton_debug_count(name):",
            "    if _AXON_TRITON_DEBUG_INTRINSICS:",
            "        _AXON_TRITON_DEBUG_COUNTS[name] = _AXON_TRITON_DEBUG_COUNTS.get(name, 0) + 1",
            "def _axon_triton_debug_report():",
            "    if _AXON_TRITON_DEBUG_INTRINSICS:",
            "        print('AXON_TRITON_DEBUG_COUNTS ' + json.dumps(_AXON_TRITON_DEBUG_COUNTS, sort_keys=True))",
            "atexit.register(_axon_triton_debug_report)",
            "try:",
            "    import triton",
            "    import triton.language as tl",
            "except Exception:",
            "    triton = None",
            "    tl = None",
            "if triton is not None:",
            "    @triton.jit",
            "    def _axon_triton_rmsnorm_noscale_kernel(x_ptr, out_ptr, rows, last_dim: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):",
            "        row = tl.program_id(0)",
            "        cols = tl.arange(0, BLOCK)",
            "        mask = cols < last_dim",
            "        offsets = row * last_dim + cols",
            "        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)",
            "        variance = tl.sum(x * x, axis=0) / last_dim",
            "        y = x * tl.rsqrt(variance + eps)",
            "        tl.store(out_ptr + offsets, y, mask=mask)",
            "    @triton.jit",
            "    def _axon_triton_rmsnorm_scaled_kernel(x_ptr, scale_ptr, out_ptr, rows, last_dim: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):",
            "        row = tl.program_id(0)",
            "        cols = tl.arange(0, BLOCK)",
            "        mask = cols < last_dim",
            "        offsets = row * last_dim + cols",
            "        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)",
            "        scale = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)",
            "        variance = tl.sum(x * x, axis=0) / last_dim",
            "        y = x * tl.rsqrt(variance + eps) * scale",
            "        tl.store(out_ptr + offsets, y, mask=mask)",
            "    @triton.jit",
            "    def _axon_triton_rmsnorm_unit_offset_scaled_kernel(x_ptr, scale_ptr, out_ptr, rows, last_dim: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):",
            "        row = tl.program_id(0)",
            "        cols = tl.arange(0, BLOCK)",
            "        mask = cols < last_dim",
            "        offsets = row * last_dim + cols",
            "        x = tl.load(x_ptr + offsets, mask=mask, other=0.0).to(tl.float32)",
            "        scale = tl.load(scale_ptr + cols, mask=mask, other=0.0).to(tl.float32)",
            "        variance = tl.sum(x * x, axis=0) / last_dim",
            "        y = x * tl.rsqrt(variance + eps) * (scale + 1.0)",
            "        tl.store(out_ptr + offsets, y, mask=mask)",
            "    @triton.jit",
            "    def _axon_triton_rope_apply_kernel(x_ptr, sin_ptr, cos_ptr, out_ptr, n_elements, rotary_dim: tl.constexpr, BLOCK: tl.constexpr):",
            "        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
            "        mask = offsets < n_elements",
            "        lane = offsets % rotary_dim",
            "        half = rotary_dim // 2",
            "        other_offsets = tl.where(lane < half, offsets + half, offsets - half)",
            "        sign = tl.where(lane < half, -1.0, 1.0)",
            "        x = tl.load(x_ptr + offsets, mask=mask, other=0.0)",
            "        rotated = tl.load(x_ptr + other_offsets, mask=mask, other=0.0)",
            "        sin = tl.load(sin_ptr + offsets, mask=mask, other=0.0)",
            "        cos = tl.load(cos_ptr + offsets, mask=mask, other=0.0)",
            "        tl.store(out_ptr + offsets, x * cos + rotated * sin * sign, mask=mask)",
            "    @triton.jit",
            "    def _axon_triton_rope_pair_apply_kernel(q_ptr, k_ptr, sin_q_ptr, cos_q_ptr, sin_k_ptr, cos_k_ptr, q_out_ptr, k_out_ptr, q_elements, k_elements, rotary_dim: tl.constexpr, BLOCK: tl.constexpr):",
            "        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
            "        lane = offsets % rotary_dim",
            "        half = rotary_dim // 2",
            "        other_offsets = tl.where(lane < half, offsets + half, offsets - half)",
            "        sign = tl.where(lane < half, -1.0, 1.0)",
            "        q_mask = offsets < q_elements",
            "        q = tl.load(q_ptr + offsets, mask=q_mask, other=0.0)",
            "        q_rotated = tl.load(q_ptr + other_offsets, mask=q_mask, other=0.0)",
            "        q_sin = tl.load(sin_q_ptr + offsets, mask=q_mask, other=0.0)",
            "        q_cos = tl.load(cos_q_ptr + offsets, mask=q_mask, other=0.0)",
            "        tl.store(q_out_ptr + offsets, q * q_cos + q_rotated * q_sin * sign, mask=q_mask)",
            "        k_mask = offsets < k_elements",
            "        k = tl.load(k_ptr + offsets, mask=k_mask, other=0.0)",
            "        k_rotated = tl.load(k_ptr + other_offsets, mask=k_mask, other=0.0)",
            "        k_sin = tl.load(sin_k_ptr + offsets, mask=k_mask, other=0.0)",
            "        k_cos = tl.load(cos_k_ptr + offsets, mask=k_mask, other=0.0)",
            "        tl.store(k_out_ptr + offsets, k * k_cos + k_rotated * k_sin * sign, mask=k_mask)",
            "    @triton.jit",
            "    def _axon_triton_swiglu_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):",
            "        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
            "        mask = offsets < n_elements",
            "        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)",
            "        up = tl.load(up_ptr + offsets, mask=mask, other=0.0)",
            "        tl.store(out_ptr + offsets, gate * tl.sigmoid(gate) * up, mask=mask)",
            "    @triton.jit",
            "    def _axon_triton_geglu_tanh_kernel(gate_ptr, up_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):",
            "        offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)",
            "        mask = offsets < n_elements",
            "        gate = tl.load(gate_ptr + offsets, mask=mask, other=0.0).to(tl.float32)",
            "        up = tl.load(up_ptr + offsets, mask=mask, other=0.0)",
            "        inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)",
            "        tanh_inner = 2.0 * tl.sigmoid(2.0 * inner) - 1.0",
            "        activated = 0.5 * gate * (1.0 + tanh_inner)",
            "        tl.store(out_ptr + offsets, activated * up, mask=mask)",
            "    @triton.jit",
            "    def _axon_triton_grouped_mm_kernel(x_ptr, w_ptr, y_ptr, offsets_ptr, IN_DIM: tl.constexpr, OUT_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):",
            "        expert = tl.program_id(0)",
            "        m_block = tl.program_id(1)",
            "        n_block = tl.program_id(2)",
            "        start = tl.load(offsets_ptr + expert - 1, mask=expert > 0, other=0)",
            "        end = tl.load(offsets_ptr + expert)",
            "        count = end - start",
            "        rows = m_block * BLOCK_M + tl.arange(0, BLOCK_M)",
            "        cols = n_block * BLOCK_N + tl.arange(0, BLOCK_N)",
            "        offs_k = tl.arange(0, BLOCK_K)",
            "        acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)",
            "        for k0 in range(0, IN_DIM, BLOCK_K):",
            "            k = k0 + offs_k",
            "            x = tl.load(x_ptr + (start + rows)[:, None] * IN_DIM + k[None, :], mask=(rows[:, None] < count) & (k[None, :] < IN_DIM), other=0.0)",
            "            w = tl.load(w_ptr + expert * IN_DIM * OUT_DIM + k[:, None] * OUT_DIM + cols[None, :], mask=(k[:, None] < IN_DIM) & (cols[None, :] < OUT_DIM), other=0.0)",
            "            acc += tl.dot(x, w)",
            "        tl.store(y_ptr + (start + rows)[:, None] * OUT_DIM + cols[None, :], acc, mask=(rows[:, None] < count) & (cols[None, :] < OUT_DIM))",
            "else:",
            "    _axon_triton_rmsnorm_noscale_kernel = None",
            "    _axon_triton_rmsnorm_scaled_kernel = None",
            "    _axon_triton_rmsnorm_unit_offset_scaled_kernel = None",
            "    _axon_triton_rope_apply_kernel = None",
            "    _axon_triton_rope_pair_apply_kernel = None",
            "    _axon_triton_swiglu_kernel = None",
            "    _axon_triton_geglu_tanh_kernel = None",
            "    _axon_triton_grouped_mm_kernel = None",
            "from brainsurgery.synapse.axon.codegen2_torch.core import _materialize_joined_parameter, _materialize_packed_parameters",
            "from brainsurgery.synapse.axon.codegen2_common import (",
            "    compose_path as _common_compose_path,",
            "    config_value as _common_config_value,",
            "    has_config_value as _common_has_config_value,",
            "    optional_state_value as _common_optional_state_value,",
            "    render_path as _common_render_path,",
            "    required_state_value as _common_required_state_value,",
            "    require_value as _common_require_value,",
            ")",
            "",
            f"_MODEL_CONFIG = {model_config!r}",
            "",
            body,
        ]
    )
    return "\n".join(header)


__all__ = [
    "emit_model_code_from_graph_ir",
    "triton_op_table_markdown",
]
