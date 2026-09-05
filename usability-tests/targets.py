"""Facts about the surgery-target models that drive the whole kit.

Every task specification, reference solution, review artifact, input builder
and grader rule is rendered from this file, so the three targets stay
consistent. Names inside a layer are relative to the layer prefix
(``layer_fmt``); ``layer_re`` is the regex with the layer index as group 1.

Head layout vocabulary (used by T2): for each head-bearing tensor,
``dim`` is the axis along which heads live, ``segments`` is how many head
groups are concatenated along that axis (3 for a fused [q | k | v]
projection, 1 otherwise), ``seg_size`` is the width of one segment and
``block`` is the width of one head inside a segment. Head ``h`` of segment
``s`` occupies indices ``s*seg_size + h*block .. + block - 1``.
"""

from __future__ import annotations

TARGETS: dict[str, dict] = {
    "gpt-2": {
        "display": "GPT-2 (124M)",
        "hf_id": "openai-community/gpt2",
        "model_dir": "models/gpt2",
        "base_layout": "single",  # inputs/base/model.safetensors
        "base_dtype": "float32",
        "n_layers": 12,
        "hidden": 768,
        "n_heads": 12,
        "head_dim": 64,
        "layer_fmt": "h.{i}.",
        "layer_re": r"h\.(\d+)\.",
        "layer_glob": r"h\.{i}\.",
        "tensors_per_layer": 13,
        "total_tensors": 160,
        "nonlayer": [
            ("wte.weight", "[50257, 768]"), ("wpe.weight", "[1024, 768]"),
            ("ln_f.weight", "[768]"), ("ln_f.bias", "[768]"),
        ],
        "layer_tensors_note": (
            "layer norms `ln_1`/`ln_2` (weight and bias), the fused attention projection "
            "`attn.c_attn` (weight `[768, 2304]`, bias `[2304]`), `attn.c_proj` (weight "
            "`[768, 768]`, bias `[768]`), the causal-mask buffer `attn.bias` (`[1, 1, 1024, 1024]`), "
            "and the MLP `mlp.c_fc` (weight `[768, 3072]`, bias `[3072]`) and `mlp.c_proj` "
            "(weight `[3072, 768]`, bias `[768]`)"
        ),
        "layout_note": (
            "GPT-2 stores projection matrices as `[in, out]` (Conv1D layout), the transpose "
            "of `nn.Linear`."
        ),
        "big_tensor": ("wte.weight", "154 MB"),
        # T1
        "drop_layers": [2, 5, 8],
        # T2
        "drop_head": 5,
        "head_tensors": [
            {"name": "attn.c_attn.weight", "dim": 1, "segments": 3, "seg_size": 768, "block": 64,
             "shape": "[768, 2304]", "role": "fused `[q | k | v]` projection; heads are column blocks inside each 768-wide segment"},
            {"name": "attn.c_attn.bias", "dim": 0, "segments": 3, "seg_size": 768, "block": 64,
             "shape": "[2304]", "role": "same layout as the columns of `attn.c_attn.weight`"},
            {"name": "attn.c_proj.weight", "dim": 0, "segments": 1, "seg_size": 768, "block": 64,
             "shape": "[768, 768]", "role": "output projection; heads are row blocks"},
        ],
        "head_untouched": ["attn.c_proj.bias (`[768]`)", "attn.bias (`[1, 1, 1024, 1024]`, the mask buffer)"],
        # T3
        "proj_matrices": [
            ("attn.c_attn.weight", "[768, 2304]"), ("attn.c_proj.weight", "[768, 768]"),
            ("mlp.c_fc.weight", "[768, 3072]"), ("mlp.c_proj.weight", "[3072, 768]"),
        ],
        "buffers": [("attn.bias", "[1, 1, 1024, 1024]", "the causal-mask buffer")],
        "keep_note": "`wte.weight`, `wpe.weight`, all layer-norm weights and biases, and all projection biases",
        "shard_t3": ("64MB", 64 * 1024 * 1024, "64 MiB (67,108,864 bytes)"),
        "shard_t5": ("100MB", 100 * 1024 * 1024, "100 MiB (104,857,600 bytes)"),
        # T4
        "mlp_tensors": [
            ("mlp.c_fc.weight", "[768, 3072]"), ("mlp.c_fc.bias", "[3072]"),
            ("mlp.c_proj.weight", "[3072, 768]"), ("mlp.c_proj.bias", "[768]"),
        ],
        "mlp_weights": ["mlp.c_fc.weight", "mlp.c_proj.weight"],
        # T5
        "lora": {
            "modules": ["attn.c_attn"], "r": 16, "alpha": 32, "fan_in_fan_out": True,
            "in": 768, "out": 2304, "weight_shape": "[768, 2304]",
            "peft_prefix": "base_model.model.",
        },
    },
    "olmo-1b": {
        "display": "OLMo-1B-0724-hf",
        "hf_id": "allenai/OLMo-1B-0724-hf",
        "model_dir": "models/olmo-1b-0724-hf",
        "base_layout": "sharded",  # inputs/base/ holds two shards and an index
        "base_dtype": "float32",
        "n_layers": 16,
        "hidden": 2048,
        "n_heads": 16,
        "head_dim": 128,
        "layer_fmt": "model.layers.{i}.",
        "layer_re": r"model\.layers\.(\d+)\.",
        "layer_glob": r"model\.layers\.{i}\.",
        "tensors_per_layer": 7,
        "total_tensors": 114,
        "nonlayer": [("model.embed_tokens.weight", "[50304, 2048]"), ("lm_head.weight", "[50304, 2048]")],
        "layer_tensors_note": (
            "separate attention projections `self_attn.q_proj`, `self_attn.k_proj`, "
            "`self_attn.v_proj`, `self_attn.o_proj` (all `[2048, 2048]`) and the MLP "
            "`mlp.gate_proj` (`[8192, 2048]`), `mlp.up_proj` (`[8192, 2048]`), `mlp.down_proj` "
            "(`[2048, 8192]`). There are no biases and no layer-norm parameters (the model "
            "uses non-parametric layer norm)"
        ),
        "layout_note": "All projection matrices use the `nn.Linear` layout `[out, in]`.",
        "big_tensor": ("model.embed_tokens.weight` and `lm_head.weight", "412 MB each"),
        "drop_layers": [2, 6, 10, 14],
        "drop_head": 5,
        "head_tensors": [
            {"name": "self_attn.q_proj.weight", "dim": 0, "segments": 1, "seg_size": 2048, "block": 128,
             "shape": "[2048, 2048]", "role": "query projection; heads are row blocks"},
            {"name": "self_attn.k_proj.weight", "dim": 0, "segments": 1, "seg_size": 2048, "block": 128,
             "shape": "[2048, 2048]", "role": "key projection; heads are row blocks"},
            {"name": "self_attn.v_proj.weight", "dim": 0, "segments": 1, "seg_size": 2048, "block": 128,
             "shape": "[2048, 2048]", "role": "value projection; heads are row blocks"},
            {"name": "self_attn.o_proj.weight", "dim": 1, "segments": 1, "seg_size": 2048, "block": 128,
             "shape": "[2048, 2048]", "role": "output projection; heads are column blocks"},
        ],
        "head_untouched": ["the three MLP matrices"],
        "proj_matrices": [
            ("self_attn.q_proj.weight", "[2048, 2048]"), ("self_attn.k_proj.weight", "[2048, 2048]"),
            ("self_attn.v_proj.weight", "[2048, 2048]"), ("self_attn.o_proj.weight", "[2048, 2048]"),
            ("mlp.gate_proj.weight", "[8192, 2048]"), ("mlp.up_proj.weight", "[8192, 2048]"),
            ("mlp.down_proj.weight", "[2048, 8192]"),
        ],
        "buffers": [],
        "keep_note": "`model.embed_tokens.weight` and `lm_head.weight` (there are no norms or biases)",
        "shard_t3": ("256MB", 256 * 1024 * 1024, "256 MiB (268,435,456 bytes)"),
        "shard_t5": ("512MB", 512 * 1024 * 1024, "512 MiB (536,870,912 bytes)"),
        "mlp_tensors": [
            ("mlp.gate_proj.weight", "[8192, 2048]"), ("mlp.up_proj.weight", "[8192, 2048]"),
            ("mlp.down_proj.weight", "[2048, 8192]"),
        ],
        "mlp_weights": ["mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"],
        "lora": {
            "modules": ["self_attn.q_proj", "self_attn.v_proj"], "r": 16, "alpha": 32, "fan_in_fan_out": False,
            "in": 2048, "out": 2048, "weight_shape": "[2048, 2048]",
            "peft_prefix": "base_model.model.",
        },
    },
    "pythia-1b": {
        "display": "Pythia-1B",
        "hf_id": "EleutherAI/pythia-1b",
        "model_dir": "models/pythia-1b",
        "base_layout": "single",
        "base_dtype": "float16",
        "n_layers": 16,
        "hidden": 2048,
        "n_heads": 8,
        "head_dim": 256,
        "layer_fmt": "gpt_neox.layers.{i}.",
        "layer_re": r"gpt_neox\.layers\.(\d+)\.",
        "layer_glob": r"gpt_neox\.layers\.{i}\.",
        "tensors_per_layer": 15,
        "total_tensors": 244,
        "nonlayer": [
            ("gpt_neox.embed_in.weight", "[50304, 2048]"), ("embed_out.weight", "[50304, 2048]"),
            ("gpt_neox.final_layer_norm.weight", "[2048]"), ("gpt_neox.final_layer_norm.bias", "[2048]"),
        ],
        "layer_tensors_note": (
            "layer norms `input_layernorm`/`post_attention_layernorm` (weight and bias), the fused "
            "attention projection `attention.query_key_value` (weight `[6144, 2048]`, bias `[6144]`), "
            "`attention.dense` (weight `[2048, 2048]`, bias `[2048]`), the MLP `mlp.dense_h_to_4h` "
            "(weight `[8192, 2048]`, bias `[8192]`) and `mlp.dense_4h_to_h` (weight `[2048, 8192]`, "
            "bias `[2048]`), and three non-parameter buffers: `attention.bias` (`[1, 1, 2048, 2048]`, "
            "uint8 causal mask), `attention.masked_bias` (scalar) and `attention.rotary_emb.inv_freq` (`[32]`)"
        ),
        "layout_note": (
            "All projection matrices use the `nn.Linear` layout `[out, in]`. The checkpoint is "
            "stored in float16. In the fused `query_key_value` projection the 6144 rows are "
            "ordered per head: head `h` owns rows `768*h .. 768*h+767`, and inside that block the "
            "first 256 rows are its query, the next 256 its key and the last 256 its value "
            "(GPT-NeoX interleaved layout, not `[q | k | v]` segments)."
        ),
        "big_tensor": ("gpt_neox.embed_in.weight` and `embed_out.weight", "206 MB each"),
        "drop_layers": [2, 6, 10, 14],
        "drop_head": 5,
        "head_tensors": [
            {"name": "attention.query_key_value.weight", "dim": 0, "segments": 1, "seg_size": 6144, "block": 768,
             "shape": "[6144, 2048]", "role": "fused projection, interleaved per head; a head is one 768-row block holding its q, k and v"},
            {"name": "attention.query_key_value.bias", "dim": 0, "segments": 1, "seg_size": 6144, "block": 768,
             "shape": "[6144]", "role": "same layout as the rows of `attention.query_key_value.weight`"},
            {"name": "attention.dense.weight", "dim": 1, "segments": 1, "seg_size": 2048, "block": 256,
             "shape": "[2048, 2048]", "role": "output projection; heads are 256-wide column blocks"},
        ],
        "head_untouched": ["attention.dense.bias (`[2048]`)", "the three attention buffers", "the MLP tensors"],
        "proj_matrices": [
            ("attention.query_key_value.weight", "[6144, 2048]"), ("attention.dense.weight", "[2048, 2048]"),
            ("mlp.dense_h_to_4h.weight", "[8192, 2048]"), ("mlp.dense_4h_to_h.weight", "[2048, 8192]"),
        ],
        "buffers": [
            ("attention.bias", "[1, 1, 2048, 2048]", "uint8 causal-mask buffer"),
            ("attention.masked_bias", "[]", "scalar buffer"),
            ("attention.rotary_emb.inv_freq", "[32]", "rotary frequency buffer"),
        ],
        "keep_note": "`gpt_neox.embed_in.weight`, `embed_out.weight`, all layer-norm weights and biases, and all projection biases (these are float16 in the input and must be upcast to float32)",
        "shard_t3": ("256MB", 256 * 1024 * 1024, "256 MiB (268,435,456 bytes)"),
        "shard_t5": ("512MB", 512 * 1024 * 1024, "512 MiB (536,870,912 bytes)"),
        "mlp_tensors": [
            ("mlp.dense_h_to_4h.weight", "[8192, 2048]"), ("mlp.dense_h_to_4h.bias", "[8192]"),
            ("mlp.dense_4h_to_h.weight", "[2048, 8192]"), ("mlp.dense_4h_to_h.bias", "[2048]"),
        ],
        "mlp_weights": ["mlp.dense_h_to_4h.weight", "mlp.dense_4h_to_h.weight"],
        "lora": {
            "modules": ["attention.query_key_value"], "r": 16, "alpha": 32, "fan_in_fan_out": False,
            "in": 2048, "out": 6144, "weight_shape": "[6144, 2048]",
            "peft_prefix": "base_model.model.",
        },
    },
}

TESTS = {
    "T1": "layer-prune",
    "T2": "head-prune",
    "T3": "mixed-precision-export",
    "T4": "task-vector-merge",
    "T5": "lora-merge",
}


def layer_name(target: dict, i: int | str, rel: str) -> str:
    return target["layer_fmt"].format(i=i) + rel


def kept_layers(target: dict) -> list[int]:
    return [i for i in range(target["n_layers"]) if i not in target["drop_layers"]]


def head_keep_slices(spec: dict, drop_head: int) -> list[tuple[int, int]]:
    """Contiguous [start, stop) index ranges that survive when ``drop_head`` is removed."""
    out = []
    for seg in range(spec["segments"]):
        base = seg * spec["seg_size"]
        lo = base + drop_head * spec["block"]
        hi = lo + spec["block"]
        if lo > base:
            out.append((base, lo))
        if hi < base + spec["seg_size"]:
            out.append((hi, base + spec["seg_size"]))
    return out


def pruned_size(spec: dict) -> int:
    return spec["segments"] * (spec["seg_size"] - spec["block"])


def esc(name: str) -> str:
    """Regex-escape a dotted tensor name for BrainSurgery references."""
    return name.replace(".", r"\.")
