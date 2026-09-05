from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file as load_safetensors_file

if TYPE_CHECKING:
	from transformers import GPT2LMHeadModel


DEFAULT_PROMPT_FILE = Path("validation/prompts.txt")
DEFAULT_OUTPUT_JSON = Path("validation/model_outputs.json")
DEFAULT_GPT2_BASE_MODEL = "models/test/gpt2"
DEFAULT_GPT2_VALIDATION_MODEL = "models/test/validation"

MODEL_PRESETS: dict[str, dict[str, str]] = {
	"gpt2-validation": {
		"base_model": DEFAULT_GPT2_BASE_MODEL,
		"test_model": DEFAULT_GPT2_VALIDATION_MODEL,
		"model_loader": "gpt2-brainsurgery",
		"tokenizer_source": DEFAULT_GPT2_BASE_MODEL,
		"tokenizer_loader": "auto",
	},
	"ministral3-3b-base-2512": {
		"base_model": "models/test/ministral3-3b-base-2512",
		"test_model": "models/test/ministral3_3b_base_2512_validation",
		"config_source": "models/test/ministral3-3b-base-2512",
		"model_loader": "hf-mistral3-conditional-generation",
		"tokenizer_source": "models/test/ministral3-3b-base-2512",
		"tokenizer_loader": "auto",
	},
	"qwen3-1.7b-base": {
		"base_model": "models/test/qwen3-1.7b-base",
		"test_model": "models/test/qwen3_1_7b_base_validation",
		"config_source": "models/test/qwen3-1.7b-base",
		"model_loader": "hf-causal-lm",
		"tokenizer_source": "models/test/qwen3-1.7b-base",
		"tokenizer_loader": "auto",
	},
	"apertus-v1.1-1.5b": {
		"base_model": "models/test/apertus-v1.1-1.5b",
		"test_model": "models/test/apertus_v1_1_1_5b_validation",
		"config_source": "models/test/apertus-v1.1-1.5b",
		"model_loader": "hf-causal-lm",
		"tokenizer_source": "models/test/apertus-v1.1-1.5b",
		"tokenizer_loader": "auto",
	},
}


def _require_transformers():
	try:
		from transformers import (
			AutoConfig,
			AutoModelForCausalLM,
			AutoTokenizer,
			GPT2Config,
			GPT2LMHeadModel,
		)
	except ModuleNotFoundError as exc:
		raise RuntimeError(
			"Missing dependency 'transformers'. Install it with: "
			"conda run -n brainsurgery pip install transformers"
		) from exc
	return AutoConfig, AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel


def _torch_dtype(dtype_name: str) -> torch.dtype | str:
	if dtype_name == "auto":
		return "auto"
	if dtype_name == "float32":
		return torch.float32
	if dtype_name == "float16":
		return torch.float16
	if dtype_name == "bfloat16":
		return torch.bfloat16
	raise ValueError(f"unsupported dtype: {dtype_name}")


def _effective_dtype_name(dtype_name: str, device: torch.device) -> str:
	if device.type == "mps" and dtype_name in {"auto", "bfloat16"}:
		return "float32"
	return dtype_name


def _choose_device(requested: str | None = None) -> torch.device:
	if requested:
		return torch.device(requested)
	if torch.cuda.is_available():
		return torch.device("cuda")
	if torch.backends.mps.is_available():
		return torch.device("mps")
	return torch.device("cpu")


def _resolve_shards(model_dir: Path) -> list[Path]:
	index_path = model_dir / "model.safetensors.index.json"
	if index_path.exists():
		payload = json.loads(index_path.read_text(encoding="utf-8"))
		weight_map = payload.get("weight_map")
		if not isinstance(weight_map, dict):
			raise RuntimeError(f"invalid safetensors index: {index_path}")
		shard_names = sorted(set(weight_map.values()))
		return [model_dir / str(name) for name in shard_names]

	shards = sorted(model_dir.glob("*.safetensors"))
	if not shards:
		raise FileNotFoundError(f"no .safetensors files found in {model_dir}")
	return shards


def _load_sharded_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
	merged: dict[str, torch.Tensor] = {}
	for shard_path in _resolve_shards(model_dir):
		if not shard_path.exists():
			raise FileNotFoundError(f"missing shard referenced by index: {shard_path}")
		shard = load_safetensors_file(str(shard_path), device="cpu")
		overlap = set(merged).intersection(shard)
		if overlap:
			raise RuntimeError(f"duplicate keys across shards: {sorted(overlap)[:5]}")
		merged.update(shard)
	return merged


def _normalize_gpt2_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
	"""Map brainsurgery GPT-2 keys to Hugging Face GPT-2 key layout."""
	normalized: dict[str, torch.Tensor] = {}
	for key, value in state_dict.items():
		if key.startswith("transformer."):
			normalized[key] = value
		elif key.startswith("h.") or key.startswith("wte.") or key.startswith("wpe.") or key.startswith("ln_f."):
			normalized[f"transformer.{key}"] = value
		else:
			normalized[key] = value
	return normalized


def _load_tokenizer(
	tokenizer_source: str | Path,
	*,
	local_files_only: bool | None = None,
	trust_remote_code: bool = True,
	fallback_to_gpt2: bool = False,
):
	_, _, AutoTokenizer, _, _ = _require_transformers()
	kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
	if local_files_only is not None:
		kwargs["local_files_only"] = local_files_only
	try:
		tok = AutoTokenizer.from_pretrained(str(tokenizer_source), **kwargs)
	except Exception:
		if not fallback_to_gpt2:
			raise
		try:
			tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=True)
		except Exception:
			tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=False)
	if getattr(tok, "pad_token", None) is None and getattr(tok, "eos_token", None) is not None:
		tok.pad_token = tok.eos_token
	return tok


def load_gpt2_brainsurgery(model_dir: str | Path, device: torch.device) -> "GPT2LMHeadModel":
	_, _, _, GPT2Config, GPT2LMHeadModel = _require_transformers()
	state_dict = _normalize_gpt2_state_dict_keys(_load_sharded_state_dict(Path(model_dir)))
	model = GPT2LMHeadModel(GPT2Config())
	missing, unexpected = model.load_state_dict(state_dict, strict=False)

	allowed_missing = {"lm_head.weight"}
	allowed_unexpected_prefixes = ("transformer.h.",)
	remaining_unexpected = [
		key
		for key in unexpected
		if not (key.startswith(allowed_unexpected_prefixes) and key.endswith(".attn.bias"))
	]
	if remaining_unexpected or (set(missing) - allowed_missing):
		raise RuntimeError(
			"state_dict load mismatch: "
			f"missing={missing}, unexpected={unexpected}"
		)

	model.tie_weights()
	model.to(device)
	model.eval()
	return model


def load_hf_causal_lm(
	model_source: str | Path,
	device: torch.device,
	*,
	dtype_name: str = "auto",
	config_source: str | Path | None = None,
	trust_remote_code: bool = True,
	local_files_only: bool = False,
) -> torch.nn.Module:
	AutoConfig, AutoModelForCausalLM, _, _, _ = _require_transformers()
	effective_dtype_name = _effective_dtype_name(dtype_name, device)
	config = None
	if config_source is not None:
		config = AutoConfig.from_pretrained(
			str(config_source),
			trust_remote_code=trust_remote_code,
			local_files_only=local_files_only,
		)
	model = AutoModelForCausalLM.from_pretrained(
		str(model_source),
		dtype=_torch_dtype(effective_dtype_name),
		config=config,
		trust_remote_code=trust_remote_code,
		local_files_only=local_files_only,
	)
	model.to(device)
	model.eval()
	return model


def load_hf_mistral3_conditional_generation(
	model_source: str | Path,
	device: torch.device,
	*,
	dtype_name: str = "auto",
	config_source: str | Path | None = None,
	trust_remote_code: bool = True,
	local_files_only: bool = False,
) -> torch.nn.Module:
	AutoConfig, _, _, _, _ = _require_transformers()
	effective_dtype_name = _effective_dtype_name(dtype_name, device)
	try:
		from transformers import Mistral3ForConditionalGeneration
	except ImportError as exc:
		raise RuntimeError(
			"Missing transformers support for Mistral3ForConditionalGeneration. "
			"Upgrade transformers in the brainsurgery environment."
		) from exc
	config = None
	if config_source is not None:
		config = AutoConfig.from_pretrained(
			str(config_source),
			trust_remote_code=trust_remote_code,
			local_files_only=local_files_only,
		)
	model = Mistral3ForConditionalGeneration.from_pretrained(
		str(model_source),
		dtype=_torch_dtype(effective_dtype_name),
		config=config,
		trust_remote_code=trust_remote_code,
		local_files_only=local_files_only,
	)
	model.to(device)
	model.eval()
	return model


def load_model(
	model_source: str | Path,
	device: torch.device,
	*,
	model_loader: str,
	dtype_name: str = "auto",
	config_source: str | Path | None = None,
	trust_remote_code: bool = True,
	local_files_only: bool = False,
) -> torch.nn.Module:
	if model_loader == "gpt2-brainsurgery":
		return load_gpt2_brainsurgery(model_source, device)
	if model_loader == "hf-causal-lm":
		return load_hf_causal_lm(
			model_source,
			device,
			dtype_name=dtype_name,
			config_source=config_source,
			trust_remote_code=trust_remote_code,
			local_files_only=local_files_only,
		)
	if model_loader == "hf-mistral3-conditional-generation":
		return load_hf_mistral3_conditional_generation(
			model_source,
			device,
			dtype_name=dtype_name,
			config_source=config_source,
			trust_remote_code=trust_remote_code,
			local_files_only=local_files_only,
		)
	raise ValueError(f"unsupported model loader: {model_loader}")


def _last_token_logits(
	model: torch.nn.Module,
	input_ids: torch.Tensor,
	attention_mask: torch.Tensor,
) -> torch.Tensor:
	with torch.inference_mode():
		logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
	last_index = int(attention_mask[0].sum().item()) - 1
	return logits[0, last_index, :].detach().float().cpu()


def _full_sequence_logit_metrics(
	base_model: torch.nn.Module,
	test_model: torch.nn.Module,
	input_ids: torch.Tensor,
	attention_mask: torch.Tensor,
) -> dict[str, float | int]:
	with torch.inference_mode():
		base_logits = base_model(input_ids=input_ids, attention_mask=attention_mask).logits.detach().float()
		test_logits = test_model(input_ids=input_ids, attention_mask=attention_mask).logits.detach().float()

	if base_logits.shape != test_logits.shape:
		raise RuntimeError(f"logit shape mismatch: {base_logits.shape} != {test_logits.shape}")

	# Causal-LM logits at position t predict token t+1; compare positions with a next token.
	base_compared = base_logits[:, :-1, :] if base_logits.shape[1] > 1 else base_logits
	test_compared = test_logits[:, :-1, :] if test_logits.shape[1] > 1 else test_logits
	base_compared_cpu = base_compared.cpu()
	test_compared_cpu = test_compared.cpu()
	per_position_cosines = F.cosine_similarity(
		base_compared_cpu[0].double(),
		test_compared_cpu[0].double(),
		dim=-1,
	).clamp(-1.0, 1.0)
	abs_diff = torch.abs(base_compared_cpu - test_compared_cpu)
	return {
		"full_sequence_positions_compared": int(per_position_cosines.numel()),
		"full_sequence_mean_logit_cosine": float(per_position_cosines.mean().item()),
		"full_sequence_min_logit_cosine": float(per_position_cosines.min().item()),
		"full_sequence_mean_abs_logit_diff": float(abs_diff.mean().item()),
		"full_sequence_max_abs_logit_diff": float(abs_diff.max().item()),
	}


def _encode_prompt(tokenizer, prompt: str, device: torch.device) -> dict[str, torch.Tensor]:
	encoded = tokenizer(prompt, return_tensors="pt")
	if hasattr(encoded, "to"):
		encoded = encoded.to(device)
	else:
		encoded = {key: value.to(device) for key, value in encoded.items()}
	if "attention_mask" not in encoded:
		encoded["attention_mask"] = torch.ones_like(encoded["input_ids"], device=device)
	return dict(encoded)


def _pad_token_id(tokenizer) -> int | None:
	pad_token_id = getattr(tokenizer, "pad_token_id", None)
	if pad_token_id is not None:
		return int(pad_token_id)
	eos_token_id = getattr(tokenizer, "eos_token_id", None)
	if eos_token_id is not None:
		return int(eos_token_id)
	return None


def _generate_text(
	model: torch.nn.Module,
	tokenizer,
	prompt: str,
	max_new_tokens: int,
	device: torch.device,
) -> str:
	encoded = _encode_prompt(tokenizer, prompt, device)
	with torch.inference_mode():
		generated = model.generate(
			**encoded,
			max_new_tokens=max_new_tokens,
			do_sample=False,
			pad_token_id=_pad_token_id(tokenizer),
		)
	return tokenizer.decode(generated[0], skip_special_tokens=True)


def _generate_ids(
	model: torch.nn.Module,
	tokenizer,
	prompt: str,
	max_new_tokens: int,
	device: torch.device,
) -> torch.Tensor:
	encoded = _encode_prompt(tokenizer, prompt, device)
	with torch.inference_mode():
		return model.generate(
			**encoded,
			max_new_tokens=max_new_tokens,
			do_sample=False,
			pad_token_id=_pad_token_id(tokenizer),
		)


def _read_prompts(prompt_args: list[str], prompt_file: Path) -> list[str]:
	if prompt_args:
		return prompt_args
	lines = [line.strip() for line in prompt_file.read_text(encoding="utf-8").splitlines()]
	prompts = [line for line in lines if line and not line.startswith("#")]
	if not prompts:
		raise RuntimeError(f"prompt file has no non-empty prompts: {prompt_file}")
	return prompts


def _tokenizer_label(tokenizer) -> str:
	name_or_path = getattr(tokenizer, "name_or_path", None)
	if name_or_path:
		return str(name_or_path)
	return tokenizer.__class__.__name__


def _token_bag_cosine(left_tokens: list[int], right_tokens: list[int]) -> float:
	if not left_tokens and not right_tokens:
		return 1.0
	if not left_tokens or not right_tokens:
		return 0.0

	left_counts = Counter(left_tokens)
	right_counts = Counter(right_tokens)
	dot = sum(count * right_counts.get(token_id, 0) for token_id, count in left_counts.items())
	left_norm = math.sqrt(sum(count * count for count in left_counts.values()))
	right_norm = math.sqrt(sum(count * count for count in right_counts.values()))
	if left_norm == 0.0 or right_norm == 0.0:
		return 0.0
	return dot / (left_norm * right_norm)


def _output_similarity_metrics(tokenizer, before: str, after: str) -> dict[str, float | int | bool]:
	before_tokens = tokenizer.encode(before, add_special_tokens=False)
	after_tokens = tokenizer.encode(after, add_special_tokens=False)
	return {
		"output_exact_match": before == after,
		"output_char_similarity": SequenceMatcher(a=before, b=after).ratio(),
		"output_token_sequence_similarity": SequenceMatcher(a=before_tokens, b=after_tokens).ratio(),
		"output_token_bag_cosine": _token_bag_cosine(before_tokens, after_tokens),
		"before_output_token_count": len(before_tokens),
		"after_output_token_count": len(after_tokens),
	}


def run_prompts(prompts: list[str], model_dir: Path, max_new_tokens: int, device: torch.device) -> list[str]:
	tokenizer = _load_tokenizer(model_dir, local_files_only=True, fallback_to_gpt2=True)
	model = load_gpt2_brainsurgery(model_dir, device)

	outputs: list[str] = []
	for prompt in prompts:
		outputs.append(_generate_text(model, tokenizer, prompt, max_new_tokens, device))
	return outputs


def _list_model_presets() -> None:
	for name, config in MODEL_PRESETS.items():
		print(
			f"{name}: base={config['base_model']} "
			f"loader={config['model_loader']} tokenizer={config['tokenizer_source']}"
		)


def _resolve_validation_config(args: argparse.Namespace) -> dict[str, str]:
	preset = MODEL_PRESETS[args.model_preset]
	base_model = args.base_model or preset["base_model"]
	test_model = args.model or preset.get("test_model")
	if test_model is None:
		raise SystemExit(
			"--model is required for this preset; pass the transformed/restored checkpoint "
			f"to compare against {base_model}"
		)
	tokenizer_source = args.tokenizer_source or preset.get("tokenizer_source") or base_model
	config_source = preset.get("config_source") or base_model
	model_loader = args.model_loader or preset["model_loader"]
	tokenizer_loader = args.tokenizer_loader or preset.get("tokenizer_loader", "auto")
	if tokenizer_loader != "auto":
		raise SystemExit(f"unsupported tokenizer loader: {tokenizer_loader}")
	return {
		"base_model": base_model,
		"test_model": test_model,
		"config_source": config_source,
		"tokenizer_source": tokenizer_source,
		"model_loader": model_loader,
		"tokenizer_loader": tokenizer_loader,
	}


def main() -> None:
	parser = argparse.ArgumentParser(description="Compare prompt inference before and after brainsurgery")
	parser.add_argument(
		"--model-preset",
		choices=sorted(MODEL_PRESETS),
		default="gpt2-validation",
		help="Named validation preset. Use --model to override the preset restored-checkpoint path.",
	)
	parser.add_argument(
		"--list-model-presets",
		action="store_true",
		default=False,
		help="List available model presets and exit",
	)
	parser.add_argument(
		"--base-model-dir",
		dest="base_model",
		type=str,
		default=None,
		help="Backward-compatible alias for --base-model",
	)
	parser.add_argument(
		"--base-model",
		dest="base_model",
		type=str,
		default=None,
		help="Pre-surgery baseline model source: local path or Hugging Face repo id",
	)
	parser.add_argument(
		"--model-dir",
		dest="model",
		type=str,
		default=None,
		help="Backward-compatible alias for --model",
	)
	parser.add_argument(
		"--model",
		dest="model",
		type=str,
		default=None,
		help="Post-surgery/restored model source: local path or Hugging Face repo id",
	)
	parser.add_argument(
		"--output-json",
		type=Path,
		default=DEFAULT_OUTPUT_JSON,
		help="Path where validation metrics are saved",
	)
	parser.add_argument(
		"--model-loader",
		choices=("gpt2-brainsurgery", "hf-causal-lm", "hf-mistral3-conditional-generation"),
		default=None,
		help="Model loading path; defaults to the selected preset",
	)
	parser.add_argument(
		"--tokenizer-source",
		type=str,
		default=None,
		help="Tokenizer source; defaults to the preset tokenizer or baseline model",
	)
	parser.add_argument(
		"--tokenizer-loader",
		choices=("auto",),
		default=None,
		help="Tokenizer loading path; defaults to the selected preset",
	)
	parser.add_argument(
		"--dtype",
		choices=("auto", "float32", "float16", "bfloat16"),
		default="auto",
		help="dtype passed to Hugging Face model loading for --model-loader hf-causal-lm",
	)
	parser.add_argument(
		"--local-files-only",
		action="store_true",
		default=False,
		help="Load Hugging Face models/tokenizers only from the local cache",
	)
	parser.add_argument(
		"--trust-remote-code",
		action=argparse.BooleanOptionalAction,
		default=True,
		help="Pass trust_remote_code to Hugging Face Auto loaders",
	)
	parser.add_argument(
		"--print-config",
		action="store_true",
		default=False,
		help="Print the resolved validation configuration and exit before loading models",
	)
	parser.add_argument(
		"--include-outputs",
		action="store_true",
		default=False,
		help="Include generated before/after output text in stdout and JSON; implies --include-prompt-details",
	)
	parser.add_argument(
		"--include-prompt-details",
		action="store_true",
		default=False,
		help="Print and save per-prompt prompts and metrics",
	)
	parser.add_argument(
		"--prompt-file",
		type=Path,
		default=DEFAULT_PROMPT_FILE,
		help="Path to newline-separated validation prompts",
	)
	parser.add_argument(
		"--max-new-tokens",
		type=int,
		default=40,
		help="Number of tokens to generate per prompt",
	)
	parser.add_argument(
		"--device",
		type=str,
		default=None,
		help="Optional device override (e.g. cpu, cuda, mps)",
	)
	parser.add_argument(
		"--prompt",
		action="append",
		default=[],
		help="Prompt text (repeat --prompt for multiple prompts)",
	)
	args = parser.parse_args()
	if args.list_model_presets:
		_list_model_presets()
		return
	resolved_config = _resolve_validation_config(args)
	if args.print_config:
		print(json.dumps(resolved_config, indent=2))
		return
	include_prompt_details = args.include_prompt_details or args.include_outputs

	device = _choose_device(args.device)
	effective_dtype_name = _effective_dtype_name(args.dtype, device)
	if effective_dtype_name != args.dtype:
		print(
			"Using dtype=float32 because Apple MPS does not support bfloat16 "
			f"for this validation path (requested dtype={args.dtype})."
		)
	prompts = _read_prompts(args.prompt, args.prompt_file)
	fallback_to_gpt2 = resolved_config["model_loader"] == "gpt2-brainsurgery"
	tokenizer = _load_tokenizer(
		resolved_config["tokenizer_source"],
		local_files_only=args.local_files_only if not fallback_to_gpt2 else True,
		trust_remote_code=args.trust_remote_code,
		fallback_to_gpt2=fallback_to_gpt2,
	)
	base_model = load_model(
		resolved_config["base_model"],
		device,
		model_loader=resolved_config["model_loader"],
		dtype_name=effective_dtype_name,
		config_source=resolved_config.get("config_source"),
		trust_remote_code=args.trust_remote_code,
		local_files_only=args.local_files_only,
	)
	test_model = load_model(
		resolved_config["test_model"],
		device,
		model_loader=resolved_config["model_loader"],
		dtype_name=effective_dtype_name,
		config_source=resolved_config.get("config_source"),
		trust_remote_code=args.trust_remote_code,
		local_files_only=args.local_files_only,
	)

	rows = []
	cosines: list[float] = []
	full_sequence_mean_cosines: list[float] = []
	full_sequence_min_cosines: list[float] = []
	full_sequence_mean_abs_diffs: list[float] = []
	full_sequence_max_abs_diffs: list[float] = []
	full_sequence_positions = 0
	exact_matches: list[bool] = []
	char_similarities: list[float] = []
	token_sequence_similarities: list[float] = []
	token_bag_cosines: list[float] = []
	tokenizer_label = _tokenizer_label(tokenizer)

	print(f"Model preset:              {args.model_preset}")
	print(f"Model loader:              {resolved_config['model_loader']}")
	print(f"Base model before surgery: {resolved_config['base_model']}")
	print(f"Model after surgery:       {resolved_config['test_model']}")
	print(f"Tokenizer:                 {tokenizer_label}")
	print(f"Running on device: {device}")
	for i, prompt in enumerate(prompts, start=1):
		encoded = _encode_prompt(tokenizer, prompt, device)

		base_generated_ids = _generate_ids(base_model, tokenizer, prompt, args.max_new_tokens, device)
		test_generated_ids = _generate_ids(test_model, tokenizer, prompt, args.max_new_tokens, device)
		base_output = tokenizer.decode(base_generated_ids[0], skip_special_tokens=True)
		test_output = tokenizer.decode(test_generated_ids[0], skip_special_tokens=True)
		output_metrics = _output_similarity_metrics(tokenizer, base_output, test_output)
		reference_attention_mask = torch.ones_like(base_generated_ids, device=device)
		full_sequence_metrics = _full_sequence_logit_metrics(
			base_model,
			test_model,
			base_generated_ids,
			reference_attention_mask,
		)
		base_logits = _last_token_logits(base_model, encoded["input_ids"], encoded["attention_mask"])
		test_logits = _last_token_logits(test_model, encoded["input_ids"], encoded["attention_mask"])
		cosine = float(F.cosine_similarity(base_logits, test_logits, dim=0).item())
		cosines.append(cosine)
		full_sequence_mean_cosines.append(float(full_sequence_metrics["full_sequence_mean_logit_cosine"]))
		full_sequence_min_cosines.append(float(full_sequence_metrics["full_sequence_min_logit_cosine"]))
		full_sequence_mean_abs_diffs.append(float(full_sequence_metrics["full_sequence_mean_abs_logit_diff"]))
		full_sequence_max_abs_diffs.append(float(full_sequence_metrics["full_sequence_max_abs_logit_diff"]))
		full_sequence_positions += int(full_sequence_metrics["full_sequence_positions_compared"])
		exact_matches.append(bool(output_metrics["output_exact_match"]))
		char_similarities.append(float(output_metrics["output_char_similarity"]))
		token_sequence_similarities.append(float(output_metrics["output_token_sequence_similarity"]))
		token_bag_cosines.append(float(output_metrics["output_token_bag_cosine"]))

		if include_prompt_details:
			row = {
				"index": i,
				"prompt": prompt,
				"last_token_logit_cosine": cosine,
				**full_sequence_metrics,
				**output_metrics,
			}
			if args.include_outputs:
				row["before_surgery"] = base_output
				row["after_surgery"] = test_output
			rows.append(row)

			print(f"\n--- Prompt {i} ---")
			print(prompt)
			print(f"Last-token logit cosine: {cosine:.8f}")
			print(f"Full-sequence mean logit cosine: {full_sequence_metrics['full_sequence_mean_logit_cosine']:.12f}")
			print(f"Full-sequence min logit cosine: {full_sequence_metrics['full_sequence_min_logit_cosine']:.12f}")
			print(f"Full-sequence mean abs logit diff: {full_sequence_metrics['full_sequence_mean_abs_logit_diff']:.12g}")
			print(f"Full-sequence max abs logit diff: {full_sequence_metrics['full_sequence_max_abs_logit_diff']:.12g}")
			print(f"Output exact match: {output_metrics['output_exact_match']}")
			print(f"Output char similarity: {output_metrics['output_char_similarity']:.8f}")
			print(f"Output token sequence similarity: {output_metrics['output_token_sequence_similarity']:.8f}")
			print(f"Output token bag cosine: {output_metrics['output_token_bag_cosine']:.8f}")
			if args.include_outputs:
				print("--- Output before surgery ---")
				print(base_output)
				print("--- Output after surgery ---")
				print(test_output)

	average_cosine = sum(cosines) / len(cosines)
	average_full_sequence_mean_cosine = sum(full_sequence_mean_cosines) / len(full_sequence_mean_cosines)
	min_full_sequence_logit_cosine = min(full_sequence_min_cosines)
	average_full_sequence_mean_abs_diff = sum(full_sequence_mean_abs_diffs) / len(full_sequence_mean_abs_diffs)
	max_full_sequence_abs_logit_diff = max(full_sequence_max_abs_diffs)
	exact_match_rate = sum(exact_matches) / len(exact_matches)
	average_char_similarity = sum(char_similarities) / len(char_similarities)
	average_token_sequence_similarity = sum(token_sequence_similarities) / len(token_sequence_similarities)
	average_token_bag_cosine = sum(token_bag_cosines) / len(token_bag_cosines)
	payload = {
		"model_preset": args.model_preset,
		"model_loader": resolved_config["model_loader"],
		"base_model_before_surgery": resolved_config["base_model"],
		"model_after_surgery": resolved_config["test_model"],
		"config_source": resolved_config.get("config_source"),
		"tokenizer": tokenizer_label,
		"tokenizer_source": resolved_config["tokenizer_source"],
		"device": str(device),
		"requested_dtype": args.dtype,
		"effective_dtype": effective_dtype_name,
		"max_new_tokens": args.max_new_tokens,
		"prompt_count": len(prompts),
		"includes_prompt_details": include_prompt_details,
		"includes_generated_outputs": args.include_outputs,
		"logit_comparison_reference": "base_model_greedy_generated_sequences",
		"full_sequence_positions_compared": full_sequence_positions,
		"average_full_sequence_mean_logit_cosine": average_full_sequence_mean_cosine,
		"min_full_sequence_logit_cosine": min_full_sequence_logit_cosine,
		"average_full_sequence_mean_abs_logit_diff": average_full_sequence_mean_abs_diff,
		"max_full_sequence_abs_logit_diff": max_full_sequence_abs_logit_diff,
		"average_last_token_logit_cosine": average_cosine,
		"output_exact_match_rate": exact_match_rate,
		"average_output_char_similarity": average_char_similarity,
		"average_output_token_sequence_similarity": average_token_sequence_similarity,
		"average_output_token_bag_cosine": average_token_bag_cosine,
	}
	if include_prompt_details:
		payload["prompts"] = rows
	args.output_json.parent.mkdir(parents=True, exist_ok=True)
	args.output_json.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

	print("\nSummary:")
	print(f"Model preset: {args.model_preset}")
	print(f"Model loader: {resolved_config['model_loader']}")
	print(f"Base model before surgery: {resolved_config['base_model']}")
	print(f"Model after surgery:       {resolved_config['test_model']}")
	print(f"Tokenizer:                 {tokenizer_label}")
	print(f"Prompt count: {len(prompts)}")
	print("Logit comparison reference: base model greedy generated sequences")
	print(f"Full-sequence positions compared: {full_sequence_positions}")
	print(f"Average full-sequence mean logit cosine: {average_full_sequence_mean_cosine:.12f}")
	print(f"Minimum full-sequence logit cosine: {min_full_sequence_logit_cosine:.12f}")
	print(f"Average full-sequence mean abs logit diff: {average_full_sequence_mean_abs_diff:.12g}")
	print(f"Maximum full-sequence abs logit diff: {max_full_sequence_abs_logit_diff:.12g}")
	print(f"Average last-token logit cosine: {average_cosine:.8f}")
	print(f"Output exact match rate: {exact_match_rate:.2%}")
	print(f"Average output char similarity: {average_char_similarity:.8f}")
	print(f"Average output token sequence similarity: {average_token_sequence_similarity:.8f}")
	print(f"Average output token bag cosine: {average_token_bag_cosine:.8f}")
	print(f"Includes prompt details: {include_prompt_details}")
	print(f"Includes generated outputs: {args.include_outputs}")
	print(f"Saved validation report to: {args.output_json}")


if __name__ == "__main__":
	main()
