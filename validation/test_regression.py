from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F
from test_inference import (
	DEFAULT_PROMPT_FILE,
	MODEL_PRESETS,
	_choose_device,
	_effective_dtype_name,
	_encode_prompt,
	_list_model_presets,
	_load_tokenizer,
	_read_prompts,
	_resolve_validation_config,
	load_model,
)


def _perplexity(model: torch.nn.Module, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> float:
	with torch.inference_mode():
		loss = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids).loss
	return float(math.exp(float(loss.item())))


def _last_token_logits(
	model: torch.nn.Module,
	input_ids: torch.Tensor,
	attention_mask: torch.Tensor,
) -> torch.Tensor:
	with torch.inference_mode():
		logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
	last_index = int(attention_mask[0].sum().item()) - 1
	return logits[0, last_index, :].detach().float().cpu()


def main() -> None:
	parser = argparse.ArgumentParser(description="Quantitative regression check before vs after brainsurgery")
	parser.add_argument("--model-preset", choices=sorted(MODEL_PRESETS), default="gpt2-validation")
	parser.add_argument("--list-model-presets", action="store_true", default=False)
	parser.add_argument("--base-model-dir", dest="base_model", type=str, default=None)
	parser.add_argument("--base-model", dest="base_model", type=str, default=None)
	parser.add_argument("--test-model-dir", dest="model", type=str, default=None)
	parser.add_argument("--model-dir", dest="model", type=str, default=None)
	parser.add_argument("--model", dest="model", type=str, default=None)
	parser.add_argument(
		"--model-loader",
		choices=("gpt2-brainsurgery", "hf-causal-lm", "hf-mistral3-conditional-generation"),
		default=None,
	)
	parser.add_argument("--tokenizer-source", type=str, default=None)
	parser.add_argument("--tokenizer-loader", choices=("auto",), default=None)
	parser.add_argument("--dtype", choices=("auto", "float32", "float16", "bfloat16"), default="auto")
	parser.add_argument("--local-files-only", action="store_true", default=False)
	parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
	parser.add_argument("--print-config", action="store_true", default=False)
	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--prompt", action="append", default=[])
	parser.add_argument("--prompt-file", type=Path, default=DEFAULT_PROMPT_FILE)
	parser.add_argument(
		"--min-cosine",
		type=float,
		default=0.9999,
		help="Fail if average last-token cosine is below this",
	)
	parser.add_argument(
		"--max-ppl-ratio",
		type=float,
		default=1.01,
		help="Fail if mean(max(ppl_a,ppl_b)/min(ppl_a,ppl_b)) is above this",
	)
	args = parser.parse_args()
	if args.list_model_presets:
		_list_model_presets()
		return
	resolved_config = _resolve_validation_config(args)
	if args.print_config:
		print(json.dumps(resolved_config, indent=2))
		return

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

	cosines: list[float] = []
	ppl_ratios: list[float] = []
	top1_matches = 0

	print(f"Device: {device}")
	print(f"Model preset: {args.model_preset}")
	print(f"Model loader: {resolved_config['model_loader']}")
	print(f"Base model: {resolved_config['base_model']}")
	print(f"Test model: {resolved_config['test_model']}")

	for idx, prompt in enumerate(prompts, start=1):
		encoded = _encode_prompt(tokenizer, prompt, device)

		base_ppl = _perplexity(base_model, encoded["input_ids"], encoded["attention_mask"])
		test_ppl = _perplexity(test_model, encoded["input_ids"], encoded["attention_mask"])
		ppl_ratio = max(base_ppl, test_ppl) / max(min(base_ppl, test_ppl), 1e-12)

		base_last = _last_token_logits(base_model, encoded["input_ids"], encoded["attention_mask"])
		test_last = _last_token_logits(test_model, encoded["input_ids"], encoded["attention_mask"])
		cosine = float(F.cosine_similarity(base_last, test_last, dim=0).item())

		base_top = int(torch.argmax(base_last).item())
		test_top = int(torch.argmax(test_last).item())
		if base_top == test_top:
			top1_matches += 1

		cosines.append(cosine)
		ppl_ratios.append(ppl_ratio)

		print(
			f"[PROMPT {idx}] cosine={cosine:.8f} ppl_base={base_ppl:.6f} "
			f"ppl_test={test_ppl:.6f} ppl_ratio={ppl_ratio:.8f} top1_match={base_top == test_top}"
		)

	mean_cosine = sum(cosines) / len(cosines)
	mean_ppl_ratio = sum(ppl_ratios) / len(ppl_ratios)
	top1_rate = top1_matches / len(prompts)

	print("\nSummary:")
	print(f"mean cosine:    {mean_cosine:.8f}")
	print(f"mean ppl ratio: {mean_ppl_ratio:.8f}")
	print(f"top1 match:     {top1_matches}/{len(prompts)} ({top1_rate:.2%})")
	print(f"thresholds: min_cosine>={args.min_cosine}, max_ppl_ratio<={args.max_ppl_ratio}")

	failed = False
	if mean_cosine < args.min_cosine:
		print("FAIL: cosine threshold not met")
		failed = True
	if mean_ppl_ratio > args.max_ppl_ratio:
		print("FAIL: perplexity ratio threshold exceeded")
		failed = True

	if failed:
		raise SystemExit(1)

	print("PASS: transformed model matches baseline within thresholds")


if __name__ == "__main__":
	main()
