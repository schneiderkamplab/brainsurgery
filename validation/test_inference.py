from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING

import torch
from safetensors.torch import load_file as load_safetensors_file

if TYPE_CHECKING:
	from transformers import GPT2LMHeadModel


def _require_transformers():
	try:
		from transformers import AutoTokenizer, GPT2Config, GPT2LMHeadModel
	except ModuleNotFoundError as exc:
		raise RuntimeError(
			"Missing dependency 'transformers'. Install it with: "
			"conda run -n brainsurgery pip install transformers"
		) from exc
	return AutoTokenizer, GPT2Config, GPT2LMHeadModel


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


def _load_tokenizer(model_dir: Path):
	AutoTokenizer, _, _ = _require_transformers()
	# Prefer local tokenizer assets if present; otherwise fallback to HF GPT-2 tokenizer.
	try:
		tok = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
	except Exception:
		tok = AutoTokenizer.from_pretrained("gpt2", local_files_only=False)
	if tok.pad_token is None:
		tok.pad_token = tok.eos_token
	return tok


def load_gpt2_brainsurgery(model_dir: Path, device: torch.device) -> "GPT2LMHeadModel":
	_, GPT2Config, GPT2LMHeadModel = _require_transformers()
	state_dict = _normalize_gpt2_state_dict_keys(_load_sharded_state_dict(model_dir))
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


def run_prompts(prompts: list[str], model_dir: Path, max_new_tokens: int, device: torch.device) -> list[str]:
	tokenizer = _load_tokenizer(model_dir)
	model = load_gpt2_brainsurgery(model_dir, device)

	outputs: list[str] = []
	for prompt in prompts:
		encoded = tokenizer(prompt, return_tensors="pt").to(device)
		with torch.inference_mode():
			generated = model.generate(
				**encoded,
				max_new_tokens=max_new_tokens,
				do_sample=False,
				pad_token_id=tokenizer.pad_token_id,
			)
		text = tokenizer.decode(generated[0], skip_special_tokens=True)
		outputs.append(text)
	return outputs


def main() -> None:
	parser = argparse.ArgumentParser(description="Minimal GPT-2 prompt inference after brainsurgery")
	parser.add_argument(
		"--model-dir",
		type=Path,
		default=Path("models/test/validation"),
		help="Path to brainsurgery output model directory",
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

	prompts = args.prompt or [
        "The history of machine learning is",
        "In a distant future, humans and AI",
        "Write a short Python function to compute Fibonacci numbers:",
        "A recipe for a perfect breakfast includes",
        "Translate the following sentence into French:",
        "Summarize the main idea of a story about a brave knight:",
        "The sound of rain on a window feels like",
        "List three ways to improve memory retention:",
        "Describe a color to someone who has never seen:",
        "The economic impact of inflation can be explained as",
        "Write a haiku about autumn leaves:",
        "Debug the following code snippet:",
        "A conversation between a cat and a dog might go like",
        "Explain the rules of chess in simple terms:",
        "The strangest dream I ever had involved",
        "Provide step-by-step instructions for tying a tie:",
        "The philosophy of existentialism suggests that",
        "Generate a creative name for a startup that",
        "Describe the taste of coffee to a robot:",
        "What would happen if humans could breathe underwater?",
        "Write a motivational quote about persistence:",
        "The main ingredients in a classic pizza are",
        "Explain how GPS navigation works:",
        "Create a fictional planet where",
        "The difference between RAM and ROM is",
        "Write a short email requesting a meeting:",
        "The cultural significance of music in society is",
        "Describe a sunset using only metaphors:",
        "List the steps to solve a quadratic equation:",
        "Imagine a world without electricity and describe",
        "The role of humor in communication is",
        "Write a tweet about a new technological breakthrough:",
        "Explain the concept of supply and demand:",
        "A mysterious package arrives containing",
        "The benefits and drawbacks of remote work include",
        "Describe how to train a dog to sit:",
        "The importance of sleep for health is",
        "Write a short sci-fi story starting with a glitch:",
        "Explain how photosynthesis differs from respiration:",
        "The most challenging puzzle I encountered was",
        "Create a dialogue between a teacher and a student about math:",
        "List five tips for effective public speaking:",
        "Describe the feeling of standing on a mountain peak:",
        "The history of the internet began when",
        "Write a product description for a smart watch:",
        "Explain the concept of gravity to a child:",
        "A detective investigates a crime scene where",
        "The role of emotions in decision-making is",
        "Write a short bedtime story for children:",
        "Predict the future of transportation in 100 years:"
    ]

	device = _choose_device(args.device)
	outputs = run_prompts(
		prompts=prompts,
		model_dir=args.model_dir,
		max_new_tokens=args.max_new_tokens,
		device=device,
	)

	print(f"Loaded model from: {args.model_dir}")
	print(f"Running on device: {device}")
	for i, (prompt, output) in enumerate(zip(prompts, outputs), start=1):
		print(f"\n--- Prompt {i} ---")
		print(prompt)
		print("--- Output ---")
		print(output)


if __name__ == "__main__":
	main()
