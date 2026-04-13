from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from test_inference import _choose_device, _load_tokenizer, load_gpt2_brainsurgery


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


def _read_prompts(prompt_args: list[str], prompt_file: Path | None) -> list[str]:
	if prompt_args:
		return prompt_args
	if prompt_file is not None:
		lines = [line.strip() for line in prompt_file.read_text(encoding="utf-8").splitlines()]
		prompts = [line for line in lines if line]
		if not prompts:
			raise RuntimeError(f"prompt file has no non-empty lines: {prompt_file}")
		return prompts
	return [
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


def main() -> None:
	parser = argparse.ArgumentParser(description="Quantitative regression check before vs after brainsurgery")
	parser.add_argument("--base-model-dir", type=Path, default=Path("models/test/gpt2"))
	parser.add_argument("--test-model-dir", type=Path, default=Path("models/test/validation"))
	parser.add_argument("--device", type=str, default=None)
	parser.add_argument("--prompt", action="append", default=[])
	parser.add_argument("--prompt-file", type=Path, default=None)
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

	device = _choose_device(args.device)
	prompts = _read_prompts(args.prompt, args.prompt_file)

	tokenizer = _load_tokenizer(args.base_model_dir)
	base_model = load_gpt2_brainsurgery(args.base_model_dir, device)
	test_model = load_gpt2_brainsurgery(args.test_model_dir, device)

	cosines: list[float] = []
	ppl_ratios: list[float] = []
	top1_matches = 0

	print(f"Device: {device}")
	print(f"Base model: {args.base_model_dir}")
	print(f"Test model: {args.test_model_dir}")

	for idx, prompt in enumerate(prompts, start=1):
		encoded = tokenizer(prompt, return_tensors="pt")
		encoded = {k: v.to(device) for k, v in encoded.items()}

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
