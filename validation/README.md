# Validation

Run commands from the repository root.

## Generate Inference Metrics

```bash
conda run -n brainsurgery python validation/test_inference.py
```

Defaults:

- pre-surgery model: `models/test/gpt2`
- post-surgery model: `models/test/validation`
- model preset: `gpt2-validation`
- prompts: `validation/prompts.txt`
- report: `validation/model_outputs.json`

By default, stdout and the JSON report include only aggregate statistics. Prompt
text, per-prompt metrics, and generated before/after text are omitted unless
explicitly requested.

The inference report compares deterministic before/after generations and also
performs a distribution-level check: it replays each baseline greedy-generated
sequence through both checkpoints and compares full-vocabulary logits at every
next-token position. The aggregate report includes mean/min full-sequence logit
cosine and absolute logit-difference statistics.

List supported model presets:

```bash
conda run -n brainsurgery python validation/test_inference.py --list-model-presets
```

Use `--model-preset` to select a reference model. Each preset also has a default
restored-checkpoint path, so the inference command is ready to run after the
transform plan writes to that path. The larger-model presets expect the original
models to be downloaded locally under `models/test/`. You can still override the
restored path with `--model`.

Qwen 3 1.7B:

```bash
conda run -n brainsurgery python validation/test_inference.py \
  --model-preset qwen3-1.7b-base \
  --local-files-only \
  --output-json validation/model_outputs_qwen3_1_7b.json
```

Default original path: `models/test/qwen3-1.7b-base`
Default restored path: `models/test/qwen3_1_7b_base_validation`

Run the Qwen transform plan first:

```bash
conda run --no-capture-output -n brainsurgery python -c "from brainsurgery import main; main()" \
  validation/validation.yaml \
  validation/validation_qwen3_1_7b.yaml \
  --provider inmemory \
  --num-workers 2
```

Apertus v1.1 1.5B:

```bash
conda run -n brainsurgery python validation/test_inference.py \
  --model-preset apertus-v1.1-1.5b \
  --local-files-only \
  --output-json validation/model_outputs_apertus_v1_1_1_5b.json
```

Default original path: `models/test/apertus-v1.1-1.5b`
Default restored path: `models/test/apertus_v1_1_1_5b_validation`

Ministral 3 3B:

```bash
conda run -n brainsurgery python validation/test_inference.py \
  --model-preset ministral3-3b-base-2512 \
  --local-files-only \
  --output-json validation/model_outputs_ministral3_3b.json
```

Default original path: `models/test/ministral3-3b-base-2512`
Default restored path: `models/test/ministral3_3b_base_2512_validation`

Use `--local-files-only` after the original and restored models/tokenizers are
already present in the local Hugging Face cache. Use `--dtype float32` for the
most conservative numerical comparison, or leave `--dtype auto` to use the
checkpoint's default loading dtype.

On Apple MPS, `--dtype auto` and `--dtype bfloat16` are promoted to `float32`
inside the validation scripts because MPS does not support bfloat16 execution.
For the most reproducible paper numbers, prefer `--device cpu --dtype float32`.

Include per-prompt prompts and metrics:

```bash
conda run -n brainsurgery python validation/test_inference.py --include-prompt-details
```

Include generated before/after text as well:

```bash
conda run -n brainsurgery python validation/test_inference.py --include-outputs
```

Useful quick smoke run:

```bash
conda run -n brainsurgery python validation/test_inference.py \
  --device cpu \
  --max-new-tokens 1 \
  --prompt "The history of machine learning is"
```

Use a different prompt file with one prompt per line:

```bash
conda run -n brainsurgery python validation/test_inference.py \
  --prompt-file validation/prompts.txt
```

## Quantitative Regression Check

```bash
conda run -n brainsurgery python validation/test_regression.py
```

This is a stricter pass/fail regression gate for two checkpoints:

- baseline/reference model: `--base-model-dir`, default `models/test/gpt2`
- transformed/test model: `--test-model-dir`, default `models/test/validation`
- prompts: `--prompt-file`, default `validation/prompts.txt`

For each prompt, the script tokenizes the prompt once with the baseline tokenizer
and feeds the same prompt tokens to both checkpoints. It does not generate
continuations. Instead, it compares the two models on the prompt itself:

- prompt perplexity for each model, using the prompt tokens as labels;
- the ratio `max(ppl_base, ppl_test) / min(ppl_base, ppl_test)`;
- cosine similarity between the full-vocabulary logits at the final prompt token;
- whether the final-token top-1 prediction is identical.

The run prints one line per prompt with those metrics, then reports:

- mean final-token logit cosine;
- mean perplexity ratio;
- top-1 match count/rate;
- configured pass/fail thresholds.

By default, the check fails if:

- mean final-token logit cosine is below `--min-cosine 0.9999`; or
- mean perplexity ratio is above `--max-ppl-ratio 1.01`.

Use this script when you want a quick quantitative regression gate. Use
`validation/test_inference.py` when you want the richer validation report with
deterministic generation, output-similarity metrics, and full-sequence
full-vocabulary logit comparisons.

Example with explicit paths:

```bash
conda run -n brainsurgery python validation/test_regression.py \
  --base-model-dir models/test/gpt2 \
  --test-model-dir models/test/validation \
  --prompt-file validation/prompts.txt
```

The same model presets are available for the regression gate. For example:

```bash
conda run -n brainsurgery python validation/test_regression.py \
  --model-preset qwen3-1.7b-base
```

## Plain-PyTorch Transform Parity Check

```bash
conda run -n brainsurgery python validation/test_equal.py
```

This checks that the declarative transform pipeline in
`validation/validation.yaml` behaves the same as an independent plain-PyTorch
implementation of the same operations.

The test uses:

- validation plan: `validation/validation.yaml`
- input checkpoint from the plan: default `models/test/gpt2`
- brainsurgery/tool output path from the plan: default `models/test/validation`
- plain-PyTorch mirror output: `models/test/validation_pytorch/model.pt`

For each transform entry in `validation/validation.yaml`, the script:

- compiles and applies the transform through the real brainsurgery transform
  engine;
- applies the same transform to an independent in-memory PyTorch tensor store
  using `validation/pytorch_example.py`;
- snapshots both stores after the step;
- checks that aliases, tensor keys, shapes, dtypes, and values match;
- checks that transform control flow, such as `exit`, agrees.

Floating-point and complex tensors are compared with
`torch.allclose(atol=1e-6, rtol=1e-6)`. Non-floating tensors are compared with
exact equality.

At the end, the plain-PyTorch mirror checkpoint is saved to
`models/test/validation_pytorch/model.pt`. Passing this test means the YAML/tool
execution and the independent PyTorch implementation agree step-by-step for the
validation transform plan. It is a transform-semantics parity check, not a model
generation or inference-quality benchmark.
