# Validation

Run commands from the repository root.

## Inference Validation

Compares an original checkpoint against a transformed/restored checkpoint using
the prompts in `validation/prompts.txt`. By default it prints and saves only
aggregate statistics: output agreement/similarity, final-token logit cosine, and
full-sequence full-vocabulary logit cosine/difference metrics.

```bash
conda run -n brainsurgery python validation/test_inference.py
```

Defaults:

- preset: `gpt2-validation`
- original model: `models/test/gpt2`
- restored model: `models/test/validation`
- report: `validation/model_outputs.json`

Useful flags:

- `--model-preset`: choose a configured model pair.
- `--model`: override the restored checkpoint path.
- `--prompt-file`: use a different prompt file.
- `--include-prompt-details`: include per-prompt metrics.
- `--include-outputs`: include generated text before/after surgery.
- `--device cpu --dtype float32`: recommended for reproducible paper numbers.

List presets:

```bash
conda run -n brainsurgery python validation/test_inference.py --list-model-presets
```

Configured presets:

| preset | original path | restored path |
| --- | --- | --- |
| `gpt2-validation` | `models/test/gpt2` | `models/test/validation` |
| `qwen3-1.7b-base` | `models/test/qwen3-1.7b-base` | `models/test/qwen3_1_7b_base_validation` |
| `apertus-v1.1-1.5b` | `models/test/apertus-v1.1-1.5b` | `models/test/apertus_v1_1_1_5b_validation` |
| `ministral3-3b-base-2512` | `models/test/ministral3-3b-base-2512` | `models/test/ministral3_3b_base_2512_validation` |

Example:

```bash
conda run -n brainsurgery python validation/test_inference.py \
  --model-preset qwen3-1.7b-base \
  --local-files-only \
  --device cpu \
  --dtype float32 \
  --output-json validation/model_outputs_qwen3_1_7b.json
```

For Hugging Face-style presets, the restored checkpoint is loaded as weights and
the config/tokenizer are read from the original model directory. On Apple MPS,
`auto`/`bfloat16` validation dtype is promoted to `float32`.

## Qwen Restored Checkpoint

Generate the Qwen restored checkpoint before running its inference/regression
validation:

```bash
conda run --no-capture-output -n brainsurgery python -c "from brainsurgery import main; main()" \
  validation/validation.yaml \
  validation/validation_qwen3_1_7b.yaml \
  --provider inmemory \
  --num-workers 2 \
  --no-summarize
```

## Regression Check

`test_regression.py` is a lighter pass/fail gate. It does not generate
continuations; it compares prompt perplexity, final-token full-vocabulary logits,
and final-token top-1 agreement.

```bash
conda run -n brainsurgery python validation/test_regression.py
```

Example with Qwen:

```bash
conda run -n brainsurgery python validation/test_regression.py \
  --model-preset qwen3-1.7b-base \
  --local-files-only \
  --device cpu \
  --dtype float32
```

Default thresholds:

- `--min-cosine 0.9999`
- `--max-ppl-ratio 1.01`

## Plain-PyTorch Parity Check

Checks that `validation/validation.yaml` behaves the same through the
brainsurgery transform engine and the independent plain-PyTorch implementation in
`validation/pytorch_example.py`.

```bash
conda run -n brainsurgery python validation/test_equal.py
```

The parity check compares aliases, tensor keys, shapes, dtypes, values, and
control flow step-by-step. It writes the plain-PyTorch mirror checkpoint to
`models/test/validation_pytorch/model.pt`.
