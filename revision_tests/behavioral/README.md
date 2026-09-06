# Behavioral regression evaluation

This directory contains the frozen, sourced prompt suite and comparison tools
for the EACL 2027 BrainSurgery revision. It replaces the undocumented
`validation/prompts.txt` list for paper-facing evaluation.

The suite contains 70 prompts from three pinned, redistributable sources:

| Source | Items | Coverage |
|---|---:|---|
| Belebele | 30 | Five parallel questions in six language varieties |
| MMLU | 30 | Five questions in six subject strata |
| HumanEval | 10 | Python code completion |

The exact items, rendered prompts, upstream identifiers, row numbers,
languages, categories, licenses, deterministic selection ranks, and SHA-256
checksums are in `prompt_manifest.jsonl`. `manifest_summary.json` records the
aggregate coverage and source fingerprints. See `DATA_LICENSES.md` before
redistributing the prompt text.

## Local validation

From the repository root:

```bash
.venv/bin/python revision_tests/behavioral/validate_manifest.py
.venv/bin/python -m pytest -q revision_tests/behavioral/test_analysis.py
```

The tests use tiny synthetic result bundles and deliberately corrupt one logit,
one generated sequence, and one prompt ID. They do not load a language model.

## Reproducing the manifest

Download the three files at the exact URLs/revisions and verify the SHA-256
values in `sources.yaml`; unpack the Belebele archive without modifying its
JSONL files. Then run:

```bash
uv run --isolated --no-project --with pyarrow --with pyyaml \
  python revision_tests/behavioral/prepare_manifest.py \
  --belebele-archive <path/to/Belebele.zip> \
  --belebele-dir <path/to/unpacked/belebele> \
  --mmlu-parquet <path/to/mmlu-test.parquet> \
  --human-eval <path/to/HumanEval.jsonl.gz>
```

The generator refuses source files whose checksums differ. A successful
reproduction must yield manifest SHA-256
`abe5a763d3f39a77c720950826e45ab32b6d1e71910769a73ad77555a4d7e412`.

Each model run creates `metadata.json`, `predictions.jsonl`, and
`last_token_logits.safetensors`. The analyzer first verifies prompt identity,
manifest and tokenizer fingerprints, architecture, software, hardware, dtype,
and decoding settings. It refuses to compare incompatible bundles.

## Frozen reported CUDA case

The reported case uses the pinned GPT-2 124M checkpoint at revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`. The frozen BrainSurgery plan
performs a copy/assert/delete round trip, moves all transformer-block keys to a
temporary namespace and back, applies an exact multiply-by-one operation, and
exports the final 160 tensors as indexed 256 MiB safetensors shards. Before
inference, `validate_lossless.py` independently requires every final tensor to
be byte-exact and validates the shard index and budget.

On the clean Linux/CUDA checkout, run:

```bash
revision_tests/behavioral/run_cuda.sh
```

This single command validates CUDA and the prompt manifest, creates and checks
the transformed checkpoint, copies the pinned configuration sidecars, runs all
70 prompts on the reference and
transformed models sequentially on `cuda:0`, and writes the comparison below
`log/revision_tests/eacl2027_behavioral_cuda_<commit>/behavioral/`. It refuses
to overwrite an existing transformed checkpoint or result bundle.

## Component model-execution commands

Internally, the behavioral comparison is deliberately split into three commands so
the reference and transformed models need not coexist in GPU memory:

```bash
.venv/bin/python revision_tests/behavioral/run_model.py \
  --role reference --model models/<reference> --tokenizer models/<reference> \
  --revision <pinned_revision> --device cuda \
  --dtype float32 \
  --output log/revision_tests/<run_id>/behavioral/reference

.venv/bin/python revision_tests/behavioral/run_model.py \
  --role transformed --model models/<transformed> --tokenizer models/<reference> \
  --config models/<reference> --revision <transformation_commit_or_manifest> \
  --device cuda --dtype float32 \
  --output log/revision_tests/<run_id>/behavioral/transformed

.venv/bin/python revision_tests/behavioral/analyze.py \
  --reference log/revision_tests/<run_id>/behavioral/reference \
  --transformed log/revision_tests/<run_id>/behavioral/transformed \
  --output log/revision_tests/<run_id>/behavioral/comparison.json
```

The reported Linux/CUDA runs must use the same GPU, dtype, tokenizer, prompt
manifest, decoding settings, and software environment for both roles. Mac
smoke runs validate the pipeline but are not mixed with Linux results. A run is
stamped `reported_eligible=true` only when it covers all 70 prompts on CUDA
from a clean Git worktree. This stamp is a provenance guard, not evidence that
the behavioral comparison passed.

For a quick non-reportable pipeline check, add `--smoke-limit 1` to both model
commands. The analyzer accepts matching leading subsets but labels their result
`non-reportable smoke`.

## Claim boundary

This is a regression/canary evaluation. For transformations declared lossless,
it checks exact next-token logits, top-1 tokens, greedy token sequences, and
multiple-choice predictions on the enumerated prompts. It does not establish
broad downstream quality, language competence, HumanEval pass rates, or safety.
Intentionally lossy transformations require a separate downstream protocol and
must not use the lossless pass rule.

The old 50 prompts remain under `validation/` for historical compatibility but
are excluded from this protocol.
