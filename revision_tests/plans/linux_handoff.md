# Linux evaluation handoff

Last verified on the source Mac: 2026-09-06

This is the ordered receiving-machine procedure for the work that cannot be
reported from macOS. It uses the underscore data root exclusively and keeps
the 47 GB transfer archive outside Git. Do not substitute the obsolete local
`models/usability-tests` directory.

## 1. Freeze the checkout and machine record

Push the prepared commits from the source machine, then on Linux check out one
exact commit. Do not run a reported cohort from a moving branch or dirty tree.

```bash
git fetch origin
git checkout "<FROZEN_REVISION_TEST_COMMIT>"
git status --short --branch
git rev-parse HEAD
python3 --version
uv --version
codex --version
uname -a
free -h
df -h .
nvidia-smi
```

Save that output under `log/revision_tests/<run_id>/`. Stop if
`git status --porcelain` is nonempty. Use `uv sync --frozen` to create the repository
environment from `uv.lock`, then confirm the imports:

```bash
uv sync --frozen
.venv/bin/python -c "import torch, safetensors, brainsurgery"
```

Run the frozen protocol checks before downloading evaluation checkpoints:

```bash
test -z "$(git status --porcelain)"
.venv/bin/python -m pytest -q \
  revision_tests/scaling/test_scaling.py \
  revision_tests/competing_tools/test_competing_tools.py
.venv/bin/python revision_tests/scaling/validate_protocol.py
.venv/bin/python revision_tests/competing_tools/validate_protocol.py
```

Stop if any command fails. The feature-coverage and distributed-claim boundary
used when interpreting the results are frozen in
`revision_tests/competing_tools/feature_coverage.md` and
`revision_tests/plans/claim_boundaries.md`.

Before starting Codex, record the exact model id, Codex CLI version, reasoning
settings, and the current official rate-card source and access date. Do not
invent prices. If no official per-token prices exist for the exact model and
access route, stop and revise the cost policy before collecting a cohort.

## 2. Download only the pinned base checkpoints

The transfer bundle deliberately excludes these retrievable base checkpoints.

```bash
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    "openai-community/gpt2",
    revision="607a30d783dfa663caf39e06633721c8d4cfcd7e",
    local_dir="models/gpt2",
    allow_patterns=["*.json", "*.safetensors", "*.txt"],
)
snapshot_download(
    "allenai/OLMo-1B-0724-hf",
    revision="d7cbab742d80589e714b1a2d7f838dcd21cbe143",
    local_dir="models/olmo-1b-0724-hf",
    allow_patterns=["*.json", "*.safetensors", "*.txt"],
)
snapshot_download(
    "EleutherAI/pythia-1b",
    revision="f73d7dcc545c8bd326d8559c8ef84ffe92fea6b2",
    local_dir="models/pythia-1b",
    allow_patterns=["*.json", "*.safetensors", "*.txt"],
)
PY
```

## 3. Verify and install the transferred data

Place `usability_tests_data.tar` in the repository root through a non-Git
transfer. Its frozen metadata is in `linux_handoff_manifest.json`.

```bash
test -f usability_tests_data.tar
test "$(stat -c %s usability_tests_data.tar)" = "50568509440"
echo "7001fedf486d026eea60213108278e4ddba41de40cb5b4a6d0db28f4dbcd28fe  usability_tests_data.tar" | sha256sum -c -
tar -tf usability_tests_data.tar | sed -n '1,20p'
test ! -e models/usability_tests
tar -xf usability_tests_data.tar -C models/
.venv/bin/python usability_tests/setup.py
.venv/bin/python usability_tests/make_docpack.py
.venv/bin/python usability_tests/make_manifest.py --verify
```

The final command must report exactly:

```text
base checkpoints: 20/20 ok
generated inputs (ft1/ft2/lora): 42/42 ok
references: 59/59 ok
docpack: 5/5 ok
verified 126/126 files
```

If base checkpoints fail, remove only the affected base checkpoint directory
and download its pinned revision again. If generated inputs or references
fail, do not regenerate them on Linux: retransmit the verified archive. If the
doc pack fails, confirm the Git commit and editable BrainSurgery installation.
Do not start any participant run until all categories pass.

## 4. Run and audit the first official Codex cell

The tracked `usability_tests/astra` directory is a macOS pilot and is excluded
from official analysis. Use `astra_eacl2027` for every official repeat. The
first cell below becomes part of repeat 1 if it passes; do not delete or replace
a participant outcome merely because it is unfavorable.

Set the four rates from the frozen, archived official rate card:

```bash
export PRICE_IN="<USD_PER_MILLION_INPUT_TOKENS>"
export PRICE_OUT="<USD_PER_MILLION_OUTPUT_TOKENS>"
export PRICE_CACHE_READ="<USD_PER_MILLION_CACHE_READ_TOKENS>"
export PRICE_CACHE_WRITE="<USD_PER_MILLION_CACHE_WRITE_TOKENS>"

.venv/bin/python usability_tests/run_codex.py T1 P \
  --agent astra_eacl2027 \
  --model gpt-6-astra \
  --target gpt-2 \
  --effort light \
  --reasoning-effort low \
  --repeat 1 \
  --venv \
  --timeout 1800 \
  --price-in "$PRICE_IN" \
  --price-out "$PRICE_OUT" \
  --price-cache-read "$PRICE_CACHE_READ" \
  --price-cache-write "$PRICE_CACHE_WRITE"

.venv/bin/python usability_tests/audit_codex.py --agent astra_eacl2027
```

Open the first cell's `transcript.jsonl`, `harness.json`, `grade.json`, and
`review.json`. Confirm the transcript-derived turns, tool calls, token fields,
executions, failed-execution numbers, first-execution result, and time/cap
fields. Classify every failed execution and confirm `review.json.detected`
against the hidden answer. Continue only when the audit has no transcript
mismatch and the manual fields are complete. A parser incompatibility is an
infrastructure failure: preserve it, fix the driver centrally, choose a new
protocol commit/cohort namespace, and repeat the preflight transparently.

## 5. Complete Codex repeat 1, close it, then run repeat 2

The full launcher skips the already complete first cell. Start with one process
because cells can have high RAM peaks.

```bash
AGENT=astra_eacl2027 MODEL=gpt-6-astra \
PRICE_IN="$PRICE_IN" PRICE_OUT="$PRICE_OUT" \
PRICE_CACHE_READ="$PRICE_CACHE_READ" PRICE_CACHE_WRITE="$PRICE_CACHE_WRITE" \
  usability_tests/run_full_codex.sh 1 1

.venv/bin/python usability_tests/audit_codex.py --agent astra_eacl2027
.venv/bin/python usability_tests/analyze.py
```

Complete all failed-execution classifications and review confirmations before
repeat 2. Then use the identical command with repeat `2`. Never pull, edit the
kit, change rates, change Codex versions, or change model settings inside a
repeat. If any of those must change between repeats, record and disclose it.

## 6. Run the CPU/Linux evidence before CUDA workloads

With no competing jobs, run the operation-matched comparison from
`../competing_tools/README.md`; it requires Linux but not CUDA. Then repeat the
frozen robustness protocol with a Linux-specific run ID:

```bash
REVISION_COMMIT_SHORT="$(git rev-parse --short HEAD)"
.venv/bin/python revision_tests/robustness/run.py \
  --run-id "eacl2027_robustness_linux_${REVISION_COMMIT_SHORT}"
```

Keep these runs separate from CUDA results. True disk exhaustion is not part
of the frozen 19-case protocol; attempt it only with a bounded disposable
filesystem or quota and a separately versioned protocol, never by filling the
host filesystem.

## 7. Run CUDA-dependent and large-model evidence

After the usability and CPU/Linux work is closed:

1. run the frozen behavioral regression suite;
2. run the selected downstream-quality case, if retained;
3. run the four Pythia scaling points and the paired GPT-2, OLMo, and Qwen2.5
   architecture/storage checks.

Run the now-frozen CUDA behavioral case with:

```bash
revision_tests/behavioral/run_cuda.sh
```

Do not substitute another model or transformation inside this reported run.
The command independently checks all tensors before starting CUDA inference.

The ten exact model revisions and expected checkpoint layouts/dtypes are
frozen in `revision_tests/scaling/cases.yaml`. Download and validate the full
matrix, then execute the CPU/I/O protocol (CUDA is deliberately disabled):

```bash
.venv/bin/python revision_tests/scaling/download_models.py
.venv/bin/python revision_tests/scaling/validate_protocol.py --check-models
.venv/bin/python revision_tests/scaling/run.py \
  --run-id eacl2027_scaling_linux \
  --repetitions 5 \
  --num-workers 1 \
  --workload-note "exclusive host; no concurrent user jobs"
```

All three methods and all ten model points must pass the independent oracle.
The GPU inventory is provenance only and no GPU performance claim is permitted
for this workload.

Use one hardware/filesystem configuration per reported comparison and retain
raw data under `log/revision_tests/<run_id>/`. Do not mix the Mac preflight
timings with Linux or CUDA measurements.

## 8. Return and submission audit

Transfer raw logs privately. Commit only compact audited summaries, generated
paper tables/text, protocols, and manifests. Before any anonymous reviewer
upload, remove hostnames, usernames, absolute paths, Git remotes, credentials,
and identifying repository metadata; then perform a human anonymity review.
Do not publish the 47 GB archive or base checkpoints as supplemental material.
