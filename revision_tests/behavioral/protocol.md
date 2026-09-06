# Behavioral regression protocol

Protocol identifier: `eacl2027_behavioral_v1`

Status: frozen before the first reported model run

Frozen model case: GPT-2 124M (`openai-community/gpt2`) at revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`, transformed by
`lossless_gpt2_plan.yaml`. The final checkpoint must contain the same 160
tensors byte-for-byte and use an indexed 256 MiB safetensors shard layout.

## Research question and scope

For a transformation declared lossless and independently verified at the
tensor level, does the transformed checkpoint reproduce the reference model's
observable next-token and greedy-generation behavior on a sourced,
stratified diagnostic suite?

This protocol is a behavioral regression check, not a downstream-quality or
general-capability benchmark. Axon and Synapse are out of scope. The protocol
does not execute HumanEval generations or infer broad multilingual competence.

## Prompt suite

The 70 prompts are selected by deterministic SHA-256 ranking, never by model
outputs:

- five parallel Belebele question identifiers are selected once and rendered
  in English, Danish, German, French, Spanish, and Simplified Chinese (30);
- five MMLU test questions are selected independently within each of abstract
  algebra, computer security, high-school world history, moral disputes,
  professional law, and global facts (30);
- ten HumanEval task identifiers are selected from the complete test set (10).

For each stratum, candidates are ordered by
`SHA256(selection_seed + NUL + source + NUL + stratum + NUL + upstream_id)`;
the first required number is retained. The selection seed, sources, revisions,
input-file hashes, strata, and sample sizes are frozen in `sources.yaml`.

MMLU prompts use zero-shot question/choice formatting and end in `Answer:`.
Belebele prompts use the source repository's documented zero-shot structure and
English instruction, with the passage, query, and choices in the row's
language. HumanEval prompts preserve the upstream function signature and
docstring. Text is normalized to Unicode NFC and LF line endings; no answer or
canonical solution is included in the visible prompt.

The complete rendered text is committed so a reported run never depends on a
mutable data host. Upstream answers are retained solely for descriptive
multiple-choice agreement and accuracy; no HumanEval functional score is
reported.

## Frozen execution settings

- model implementation: Hugging Face `AutoModelForCausalLM`;
- tokenizer: the exact pinned reference-model tokenizer for both checkpoints;
- model mode: `eval()` with `torch.inference_mode()`;
- device: one recorded CUDA device for both roles;
- dtype: explicit and identical for both roles (`float32`, `float16`, or
  `bfloat16`; never `auto` in a reported run);
- seed: 0 for Python and PyTorch, although decoding is greedy;
- deterministic algorithms: requested with
  `torch.use_deterministic_algorithms(True)`;
- prompt truncation: none; stop if any prompt exceeds the model context;
- batch size: 1;
- greedy continuation: 32 new tokens, `do_sample=false`;
- special tokens: tokenizer defaults recorded in run metadata;
- choice score: length-normalized conditional log-likelihood of ` A`, ` B`,
  ` C`, and ` D`, using the same tokenizer;
- next-token vector: float32 CPU copy of the final prompt-position logits.

The reference and transformed roles are executed separately to fit 8 GB GPUs.
Every result records the repository commit, exact command, model and tokenizer
paths/revisions, model config fingerprint, tokenizer-file fingerprint, package
versions, hardware, manifest checksum, and execution settings.

## Endpoints and decision rule

For a lossless transformation, all of these primary endpoints must pass for all
70 prompts:

1. the full next-token logit vector is byte-identical;
2. the next-token top-1 token ID is identical;
3. the 32-token greedy continuation ID sequence is identical; and
4. the predicted multiple-choice label is identical for all 60 applicable
   prompts.

The analyzer also reports maximum and mean absolute logit differences, cosine
similarity, reference and transformed multiple-choice accuracy by source and
stratum, and generated-text equality. These are secondary diagnostics. There
is no majority threshold: one mismatch fails the enumerated lossless suite.

Exact comparison is valid only when both roles use the same recorded hardware,
software, dtype, tokenizer, and deterministic settings. Cross-hardware results
are reported separately and are not judged with the byte-exact logit rule.

## Independence and negative controls

Prompt selection and the comparison analyzer do not import BrainSurgery. Model
execution loads already produced checkpoints through Transformers. Tensor-level
correctness remains a separate independent-oracle evaluation.

Synthetic tests must prove that the analyzer detects at least:

- one changed logit value;
- one changed generated token ID;
- one missing or substituted prompt ID; and
- incompatible run metadata.

No reported result is valid unless those controls pass.

## Reporting

Report counts alongside percentages: prompt comparisons, exact logit vectors,
top-1 matches, greedy-sequence matches, multiple-choice prediction matches,
and source/stratum coverage. Report multiple-choice accuracy descriptively and
with the small sample size visible. Do not describe 70 diagnostic prompts as a
general downstream evaluation.

Raw model outputs belong under `log/revision_tests/<run_id>/behavioral/`.
Commit only the protocol, prompt/source manifests, code, tests, and compact
summaries. Linux/CUDA results must remain separate from Mac smoke results.
