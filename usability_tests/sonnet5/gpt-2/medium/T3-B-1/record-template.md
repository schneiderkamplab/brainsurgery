# Run record

One record per (model, condition, task, repeat). The experimenter fills the
"Harness record" from logs; the participant fills the "Participant self-report"
into `out/<task>/REPORT.md` at the end of the run. Keep both; they are
compared.

## Harness record (`harness.json`)

Measures the study reports, per run:

| Field | Value |
|---|---|
| run_id / agent / target / effort / test / condition / repeat | `<test>-<condition>-<repeat>` under `<agent>/<target>/<effort>/`; effort is `low`/`medium`/`high` (Claude Code) or `light`/`medium`/`high` (OpenAI) |
| model_id | exact model identifier used by the driver |
| started_at / finished_at / wall_clock_s | solve phase; wall clock is the time to solution when the run passes |
| turns / tool_calls | assistant turns and tool invocations |
| tokens_in / tokens_out / cache_read_tokens / cache_write_tokens / tokens_in_total | from the provider's usage report; `tokens_in` is uncached input only, `tokens_in_total` adds cache reads and writes and is what the analysis reports |
| cost_usd | from the provider (Claude Code reports it) or computed from the vendor rate card; record which |
| executions | number of times the script, tool command or plan was executed (retries = executions - 1) |
| failed_executions | one row per failed execution: number, command, error_class, first error line |
| first_execution_success | yes / no |
| executions_until_first_success | integer, or "never" |
| cap_hit | none / turns / time / budget |
| final grade | `grade.json`: PASS / FAIL with findings (success rate is aggregated over runs) |
| review | `review.json`: bug-detection phase, see below |
| sandbox | `usability_tests/<agent>/<target>/<effort>/<test>-<condition>-<repeat>/` |
| env_fingerprint | `env-freeze.txt` in the sandbox |
| tools_used | condition F only: tools and versions the participant reports using |
| artifact_loc | non-blank, non-comment lines of the final script or plan |

Per failed execution, one row:

| execution # | error_class | message (first line) |
|---|---|---|

Error classes (pick one per failed execution):

- `crash`: exception in the participant's code, or a plan that does not parse or compile
- `failed_assertion`: a check the participant wrote fired
- `no_match`: a pattern or name matched nothing
- `overmatch`: a pattern matched more tensors than intended (detected by a check or by grading)
- `dest_exists`: destination tensor or file already existed
- `shape_dtype`: shape or dtype mismatch reported by the tool or the library
- `save_error`: failure while writing the output (non-contiguous or shared tensors, sharding, path)
- `partial_output`: the run failed after writing some output files
- `other`: describe in the message column

Pitfalls (free text, one line each, classified by the experimenter after the
run): what the participant misunderstood or had to discover, e.g. regex
overreach onto `mlp.c_proj`, unescaped dots, mask buffers named `attn.bias`,
shared-memory tensors rejected by safetensors, oversized tensor vs shard
budget, Conv1D `[in, out]` layout vs Linear `[out, in]`, PEFT name prefixes,
renumbering collisions when moving layers in the wrong order, provider choice.

## Bug-detection record (`review.json`)

After the solve phase, a fresh single-turn session of the same model reads the
task specification and one artifact for the same task (the defective version
from `review/<target>/` on odd repeats, the correct reference on even repeats)
and states whether it meets the specification and what is wrong.

| Field | Value |
|---|---|
| artifact_kind | defective / correct |
| verdict_text | the model's answer |
| detected | experimenter-confirmed: for a defective artifact, true if the stated problem matches `expected_defect`; for a correct artifact, true means a false alarm |
| tokens_in / tokens_out / cost_usd | for the review call |

## Participant self-report

Written by the participant to `out/<task>/REPORT.md` when done.

- Final artifact path:
- Number of times you executed the script or plan:
- Which executions failed, and why (one line each):
- Pitfalls or surprises you hit (one line each):
- Anything in the task text or documentation that was unclear:
- Tools used (condition F): name, version, and why:
- Approximate time spent, if you can tell:
