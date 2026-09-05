# brainsurgery
Swiss army knife for scripted tensor surgery on model checkpoints.

## Installation
```bash
pip install brainsurgery
```

## Developer workflow
Install development tooling:

```bash
pip install -e ".[dev]"
```

Enable git hooks:

```bash
pre-commit install
```

Run hooks manually:

```bash
pre-commit run --all-files
```

Run mypy on demand (manual hook):

```bash
pre-commit run mypy --all-files
```

## Quick start
Run a YAML plan:

```bash
brainsurgery examples/gpt2.yaml
```

Run multiple config inputs (deep-merged in order), then CLI overrides:

```bash
brainsurgery base.yaml override.yaml transforms[0].dump.format=compact
```

Enable interactive mode after configured transforms:

```bash
brainsurgery -i examples/gpt2.yaml
```

Launch the browser-based Web UI:

```bash
brainsurgery webui
```

## Documentation
Detailed project documentation is in `docs/`:

- [Documentation Index](/Users/petersk/Nobackup/brainsurgery/docs/README.md)
  - Jump point for all project documentation pages.
- [Codebase Reference](/Users/petersk/Nobackup/brainsurgery/docs/codebase-reference.md)
  - Architecture, core classes/protocols, execution processes, extension points, and module map.
- [Interfaces Reference](/Users/petersk/Nobackup/brainsurgery/docs/interfaces-reference.md)
  - Batch CLI, interactive CLI, OLY, webcli, webui, and when to use each interface.
- [OLY Specification](/Users/petersk/Nobackup/brainsurgery/docs/oly-spec.md)
  - One-line command grammar and mapping semantics.
- [OLY YAML Grammar](/Users/petersk/Nobackup/brainsurgery/docs/oly-yaml-grammar.md)
  - Exact YAML data model representable through OLY.
- [OLY Conformance Matrix](/Users/petersk/Nobackup/brainsurgery/docs/oly-conformance.md)
  - Acceptance/rejection cases and round-trip guarantees.
- [SYNAPSE/1 Specification](/Users/petersk/Nobackup/brainsurgery/docs/synapse-spec.md)
  - Declarative high-level model assembly language for PyTorch-oriented architectures.

## CLI
`brainsurgery [OPTIONS] [CONFIG_ITEMS]...`

`CONFIG_ITEMS` can be:
- YAML files (`.yaml`, `.yml`) loaded and deep-merged in order.
- `key=value` overrides applied last (supports dotted keys and list indices like `transforms[0].dump.format=tree`).

Options:
- `--shard-size TEXT` (default: `5GB`) default shard size for directory outputs.
- `--num-workers INTEGER` (default: `8`) max parallel I/O workers.
- `--provider TEXT` (default: `inmemory`) one of `inmemory`, `arena`.
- `--arena-root PATH` (default: `.brainsurgery`) arena storage root.
- `--arena-segment-size TEXT` (default: `1GB`) segment size for arena provider.
- `-i, --interactive` run configured transforms, then prompt for extra transforms.
- `-s, --summarize / --no-summarize` (default: summarize) emit executed transform summary.
- `--summarize-path PATH` write summary to file (otherwise prints YAML to stdout).
- `--log-level TEXT` one of `debug`, `info`, `warning`, `error`, `critical`.
- `--install-completion` install shell completion for current shell.
- `--show-completion` print completion script for current shell.

## Plan format
```yaml
inputs:
  - model::/path/to/input.safetensors   # alias::path
  # with a single input, alias is optional and defaults to "model"

transforms:
  - dump: { target: ".*", format: compact }
  - copy: { from: ln_f.weight, to: ln_f_copy.weight }

output:
  path: /path/to/output                  # file or directory
  format: safetensors                    # optional: safetensors|torch
  shard: 5GB                             # optional (safetensors only)
```

Notes:
- `output` is optional. If omitted, no final checkpoint is written.
- Transforms are always a list of single-key mappings.

## Tensor references
Most transforms use tensor references:
- `alias::expr`
- `expr` (uses default model alias when available)
- `alias::expr::[slice]`

`expr` supports:
- Regex string matching (full-match).
- Structured path patterns (list form), matched against dot-separated tensor names.

Slices follow Python-like syntax in brackets, e.g. `[:8]`, `[:, :10]`, `[1:3, :]`.

## Structured expressions
Structured expressions are list tokens (instead of regex strings), for example:

```yaml
from: ["block", "$i", "weight"]
to:   ["backup", "${i}", "weight"]
```

Supported source tokens:
- `literal` exact segment
- `$x` capture one segment
- `*xs` capture zero or more segments
- `~REGEX` regex on one segment
- `~x::REGEX` bind one capture
- `~x,y::REGEX` bind multiple captures

Output tokens:
- literals (with `${x}` interpolation)
- `*xs` variadic splice

Equivalent command forms:
- YAML: `copy: { from: ln_f.weight, to: ln_f_copy.weight }`
- OLY: `copy: from: ln_f.weight, to: ln_f_copy.weight`
- YAML: `help: { assert: equal }`
- OLY: `help: assert: equal`

## Batch vs interactive mode
Batch mode (default):
- Executes configured transforms in order.
- Any transform failure aborts execution (raises error).

Interactive mode (`-i`):
- First executes configured transforms in batch mode.
- Then opens prompt for additional transform blocks (YAML or OLY).
- On failure in an interactive block, logs error, stops only that submitted block, and returns to prompt.
- `exit` transform cleanly stops the execution loop.
- Interactive prompt uses a richer UI and supports tab completion for command names, payload keys, model aliases, and loaded tensor names.

Interactive prompt accepts:
- a single transform mapping (YAML or OLY), or
- a YAML list of transform mappings.

Special transforms:
- `help`
- `prefixes`
- `exit`

History is stored in `~/.brainsurgery_history`.

## Commands (transforms)
All registered transforms (see [Interfaces Reference](/Users/petersk/Nobackup/brainsurgery/docs/interfaces-reference.md) and command-specific `help` output for operational details):

- `help`: show command/assert help (`help`, `help: copy`, `help: assert`, `help: { assert: equal }`).
  OLY shorthand also works (for example `help: assert: equal`).
- `prefixes`: list or manage model prefixes (`alias::`):
  `mode=list` (default), `mode=add` with `alias`, `mode=remove` with `alias`,
  `mode=rename` with `from` + `to`.
- `exit`: stop current execution loop.
- `dump`: print tensor summaries (`format`: `json|tree|compact`, `verbosity`: `shape|stat|full`).
- `diff`: compare two tensor sets by name, reporting missing-on-left, missing-on-right, and differing tensors (`mode=refs` with `left`/`right`, or `mode=aliases` with `left_alias`/`right_alias`; optional `eps`).
- `assert`: evaluate assertion expressions; does not modify tensors.
- `load`: load full state_dict (`path`, optional `alias`) or single tensor (`path` + `to`, optional `format`).
- `save`: save full state_dict (`path`, optional `alias`, `format`, `shard`) or single tensor (`target`).
- `copy`: clone source tensor(s) to new destination(s) (destination must not exist).
- `move`: move tensor slot(s) without copy (no slicing; destination must not exist).
- `delete`: delete tensor(s) by target pattern.
- `assign`: copy values into existing destination tensor(s), shape/dtype/device must match.
- `add`: `to = from_a + from_b` into existing destination(s).
- `subtract`: `to = from_a - from_b` into existing destination(s).
- `multiply`: `to = from_a * from_b` into existing destination(s).
- `matmul`: `to = from_a @ from_b` into new destination(s).
- `add_`: in-place `to += from`.
- `subtract_`: in-place `to -= from`.
- `scale`: create new tensor(s): `to = from * by`.
- `scale_`: in-place scaling of target(s).
- `cast`: create new tensor(s) with new dtype.
- `cast_`: in-place dtype cast.
- `clamp`: create new tensor(s) clamped by `min`/`max`.
- `clamp_`: in-place clamp.
- `reshape`: create new reshaped tensor(s) (`shape` allows one `-1`).
- `reshape_`: in-place reshape/rebind.
- `permute`: create new tensor(s) with reordered dimensions (`order` permutation).
- `fill`: create new tensor(s) using source shape:
  - `mode=constant` + `value`
  - `mode=rand` (+ `distribution`, `low/high` or `mean/std`, optional `seed`)
  - `mode=tensor` + `values` (broadcast allowed)
- `fill_`: in-place fill with same modes as `fill`.
- `zeroes`: create new zero-filled tensor (`target`, `shape`).
- `ones`: create new one-filled tensor (`target`, `shape`).
- `rand`: create new random tensor (`target`, `shape`, optional `distribution`, params, `seed`).
- `split`: split one source tensor into multiple new tensors (`to` list, `sizes`, optional `dim`).
- `concat`: concatenate multiple source refs into one new tensor (`from` list, optional `dim`).
- `phlora_`: in-place low-rank reconstruction (`rank`) on 2D tensors.
- `phlora`: split 2D target tensors into low-rank factors (`target_a`, `target_b`, `rank`);
  optional `delete_original` (default `true`), `require_missing_dest` (default `true`).

## Assert expressions
Used via:

```yaml
- assert: { equal: { left: a.weight, right: b.weight, eps: 1e-6 } }
```

OLY equivalent:

```text
assert: equal: { left: a.weight, right: b.weight, eps: 1e-6 }
```

Supported operators:
- `exists`: reference matches at least one tensor.
- `count`: exact number of matches (`of`, `is`).
- `shape`: tensor shape check (`of`, `is` list of ints).
- `dimensions`: rank comparison (`of`, any of `is|ge|gt|le|lt`).
- `dtype`: dtype check (`of`, `is` dtype string).
- `equal`: pairwise equality between mapped tensors (`left`, `right`, optional `eps`).
  `right` is resolved as a rewrite of each `left` match, exactly like `to` in `copy`/`move`:
  capture groups from `left` can be used in `right` (`\1`, `\g<0>`), and aliases may differ.
  `assert: { equal: { left: 'a::(.+)', right: 'b::\1' } }` checks every tensor of alias `a`
  against the same name in alias `b`; `left: 'a::(?!h\.\d+\.mlp\.).+', right: 'b::\g<0>'`
  does the same for the tensors outside `h.<i>.mlp.*`. Every `left` match must map to an
  existing `right` tensor.
- `iszero`: all-zero check (`of`, optional `eps`).
- `reads`: access-count comparison for instrumented backends (`of`, any of `is|ge|gt|le|lt`; `at_least`/`at_most` accepted as compatibility aliases).
- `writes`: access-count comparison for instrumented backends (`of`, any of `is|ge|gt|le|lt`; `at_least`/`at_most` accepted as compatibility aliases).
- `all`: all nested assertions succeed.
- `any`: at least one nested assertion succeeds.
- `not`: nested assertion fails.

## Output behavior
- If `output` is omitted: no final checkpoint is saved.
- If `output` is a directory-like path (or no suffix), safetensors output defaults to sharded writing using `--shard-size` unless shard is disabled.
- `torch` output requires file suffix `.pt`, `.pth`, or `.bin`.
- `output.shard` and `save.shard` are safetensors-only.
- Which alias is written: with one input, its alias (`model` by default). With several
  inputs, the alias is inferred as the single alias that the transforms write to
  (`copy`/`move`/`fill`/... destinations, in-place targets such as `scale_`, `cast_`,
  `add_`, and `delete`). `assert`, `diff`, `dump` and `help` do not count. If the transforms
  write to more than one alias, or to none, the run fails with
  `cannot infer output model uniquely`; keep every edit on one alias (use `alias::` on
  destinations, e.g. copy from `ft::...` to `base::...`) so the output is unambiguous.
- Shard sizes (`--shard-size`, `output.shard`, `save.shard`) accept `KB`, `MB`, `GB`, `TB` as
  binary units: `100MB` = 100 x 1024 x 1024 = 104,857,600 bytes. The budget counts tensor
  data only (file headers are extra). Tensors are packed into shards in state-dict order
  up to the budget; a single tensor larger than the budget is written alone in its own
  shard. Sharded output produces `model-00001-of-0000N.safetensors` files plus
  `model.safetensors.index.json` whose `weight_map` maps every tensor name to its shard.

## Summary output
When summarize is enabled (default), brainsurgery emits the exact transforms that actually ran:
- to stdout by default
- or to `--summarize-path`

Useful for reproducibility when interactive edits or early `exit` changed execution.
