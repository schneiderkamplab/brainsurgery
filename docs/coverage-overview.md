# Coverage Overview

This document summarizes current repository coverage counts for built-in Axon code, generic model Axon code, and checkpoint pragmas.

## Methodology

All counts were computed from files under:

- `brainsurgery/synapse/builtins/*.axon`
- `brainsurgery/synapse/models/*/generic-*.axon` (excluding `*-basic.axon`)

Line counting rules (for code-line coverage metrics):

- Do **not** count blank lines.
- Do **not** count comment lines (`-- ...`).
- Do **not** count `import ...` lines.
- Do **not** count `export ...` lines.
- For model generic-file counting, do **not** count pragma lines (`{-# ... #-}`).

Checkpoint coverage rules:

- Read `{-# CHECKPOINTS ... #-}` pragmas in `brainsurgery/synapse/models/*/generic-*.axon` (excluding `*-basic.axon`).
- Parse payload as either:
  - a single string checkpoint id, or
  - a list/tuple of string checkpoint ids.
- Report both total mentions and unique checkpoint ids.

## Results

### Built-ins code lines

Scope:

- `brainsurgery/synapse/builtins/*.axon`

Result:

- **1052** counted lines

### Generic model code lines

Scope:

- `brainsurgery/synapse/models/*/generic-*.axon`
- Excluding `generic-*-basic.axon`

Result:

- Files counted: **72**
- Counted lines: **4508**

### Checkpoint pragma coverage

Scope:

- `brainsurgery/synapse/models/*/generic-*.axon` (excluding `*-basic.axon`)

Result:

- Model files scanned: **72**
- Files with `CHECKPOINTS` pragma: **72**
- Checkpoint mentions (all pragma entries): **245**
- Unique checkpoints covered: **241**
