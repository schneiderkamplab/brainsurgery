---
paths:
  - "docs/**"
  - "wiki/**"
  - "README.md"
---

# Docs and wiki rules

@../../docs/AGENTS.md

## docs/

- Jointly owned by user and agents. Keep edits scoped to the requested change;
  do not rewrite sections opportunistically or change tone.
- Commands must be copy-paste-safe. Prefer explicit flags over implicit defaults.
- If behavior differs by backend or runtime, say so explicitly.
- Renaming or removing major docs, or changing normative policy language, needs approval.

## wiki/

Operating model is in `wiki/AGENTS.md`. Summary:

- `wiki/` is durable agent memory. Put recurring failure classes, benchmark
  protocols, script contracts, and settled decisions there. Not transient output.
- Dated observations go in `wiki/log.md`. Promote repeated observations into a
  topic page rather than appending duplicates.
- Topic pages carry the header `status / last-confirmed / owners / confidence`.
- Do not silently delete contradicted facts. Mark them `superseded-by`,
  `fixed-by`, or `obsolete as of YYYY-MM-DD`.
- Every non-AGENTS wiki page must be listed in `wiki/index.md`. Every maintained
  script in `scripts/` must be documented in `wiki/scripts.md`.
- Never write API keys, tokens, or private data into the wiki.
