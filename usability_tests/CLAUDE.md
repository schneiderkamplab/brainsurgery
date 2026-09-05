# usability_tests/CLAUDE.md

@AGENTS.md

Claude Code specifics for the experimenter role:

- Use the repository interpreter `.venv/bin/python` from the repository root
  for every kit script; the kit imports `targets.py` from its own directory.
- `run_claude.py` drives participant sessions through `claude -p` with
  `--output-format stream-json`; it bypasses permissions inside the sandbox,
  so only launch it on sandboxes created by `make_sandbox.py`.
- When the user asks for results, run `analyze.py` and report its table;
  do not recompute rates by hand from individual files.
- Commit only when asked. Behaviour changes to `brainsurgery/*` still need
  approval per the repository `AGENTS.md`; this kit changes nothing there.
