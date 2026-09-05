# Condition P: Python / PyTorch baseline

You are given a checkpoint-editing task (TASK.md below). Solve it with a
standalone Python script using PyTorch and the `safetensors` package, the way
a practitioner writes a one-off state-dict script. Only the packages listed
in `requirements-P.txt` are installed in your environment. Do not use the
`brainsurgery` package or CLI, and do not install anything.

Your environment: this sandbox directory is your working directory and your
Python environment is private to this run (see "Environment" in TASK.md).
Inputs are under `inputs/`; the output must be written exactly where TASK.md
says, under `out/`.

Rules:

- Write your script to `out/<task>/solution.py` and run it with
  `python out/<task>/solution.py`. Every execution of the script counts as an
  attempt; do not test ideas in a REPL or with partial snippets.
- Your script must implement the "Required checks" in TASK.md and fail
  loudly if they do not hold.
- When you are done, write `out/<task>/REPORT.md` with the fields in
  `record-template.md`, section "Participant self-report".
