# Condition B: BrainSurgery plan

You are given a checkpoint-editing task (TASK.md below). Solve it with a
BrainSurgery plan: a YAML file with `inputs`, `transforms` and `output`,
executed with the `brainsurgery` command-line tool. Do not write Python; the
whole edit must be expressed in the plan. Reading the tool's documentation
and running `brainsurgery` with `help` transforms is allowed and encouraged.

Documentation available to you (the doc pack):

- `docpack/README.md` (BrainSurgery README: plan format, tensor references,
  transform list, assert operators, output behavior)
- `docpack/interfaces-reference.md`
- `docpack/help.txt` (the built-in `help` output for every transform and
  assert expression)
- `docpack/examples/` (worked example plans unrelated to the tasks)

Your environment: this sandbox directory is your working directory and your
Python environment is private to this run (see "Environment" in TASK.md).
Only `brainsurgery` and its dependencies are installed. Inputs are under
`inputs/`; the output must be written exactly where TASK.md says, under
`out/`.

Rules:

- Write your plan to `out/<task>/plan.yaml` and run it with
  `brainsurgery out/<task>/plan.yaml` (CLI options such as `--provider` are
  allowed). Every execution of the plan counts as an attempt.
- Your plan must implement the "Required checks" in TASK.md as `assert`
  transforms, so the run fails if they do not hold.
- When you are done, write `out/<task>/REPORT.md` with the fields in
  `record-template.md`, section "Participant self-report".
