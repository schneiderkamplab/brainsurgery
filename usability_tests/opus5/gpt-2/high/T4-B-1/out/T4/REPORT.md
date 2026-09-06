# T4 — Participant self-report

- **Final artifact path:** `out/T4/plan.yaml` (output written to
  `out/T4/model.safetensors`). `out/T4/verify.yaml` is a separate
  read-only plan that re-checks the written file; it is not part of the
  solution.

- **Number of times you executed the script or plan:** 1 execution of
  `out/T4/plan.yaml`, which succeeded. Before writing it I ran 5 throwaway
  exploration plans (`explore*.yaml`, since deleted) to dump the tensor
  names, confirm the regex/rewrite semantics of `subtract_`, `add_`,
  `scale_` and `assert equal` with a lookahead pattern, and to confirm that
  a false assert aborts with exit code 1. Afterwards I ran `verify.yaml`
  once. None of these wrote a checkpoint.

- **Which executions failed, and why (one line each):** none — first
  execution of the plan succeeded.

- **Pitfalls or surprises you hit (one line each):**
  - The ordering hazard is real but easy to avoid in a plan: both task
    vectors have to be materialised *and* scaled before the first `add_`
    touches the base, otherwise the second `subtract_` reads an already
    merged base.
  - `add_`/`subtract_`/`scale_` all work in place on the destination, so
    the task vectors need scratch tensors; I put them in the `base` alias
    under `tv1.` / `tv2.` prefixes and deleted them before the output, both
    to keep the output at 160 tensors and because the output alias is
    inferred from the alias the transforms write to (writing scratch into
    a second alias would have made that ambiguous).
  - Scratch names had to be chosen so the MLP pattern `h\.\d+\.mlp\..*`
    could not match them; since references are full-match regexes, a
    `tv1.` prefix is enough, but `.*`-style counting patterns do see the
    scratch tensors, so the 160-tensor check has to come after the delete.
  - `assert equal` resolves `right` as a rewrite of each `left` match and
    fails if the right-hand tensor does not exist, so a single
    `left: 'ft1::(?!h\.\d+\.mlp\.).+', right: 'base::\g<0>'` covers both
    "same names" and "identical values" for the 112 shared tensors. That
    lookahead-plus-`\g<0>` idiom is documented in the README, which saved
    a lot of guessing.
  - Nothing needed a dtype cast: all three checkpoints are float32, which
    the plan asserts rather than assumes.

- **Anything in the task text or documentation that was unclear:**
  - "verify that the three checkpoints have the same tensor names" — for
    the 112 non-MLP tensors the equality assert proves name identity as a
    side effect, but there is no assert operator that compares two name
    *sets* across aliases directly, so for the 48 MLP names I had to prove
    it indirectly (equal per-alias counts and shapes, plus the fact that
    the `subtract_`/`add_` rewrites fail loudly if a counterpart name is
    missing). A `names_equal`-style assert, or a `diff` that can be made
    to fail, would express step 1 more directly.
  - "exactly 48 tensors were merged" has no direct counter either; I
    asserted that each task vector consists of exactly 48 tensors, which
    is the same thing given that the merge is driven by those tensors.
  - The README's doc links point at an absolute path on someone else's
    machine (`/Users/petersk/...`), which is dead in the sandbox.

- **Tools used (condition F):** n/a — condition B, only the `brainsurgery`
  CLI and its YAML plans.

- **Approximate time spent, if you can tell:** about 10 minutes, roughly
  half of it reading `docpack/README.md` and the `help` output.
