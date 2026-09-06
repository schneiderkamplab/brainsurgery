# Downstream quality

This area evaluates whether intentionally lossy transformations preserve or
improve useful model behavior. Candidate cases are PHLoRA factorization, MoE
upcycling, and low-rank rewriting.

Every case must specify:

- the scientific hypothesis and expected direction;
- the unmodified checkpoint baseline;
- task, dataset, split, prompt/template, and metric;
- decoding and random-seed settings;
- exact model revision and transformation plan;
- uncertainty or repeated-run treatment;
- compute budget and a stop condition.

Do not use downstream results as evidence for lossless checkpoint equivalence.
If the evaluation cannot be run convincingly within the revision budget,
narrow the corresponding paper claim instead.
