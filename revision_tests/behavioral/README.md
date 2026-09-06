# Behavioral regression evaluation

This area replaces an undocumented set of 50 prompts with a versioned prompt
manifest and a stated sampling procedure.

The manifest should record source, license, stable identifier, language, task
category, split, and any filtering or normalization. The protocol should define
model revision, tokenizer, prompt template, decoding settings, random seeds,
comparison metric, tolerance or decision rule, and exclusions.

Behavioral agreement complements tensor-level correctness. It is not a
substitute for an independent checkpoint oracle, and exact agreement under
deterministic decoding should not be described as broad downstream quality.
