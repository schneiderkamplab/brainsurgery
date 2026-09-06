# Prompt-data attribution and licenses

The repository's MIT license does not replace the licenses of upstream prompt
text stored in `prompt_manifest.jsonl`. Each manifest row identifies its source
and license.

## Belebele

- Source: <https://github.com/facebookresearch/belebele>
- Revision: `918890beb2290a8d3ef2d7a90369925959e1bacf`
- Data archive SHA-256:
  `c645e750e111404806751509b5aa3f808a01e6029cd225d37e966dc694aca9e4`
- License: CC BY-SA 4.0
- Changes: deterministic subset selection, NFC/LF normalization, and insertion
  into the source's documented zero-shot prompt template.
- Citation: Bandarkar et al., *The Belebele Benchmark: a Parallel Reading
  Comprehension Dataset in 122 Language Variants*, ACL 2024.

## MMLU

- Original source: <https://github.com/hendrycks/test>
- Repository revision: `4450500f923c49f1fb1dd3d99108a0bd9717b660`
- Pinned CAIS Hugging Face data revision:
  `c30699e8356da336a370243923dbaf21066bb9fe`
- Test parquet SHA-256:
  `74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7`
- License: MIT
- Changes: deterministic stratified subset selection, NFC/LF normalization,
  and zero-shot multiple-choice rendering.
- Citation: Hendrycks et al., *Measuring Massive Multitask Language
  Understanding*, ICLR 2021.

## HumanEval

- Source: <https://github.com/openai/human-eval>
- Revision: `6d43fb980f9fee3c892a914eda09951f772ad10d`
- Data-file SHA-256:
  `b796127e635a67f93fb35c04f4cb03cf06f38c8072ee7cee8833d7bee06979ef`
- License: MIT
- Changes: deterministic subset selection and NFC/LF normalization. Canonical
  solutions and tests are not copied into the prompt manifest.
- Citation: Chen et al., *Evaluating Large Language Models Trained on Code*,
  2021.

The suite is diagnostic and deliberately small. Users redistributing adapted
Belebele text must comply with CC BY-SA 4.0 attribution and share-alike terms.
