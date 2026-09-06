# Scaling result summaries

Only compact summaries from a completed, audited Linux run belong here. Raw
commands, outputs, tensor checks, and resource samples remain under
`log/revision_tests/<run_id>/scaling/` or in private archival storage.

Mac/synthetic preflights must remain visibly labelled non-reportable and must
not supply timing or memory values to the paper.

## Linux reported candidate

- [`linux_2dbcd50/`](linux_2dbcd50/paper_table.md): ten checkpoints, three methods, and 150/150 correct measured attempts for run `eacl2027_scaling_linux_2dbcd50` at commit `2dbcd505115100f892e906413076ae93b3fcaa16`.
- The compact `summary.json` omits raw per-file and per-tensor manifests; those remain in `log/revision_tests/eacl2027_scaling_linux_2dbcd50/scaling/`.
- All automated reportability gates passed. The result is CPU/I/O evidence; GPU inventory is provenance only.
