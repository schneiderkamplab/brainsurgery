# TODO

- Fix `rope_pair_base` typing in `brainsurgery/synapse/builtins/Prelude.axon` to allow different query/key head counts (`HQ` vs `HK`) while keeping shared `B`, `T`, and `HD`.
- Re-run the previously failing Qwen3 pairs after the signature update and verify no regressions in `masked_top1_eq` / `masked_max_abs_diff`.
