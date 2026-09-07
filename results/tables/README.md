# Result provenance

`huflit_baselines.json` is the only detector-results file used by the current
manuscript and measured figures. Masked models in it must be produced by the
corrected token-replacement objective and leave-one-position-out scoring.

`huflit_hardened_results.json`, `evaluation_summary.json`, and
`real_multidataset_benchmark.json` are retained only as deprecated historical
outputs. They must not be cited: the first two used a loss mask without replacing
input tokens, and the multi-dataset file belongs to an earlier, non-comparable
harness.

The persisted HUFLIT arrays use heuristic proxy labels and a stream-position
split. Repeated records across backup archives mean the split is not leakage-free.
