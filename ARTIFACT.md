# Artifact evaluation guide

**Paper:** *Campus Log Anomaly Detection under Proxy Labels: Validity Limits of Archive Splits and Masked TCN–Transformers*

**Authors (LNCS author block):** Hoang Tu Trinh, Thanh Tien Cao, and Ha Manh Tran

**Venue:** SOICT 2026 (4–5 December 2026). Accepted; Springer CCIS forthcoming. No DOI is claimed.

Evaluation validity is the primary object of study. This artifact does **not** claim operational incident detection, Drain3, GraphWalk, LLM/RCA measurement, or BGL/HDFS results.

## Claims supported by this artifact

1. Selective HUFLIT campus backup extracts under HTTP/keyword **proxy** labels — not SOC ground truth.
2. Protocol A (archive-order stream split with validation contamination) versus Protocol B (exact-line dedup, source-bounded windows, backup-day holdout).
3. OR-window labels inflate prevalence (Protocol A test rate 80.5%; always-anomaly F1 ≈ 0.892).
4. On these extracts the evaluated masked TCN–Transformer does not add a reliable accuracy gain over a masked Transformer without TCN, while increasing leave-one-out scoring latency.
5. Companion to CSoNet 2026: custom similarity parser (not Drain3); no reuse of CSoNet tables or GraphWalk claims.

Headline numbers in the manuscript (mean±std): TCN–Transformer F1 **0.0395±0.0163**, AUC **0.7226±0.0178**; masked Transformer without TCN F1 **0.3475±0.4097**, AUC **0.7685±0.0351**.

## Frozen dumps

Tracked metric tables live under `results/tables/` (JSON). Raw campus logs are **not** released.

## Evaluation steps

1. `python -m pip install -r requirements.txt`
2. Without raw logs, inspect `results/tables/` against the manuscript.
3. With processed arrays only:

```bash
python prototype/pipeline/run_baselines_huflit.py --threads 2
```

4. Source manuscript: `paper/soict2026.tex`. The compiled PDF is not published in this repository.

`prototype/llm/` is an unevaluated prompt/schema mock and is not a measured result.
