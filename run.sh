#!/bin/bash
set -e

echo "=========================================================================="
echo " SOICT 2026: safe processed-array verification (no raw/FortiGate ingest)"
echo "=========================================================================="

echo "[1/3] Testing masked TCN-Transformer implementation..."
python prototype/pipeline/tcn_transformer.py

echo "[2/3] Showing the RCA design-only schema..."
python prototype/llm/rca_narration.py

echo "[3/3] To rerun measured baselines from existing .npy arrays:"
echo "python prototype/pipeline/run_baselines_huflit.py --threads 2"
echo "Raw ingest is intentionally not run. FortiGate is excluded by default there."
echo "=========================================================================="
