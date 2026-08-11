#!/bin/bash
set -e

echo "=========================================================================="
echo " SOICT 2026: LLM-Augmented Log Anomaly Detection & RCA Master Pipeline"
echo "=========================================================================="

echo "[1/4] Running Drain3 Log Parsing Demonstration..."
python prototype/drain3/log_parser.py

echo "[2/4] Testing Hybrid TCN-Transformer Autoencoder Model..."
python prototype/pipeline/tcn_transformer.py

echo "[3/4] Generating Foundation Model RCA Narration Sample..."
python prototype/llm/rca_narration.py

echo "[4/4] Pipeline Check Complete! Output logs saved."
echo "=========================================================================="
