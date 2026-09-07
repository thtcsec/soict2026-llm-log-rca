"""
train_eval_soict.py - Full Master Benchmark Training & Evaluation Script for SOICT 2026
"""

import os
import sys
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

# Add prototype modules to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pipeline.tcn_transformer import TCNTransformerAutoencoder, calculate_anomaly_scores
from llm.rca_narration import LLMRCANarrationEngine

def run_pipeline():
    processed_dir = Path("data/processed/huflit")
    train_seqs_file = processed_dir / "train_sequences.npy"
    test_seqs_file = processed_dir / "test_sequences.npy"
    test_labels_file = processed_dir / "test_labels.npy"
    templates_file = processed_dir / "templates.json"

    if not train_seqs_file.exists():
        print(f"[!] Preprocessed data not found at {processed_dir}. Run data_preprocessor.py first.")
        return

    # 1. Load Data
    X_train = np.load(train_seqs_file)
    X_test = np.load(test_seqs_file)
    y_test = np.load(test_labels_file)

    with open(templates_file, "r", encoding="utf-8") as f:
        templates = json.load(f)

    vocab_size = max(templates.values()) + 2
    seq_len = X_train.shape[1]

    print("==========================================================================")
    print(" SOICT 2026 Master Benchmark: HUFLIT Campus Log Dataset")
    print("==========================================================================")
    print(f" Train Sequences : {len(X_train):,}")
    print(f" Test Sequences  : {len(X_test):,}")
    print(f" Anomaly Labels  : {np.sum(y_test):,} ({np.mean(y_test)*100:.2f}%)")
    print(f" Vocabulary Size : {vocab_size:,} templates")
    print(f" Sequence Length : {seq_len}")
    print("--------------------------------------------------------------------------")

    # 2. Model Training Simulation (TCN-Transformer)
    print("[Stage 1 & 2] Training TCN-Transformer Autoencoder Model...")
    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TCNTransformerAutoencoder(vocab_size=vocab_size, embed_dim=64, num_heads=4, hidden_dim=128, seq_len=seq_len).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Convert to tensors
    train_tensor = torch.tensor(X_train, dtype=torch.long).to(device)
    test_tensor = torch.tensor(X_test, dtype=torch.long).to(device)

    # 3 Epochs fast benchmark fit
    model.train()
    batch_size = 256
    for epoch in range(1, 4):
        permutation = torch.randperm(train_tensor.size(0))
        epoch_loss = 0.0
        for i in range(0, train_tensor.size(0), batch_size):
            indices = permutation[i:i+batch_size]
            batch = train_tensor[indices]
            
            optimizer.zero_grad()
            logits = model(batch)
            loss = nn.CrossEntropyLoss()(logits.view(-1, vocab_size), batch.view(-1))
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"  Epoch {epoch}/3 | Loss: {epoch_loss / (train_tensor.size(0)/batch_size):.4f}")

    train_time = time.time() - t0

    # 3. Model Inference & Evaluation
    print("\n[Stage 2 Evaluation] Running Model Inference on Test Set...")
    model.eval()
    t_inf_start = time.time()
    with torch.no_grad():
        test_logits = model(test_tensor)
        scores = calculate_anomaly_scores(test_logits, test_tensor).cpu().numpy()
    inf_time = time.time() - t_inf_start
    per_seq_latency_ms = (inf_time / len(X_test)) * 1000

    # Dynamic threshold tuning
    threshold = np.percentile(scores, 100 - (np.mean(y_test) * 100))
    y_pred = (scores > threshold).astype(int)

    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, scores)

    print("--------------------------------------------------------------------------")
    print(" Anomaly Detection Metric Results:")
    print(f"  • Precision   : {prec:.4f}")
    print(f"  • Recall      : {rec:.4f}")
    print(f"  • F1-Score    : {f1:.4f}")
    print(f"  • ROC-AUC     : {auc:.4f}")
    print(f"  • Avg Latency : {per_seq_latency_ms:.4f} ms / sequence")
    print("--------------------------------------------------------------------------")

    # 4. Stage 3: Foundation Model RCA Narration Execution
    print("\n[Stage 3] Executing Foundation Model RCA Narration Engine (Phi-3 / Mistral)...")
    llm_engine = LLMRCANarrationEngine()
    
    # Pick top anomalous sequence
    max_idx = np.argmax(scores)
    anom_seq_ids = X_test[max_idx]
    
    # Map back to templates
    id_to_template = {v: k for k, v in templates.items()}
    anom_templates = [id_to_template.get(tid, f"<Template_{tid}>") for tid in anom_seq_ids]

    meta = {"location": "HUFLIT Campus Core Server Gateway", "services": "Active Directory / Fortinet Firewall"}
    rca_res = llm_engine.generate_rca(anom_templates[:5], float(scores[max_idx]), meta)

    print("--------------------------------------------------------------------------")
    print(" Generated Root Cause Analysis (RCA) Sample Output:")
    print(json.dumps(rca_res, indent=2, ensure_ascii=False))
    print("==========================================================================")

    # 5. Save Results to CSV & Table File
    results_dir = Path("results/tables")
    results_dir.mkdir(parents=True, exist_ok=True)
    res_path = results_dir / "legacy_evaluation_summary.json"
    
    summary_data = {
        "dataset": "LEGACY prototype output — not paper evidence",
        "total_test_sequences": int(len(X_test)),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
        "roc_auc": float(auc),
        "latency_ms_per_seq": float(per_seq_latency_ms),
        "sample_rca": rca_res
    }
    
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, ensure_ascii=False)

    print(f"[+] Empirical evaluation summary saved to: {res_path.resolve()}")

if __name__ == "__main__":
    run_pipeline()
