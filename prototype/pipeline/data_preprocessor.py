"""
data_preprocessor.py
====================
SOICT 2026 — Stage 0: Raw Log Ingestion & Dataset Preprocessing

Handles:
  - Legacy multi-dataset preprocessing prototype; not the source of paper metrics
  - BGL, HDFS (public loghub benchmarks)
  - UNSW-NB15, ToN_IoT (network intrusion datasets)

Output:
  - data/processed/{dataset}/train_sequences.npy
  - data/processed/{dataset}/test_sequences.npy
  - data/processed/{dataset}/train_labels.npy
  - data/processed/{dataset}/test_labels.npy
  - data/processed/{dataset}/templates.json
"""

import os
import re
import json
import time
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict

# ─── Custom similarity parser (not Drain3) ───────────────────────────────────

class DrainParser:
    """
    Lightweight, streaming Drain-style log parser.
    Groups log messages into templates by masking variable tokens.
    """
    MASK_TOKENS = [
        (r'\b(?:\d{1,3}\.){3}\d{1,3}\b',          '<IP>'),
        (r'\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b', '<UUID>'),
        (r'\b[0-9a-fA-F]{12,}\b',                  '<HEX>'),
        (r'\b\d{4}[-/]\d{2}[-/]\d{2}[T ]\d{2}:\d{2}:\d{2}(?:\.\d+)?Z?\b', '<TS>'),
        (r'\b\d{1,2}/\w+/\d{4}:\d{2}:\d{2}:\d{2}\b', '<TS>'),
        (r'\b\d+\.\d+\b',                           '<FLOAT>'),
        (r'\b\d{5,}\b',                             '<NUM>'),
        (r'/(?:home|var|usr|etc|tmp|proc)/\S+',    '<PATH>'),
    ]

    def __init__(self, sim_th=0.5, max_vocab=2000):
        self.sim_th = sim_th
        self.max_vocab = max_vocab
        self.templates: list[list[str]] = []
        self.template_to_id: dict[str, int] = {}
        self.counts: dict[int, int] = defaultdict(int)

    def _sanitize(self, msg: str) -> str:
        for pattern, replacement in self.MASK_TOKENS:
            msg = re.sub(pattern, replacement, msg)
        return msg.strip()

    def _similarity(self, a: list[str], b: list[str]) -> float:
        if len(a) != len(b):
            return 0.0
        matches = sum(1 for x, y in zip(a, b) if x == y)
        return matches / max(len(a), 1)

    def parse(self, raw_line: str) -> tuple[int, str]:
        sanitized = self._sanitize(raw_line)
        tokens = sanitized.split()
        if not tokens:
            return 0, "<EMPTY>"

        best_idx, best_sim = -1, -1.0
        for i, tmpl in enumerate(self.templates):
            sim = self._similarity(tokens, tmpl)
            if sim > best_sim and sim >= self.sim_th:
                best_sim, best_idx = sim, i

        if best_idx >= 0:
            merged = [a if a == b else '<*>' for a, b in zip(tokens, self.templates[best_idx])]
            self.templates[best_idx] = merged
            self.counts[best_idx] += 1
            tmpl_str = ' '.join(merged)
        else:
            if len(self.templates) >= self.max_vocab:
                best_idx = 0
            else:
                best_idx = len(self.templates)
                self.templates.append(tokens)
            self.counts[best_idx] += 1
            tmpl_str = ' '.join(tokens)

        if tmpl_str not in self.template_to_id:
            self.template_to_id[tmpl_str] = len(self.template_to_id) + 1  # 0 = PAD
        return self.template_to_id[tmpl_str], tmpl_str


# ─── Log File Processors ─────────────────────────────────────────────────────

def stream_huflit_logs(log_paths: list[Path], max_lines: int = 500000):
    """
    Streams HUFLIT raw campus syslog telemetry line by line.
    Yields: (is_anomaly: bool, clean_message: str)
    """
    count = 0
    pattern_anomaly = re.compile(r'\b(error|fail|denied|critical|refused|invalid|attack|flood|drop|reject|alert)\b', re.I)
    
    for file_path in log_paths:
        if count >= max_lines:
            break
        print(f"  • Reading stream: {file_path.name} ({file_path.stat().st_size / 1e6:.1f} MB)...")
        try:
            with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    label = bool(pattern_anomaly.search(line))
                    msg = re.sub(r'^\w{3}\s+\d+\s+\d{2}:\d{2}:\d{2}\s+\S+\s+', '', line)
                    yield label, msg
                    count += 1
                    if count >= max_lines:
                        break
        except Exception as e:
            print(f"    [!] Error reading {file_path.name}: {e}")

# ─── Sequence Builder ──────────────────────────────────────────────────────────

def build_sequences_from_stream(stream_generator,
                                parser: DrainParser,
                                window_size: int = 20,
                                step: int = 10):
    token_ids, raw_labels = [], []
    for label, msg in stream_generator:
        tid, _ = parser.parse(msg)
        token_ids.append(tid)
        raw_labels.append(int(label))

    sequences, seq_labels = [], []
    for i in range(0, len(token_ids) - window_size, step):
        window = token_ids[i:i + window_size]
        window_label = int(any(raw_labels[i:i + window_size]))
        sequences.append(window)
        seq_labels.append(window_label)

    return sequences, seq_labels

# ─── Main Execution ───────────────────────────────────────────────────────────

def main():
    parser_cli = argparse.ArgumentParser(description='SOICT 2026 Dataset Preprocessor')
    parser_cli.add_argument('--data_dir',  default='data', help='Root data directory')
    parser_cli.add_argument('--out_dir',   default='data/processed', help='Output directory')
    parser_cli.add_argument('--window',    type=int, default=20, help='Sliding window size')
    parser_cli.add_argument('--step',      type=int, default=10, help='Sliding window step')
    parser_cli.add_argument('--max_vocab', type=int, default=2000, help='Max template vocab size')
    parser_cli.add_argument('--max_lines', type=int, default=500000, help='Max log lines to process for benchmarking')
    args = parser_cli.parse_args()

    drain = DrainParser(sim_th=0.5, max_vocab=args.max_vocab)
    root = Path(args.data_dir) / "raw" / "huflit_logs"

    all_log_files = [f for f in root.rglob('*') if f.is_file() and f.suffix.lower() not in ('.zip', '.rar', '.gitkeep')]
    print(f"==========================================================================")
    print(" LEGACY PROTOTYPE: output is not paper evidence")
    print(f"==========================================================================")
    print(f" Total Log Files Available: {len(all_log_files)}")
    
    t0 = time.time()
    stream = stream_huflit_logs(all_log_files, max_lines=args.max_lines)
    seqs, labels = build_sequences_from_stream(stream, drain, args.window, args.step)
    
    seqs_arr = np.array(seqs, dtype=np.int32)
    labels_arr = np.array(labels, dtype=np.int8)

    huflit_out = Path(args.out_dir) / "huflit"
    huflit_out.mkdir(parents=True, exist_ok=True)

    split = int(len(seqs_arr) * 0.8)
    np.save(huflit_out / 'train_sequences.npy', seqs_arr[:split])
    np.save(huflit_out / 'test_sequences.npy',  seqs_arr[split:])
    np.save(huflit_out / 'train_labels.npy',    labels_arr[:split])
    np.save(huflit_out / 'test_labels.npy',     labels_arr[split:])

    # Save template vocabulary
    with open(huflit_out / 'templates.json', 'w', encoding='utf-8') as f:
        json.dump(drain.template_to_id, f, indent=2, ensure_ascii=False)

    elapsed = time.time() - t0
    anomaly_rate = float(labels_arr.mean() * 100) if len(labels_arr) > 0 else 0.0

    print(f"--------------------------------------------------------------------------")
    print(f" Preprocessing Summary:")
    print(f"  • Total Generated Sequences : {len(seqs_arr):,}")
    print(f"  • Train Sequences (80%)     : {split:,}")
    print(f"  • Test Sequences (20%)      : {len(seqs_arr) - split:,}")
    print(f"  • Anomaly Sequence Rate     : {anomaly_rate:.2f}%")
    print(f"  • Total Unique Templates    : {len(drain.template_to_id):,}")
    print(f"  • Execution Time            : {elapsed:.2f} seconds")
    print(f"  • Saved Files Location      : {huflit_out.resolve()}")
    print(f"==========================================================================")

if __name__ == '__main__':
    main()
