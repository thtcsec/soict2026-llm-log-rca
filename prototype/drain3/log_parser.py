"""
log_parser.py - Streaming Log Parser using Drain3 for SOICT 2026 Pipeline
"""

import re
import json
import numpy as np

class SimplifiedDrain3Parser:
    """
    Lightweight Drain3-compatible online log parser for streaming enterprise campus logs.
    """
    def __init__(self, sim_th=0.6, depth=4):
        self.sim_th = sim_th
        self.depth = depth
        self.templates = []
        self.template_counts = {}

    def _sanitize(self, log_msg):
        # Mask IP addresses, UUIDs, timestamps, numbers
        msg = re.sub(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', '<IP>', log_msg)
        msg = re.sub(r'\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b', '<UUID>', msg)
        msg = re.sub(r'\b\d{4}-\d{2}-\d{2}[T ]\d{2}:\d{2}:\d{2}\b', '<TIMESTAMP>', msg)
        msg = re.sub(r'\b\d+\b', '<NUM>', msg)
        return msg.strip()

    def parse_line(self, raw_line):
        sanitized = self._sanitize(raw_line)
        tokens = sanitized.split()
        
        best_match_id = -1
        best_sim = -1.0
        
        for idx, template_tokens in enumerate(self.templates):
            if len(template_tokens) != len(tokens):
                continue
            
            matching_count = sum(1 for a, b in zip(tokens, template_tokens) if a == b or a == '<NUM>' or b == '<NUM>')
            sim = matching_count / max(len(tokens), 1)
            
            if sim > best_sim and sim >= self.sim_th:
                best_sim = sim
                best_match_id = idx
                
        if best_match_id != -1:
            # Update template with wildcard
            updated = []
            for a, b in zip(tokens, self.templates[best_match_id]):
                if a == b:
                    updated.append(a)
                else:
                    updated.append('<*>')
            self.templates[best_match_id] = updated
            self.template_counts[best_match_id] += 1
            return best_match_id, " ".join(updated)
        else:
            new_id = len(self.templates)
            self.templates.append(tokens)
            self.template_counts[new_id] = 1
            return new_id, " ".join(tokens)

    def parse_batch(self, lines):
        results = []
        for line in lines:
            if line.strip():
                template_id, template_str = self.parse_line(line)
                results.append((template_id, template_str))
        return results

if __name__ == "__main__":
    sample_logs = [
        "2026-08-11 10:00:01 auth_service IP 192.168.1.50 User admin login success",
        "2026-08-11 10:00:02 auth_service IP 192.168.1.51 User guest login failed invalid password",
        "2026-08-11 10:00:03 firewall_core Drop packet from 10.0.0.5 port 8080 to 172.16.0.1",
        "2026-08-11 10:00:04 auth_service IP 192.168.1.52 User guest login failed invalid password"
    ]
    
    parser = SimplifiedDrain3Parser()
    parsed = parser.parse_batch(sample_logs)
    print("[+] Drain3 Parsed Results:")
    for t_id, t_str in parsed:
        print(f"  Template #{t_id}: {t_str}")
