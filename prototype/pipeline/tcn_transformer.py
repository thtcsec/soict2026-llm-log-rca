"""
tcn_transformer.py - Hybrid TCN-Transformer Autoencoder for SOICT 2026
"""

import torch
import torch.nn as nn
import numpy as np

class TCNTransformerAutoencoder(nn.Module):
    def __init__(self, vocab_size=500, embed_dim=64, num_heads=4, hidden_dim=128, seq_len=20):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # Temporal Convolutional Layer (dilated causal 1D conv)
        self.tcn = nn.Sequential(
            nn.Conv1d(embed_dim, hidden_dim, kernel_size=3, padding=1, dilation=1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=2, dilation=2),
            nn.ReLU()
        )
        
        # Transformer Encoder Block
        encoder_layer = nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=num_heads, dim_feedforward=256, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Decoder / Reconstruction Head
        self.decoder = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # x: [B, L]
        emb = self.embedding(x) # [B, L, E]
        emb_trans = emb.transpose(1, 2) # [B, E, L]
        tcn_out = self.tcn(emb_trans).transpose(1, 2) # [B, L, H]
        
        trans_out = self.transformer_encoder(tcn_out) # [B, L, H]
        logits = self.decoder(trans_out) # [B, L, Vocab]
        return logits

def calculate_anomaly_scores(logits, targets):
    criterion = nn.CrossEntropyLoss(reduction='none')
    # logits: [B, L, V], targets: [B, L]
    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss_per_seq = loss.view(targets.size(0), targets.size(1)).mean(dim=1)
    return loss_per_seq

if __name__ == "__main__":
    model = TCNTransformerAutoencoder(vocab_size=100, seq_len=10)
    dummy_input = torch.randint(0, 100, (4, 10))
    logits = model(dummy_input)
    scores = calculate_anomaly_scores(logits, dummy_input)
    print(f"[+] Output Logits Shape: {logits.shape}")
    print(f"[+] Sequence Anomaly Scores: {scores.detach().numpy()}")
