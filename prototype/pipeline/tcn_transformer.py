"""Causal TCN--Transformer masked language model used in the SOICT study."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(max_len, dim)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)].to(dtype=x.dtype)


class CausalConv1d(nn.Conv1d):
    """Left-padded convolution: output at t cannot see t+1."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        self.left_padding = (kernel_size - 1) * dilation
        super().__init__(in_channels, out_channels, kernel_size, dilation=dilation, padding=self.left_padding)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = super().forward(x)
        return y[:, :, : -self.left_padding] if self.left_padding else y


class TCNTransformerAutoencoder(nn.Module):
    """Masked-token model with a residual causal TCN and Transformer encoder."""

    def __init__(self, vocab_size=500, embed_dim=64, num_heads=4, hidden_dim=128,
                 seq_len=20, pad_id=0, mask_token_id=None):
        super().__init__()
        self.pad_id = pad_id
        self.mask_token_id = vocab_size - 1 if mask_token_id is None else mask_token_id
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        self.input_projection = nn.Conv1d(embed_dim, hidden_dim, kernel_size=1)
        self.tcn = nn.Sequential(
            CausalConv1d(embed_dim, hidden_dim, kernel_size=3, dilation=1), nn.GELU(),
            CausalConv1d(hidden_dim, hidden_dim, kernel_size=3, dilation=2), nn.GELU(),
        )
        self.tcn_norm = nn.LayerNorm(hidden_dim)
        self.position = SinusoidalPositionEncoding(hidden_dim, seq_len)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, dim_feedforward=256,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.decoder = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pad_mask = x.eq(self.pad_id)
        emb = self.embedding(x).transpose(1, 2)
        tcn_out = self.tcn(emb) + self.input_projection(emb)
        encoded = self.position(self.tcn_norm(tcn_out.transpose(1, 2)))
        encoded = self.transformer_encoder(encoded, src_key_padding_mask=pad_mask)
        return self.decoder(encoded)


def make_masked_inputs(targets: torch.Tensor, mask_token_id: int,
                       mask_ratio: float = 0.15, pad_id: int = 0):
    """Replace sampled non-padding targets by the reserved mask token."""
    eligible = targets.ne(pad_id)
    mask = (torch.rand_like(targets, dtype=torch.float32) < mask_ratio) & eligible
    empty = ~mask.any(dim=1)
    if empty.any():
        first = eligible.float().argmax(dim=1)
        mask[empty, first[empty]] = True
    masked = targets.clone()
    masked[mask] = mask_token_id
    return masked, mask


def masked_token_loss(logits: torch.Tensor, targets: torch.Tensor, mask: torch.Tensor):
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
    return nll.view_as(targets)[mask].mean()


def calculate_anomaly_scores(logits: torch.Tensor, targets: torch.Tensor):
    """Legacy full-reconstruction score kept for old, non-paper harnesses only."""
    nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), reduction="none")
    return nll.view_as(targets).mean(dim=1)


@torch.no_grad()
def pseudo_log_likelihood_scores(model: nn.Module, batch: torch.Tensor,
                                 mask_token_id: int, pad_id: int = 0):
    """Leave-one-position-out mean NLL for each sequence."""
    total = torch.zeros(batch.size(0), device=batch.device)
    count = batch.ne(pad_id).sum(dim=1).clamp_min(1)
    for pos in range(batch.size(1)):
        active = batch[:, pos].ne(pad_id)
        if not active.any():
            continue
        masked = batch.clone()
        masked[:, pos] = mask_token_id
        logits = model(masked)[:, pos, :]
        total += F.cross_entropy(logits, batch[:, pos], reduction="none") * active
    return total / count


if __name__ == "__main__":
    model = TCNTransformerAutoencoder(vocab_size=100, seq_len=10)
    sample = torch.randint(1, 99, (4, 10))
    masked, selected = make_masked_inputs(sample, model.mask_token_id)
    loss = masked_token_loss(model(masked), sample, selected)
    scores = pseudo_log_likelihood_scores(model.eval(), sample, model.mask_token_id)
    print(f"loss={loss.item():.4f}; scores={scores.numpy()}")
