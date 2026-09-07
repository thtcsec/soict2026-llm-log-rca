"""Small integrity tests; they do not read raw logs or processed datasets."""

import torch

from pipeline.tcn_transformer import (
    CausalConv1d,
    TCNTransformerAutoencoder,
    make_masked_inputs,
    pseudo_log_likelihood_scores,
)


def test_mask_token_is_really_inserted():
    torch.manual_seed(7)
    x = torch.arange(1, 41).view(2, 20)
    masked, selected = make_masked_inputs(x, mask_token_id=99, mask_ratio=0.5)
    assert selected.any()
    assert torch.equal(masked[selected], torch.full_like(masked[selected], 99))
    assert torch.equal(masked[~selected], x[~selected])


def test_causal_convolution_cannot_see_future():
    torch.manual_seed(7)
    conv = CausalConv1d(1, 2, kernel_size=3, dilation=2).eval()
    a = torch.zeros(1, 1, 10)
    b = a.clone()
    b[:, :, 8:] = 10
    assert torch.allclose(conv(a)[:, :, :8], conv(b)[:, :, :8])


def test_pll_returns_one_finite_score_per_sequence():
    torch.manual_seed(7)
    model = TCNTransformerAutoencoder(vocab_size=32, hidden_dim=32, seq_len=6).eval()
    x = torch.randint(1, 31, (3, 6))
    scores = pseudo_log_likelihood_scores(model, x, model.mask_token_id)
    assert scores.shape == (3,)
    assert torch.isfinite(scores).all()
