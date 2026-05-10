"""tests/test_residual_gru.py — ResidualGRU forward shape + gradient flow.

Per plan-003 §4.6.
"""
from __future__ import annotations

import torch

from src.models.residual_gru import ResidualGRU


def test_forward_shape_input_dim_3():
    m = ResidualGRU(input_dim=3, hidden=64, layers=2, dropout=0.1)
    X = torch.randn(8, 11, 3)
    y = m(X)
    assert y.shape == (8, 3)
    assert torch.isfinite(y).all()


def test_forward_shape_input_dim_13():
    m = ResidualGRU(input_dim=13)
    X = torch.randn(8, 11, 13)
    assert m(X).shape == (8, 3)


def test_forward_shape_input_dim_22():
    m = ResidualGRU(input_dim=22)
    X = torch.randn(8, 11, 22)
    assert m(X).shape == (8, 3)


def test_dropout_zero_when_layers_1():
    m = ResidualGRU(input_dim=3, hidden=32, layers=1, dropout=0.5)
    # PyTorch nn.GRU emits dropout=0 internally when num_layers==1
    assert m.gru.dropout == 0.0


def test_gradient_flow():
    m = ResidualGRU(input_dim=3)
    X = torch.randn(4, 11, 3, requires_grad=False)
    y = m(X)
    target = torch.randn(4, 3)
    loss = (y - target).pow(2).mean()
    loss.backward()
    grads = [p.grad for p in m.parameters() if p.grad is not None]
    assert len(grads) > 0
    for g in grads:
        assert torch.isfinite(g).all()
