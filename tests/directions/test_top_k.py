"""Tests des directions multi-direction (SVD du contraste harmful↔harmless)."""
import torch

from meridian.directions import top_k_directions


def test_k1_is_collinear_with_mean_difference():
    torch.manual_seed(0)
    h = torch.randn(50, 8) + torch.tensor([3.0, 0, 0, 0, 0, 0, 0, 0])
    l = torch.randn(50, 8)
    d = top_k_directions(h, l, k=1)
    assert d.shape == (1, 8)
    mean_diff = (h.mean(0) - l.mean(0))
    mean_diff = mean_diff / mean_diff.norm()
    cos = abs(float(d[0] @ mean_diff))
    assert cos > 0.9  # k=1 ≈ mean-difference


def test_directions_are_unit_and_orthonormal():
    torch.manual_seed(1)
    h = torch.randn(40, 6) + 2.0
    l = torch.randn(40, 6)
    d = top_k_directions(h, l, k=3)
    assert d.shape == (3, 6)
    norms = d.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
    gram = d @ d.T
    assert torch.allclose(gram, torch.eye(3), atol=1e-4)


def test_directions_oriented_towards_refusal():
    torch.manual_seed(2)
    h = torch.randn(30, 5) + torch.tensor([5.0, 0, 0, 0, 0])
    l = torch.randn(30, 5)
    ref = (h.mean(0) - l.mean(0))
    d = top_k_directions(h, l, k=1)
    assert float(d[0] @ ref) > 0  # orientée vers le refus, pas à l'opposé


def test_rejects_non_matrix_input():
    import pytest
    with pytest.raises(ValueError):
        top_k_directions(torch.randn(8), torch.randn(8))
