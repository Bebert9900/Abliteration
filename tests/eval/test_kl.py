"""Tests KL : scalaire, par position, et diagnostic (max/p95/top positions)."""
import torch

from abliteration.eval import kl_diagnostic, kl_divergence, per_token_kl


def test_identical_logits_have_zero_kl():
    logits = torch.randn(4, 10)
    assert abs(kl_divergence(logits, logits)) < 1e-6
    assert torch.allclose(per_token_kl(logits, logits), torch.zeros(4), atol=1e-6)


def test_per_token_kl_shape_and_mean_matches_scalar():
    p = torch.randn(6, 8)
    q = torch.randn(6, 8)
    pt = per_token_kl(p, q)
    assert pt.shape == (6,)
    assert abs(float(pt.mean()) - kl_divergence(p, q)) < 1e-6


def test_diagnostic_flags_the_perturbed_position():
    torch.manual_seed(0)
    base = torch.zeros(5, 4)            # distribution uniforme partout
    abl = base.clone()
    abl[3] = torch.tensor([10.0, 0, 0, 0])   # position 3 fortement perturbée
    diag = kl_diagnostic(base, abl, top_k=1)
    assert diag.top_positions == [3]
    assert diag.max >= diag.mean
    assert diag.max > 0.0


def test_diagnostic_empty_is_safe():
    diag = kl_diagnostic(torch.zeros(0, 4), torch.zeros(0, 4))
    assert diag.mean == 0.0 and diag.top_positions == []
