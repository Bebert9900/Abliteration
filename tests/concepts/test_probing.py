"""Tests du probing linéaire : décodabilité d'un concept couche par couche."""
import pytest
import torch

from meridian.concepts import probe_per_layer, train_linear_probe


def test_probe_separates_well_separated_gaussians():
    torch.manual_seed(0)
    pos = torch.randn(40, 6) + torch.tensor([5.0, 0, 0, 0, 0, 0])
    neg = torch.randn(40, 6) - torch.tensor([5.0, 0, 0, 0, 0, 0])
    acc, w = train_linear_probe(pos, neg)
    assert acc > 0.95
    assert w.shape == (6,)


def test_probe_near_chance_on_identical_distributions():
    torch.manual_seed(1)
    pos = torch.randn(60, 5)
    neg = torch.randn(60, 5)            # même distribution -> indécodable
    acc, _ = train_linear_probe(pos, neg)
    assert 0.3 <= acc <= 0.7           # autour du hasard


def test_probe_per_layer_finds_the_separable_layer():
    torch.manual_seed(2)
    L, N, H = 3, 40, 4
    pos = torch.randn(L, N, H)
    neg = torch.randn(L, N, H)
    # seule la couche 1 sépare nettement les deux classes
    pos[1] += torch.tensor([6.0, 0, 0, 0])
    neg[1] -= torch.tensor([6.0, 0, 0, 0])
    report = probe_per_layer(pos, neg)
    assert len(report.accuracy_per_layer) == L
    assert report.best_layer == 1
    assert report.accuracy_per_layer[1] > max(report.accuracy_per_layer[0],
                                              report.accuracy_per_layer[2])


def test_probe_per_layer_rejects_wrong_shape():
    with pytest.raises(ValueError):
        probe_per_layer(torch.randn(10, 4), torch.randn(10, 4))   # (N, H) au lieu de (L+1, N, H)
