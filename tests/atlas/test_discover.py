"""Découverte non supervisée de directions latentes (SVD/PCA par couche)."""
import torch

from meridian.atlas import discover_directions


def test_top_component_aligns_with_high_variance_axis():
    # Données concentrées : grande variance sur x, faible sur y, nulle sur z.
    g = torch.Generator().manual_seed(0)
    n = 300
    x = torch.randn(n, generator=g) * 5.0
    y = torch.randn(n, generator=g) * 0.4
    z = torch.zeros(n)
    acts = torch.stack([x, y, z], dim=1).unsqueeze(0)   # (1, N, 3)

    basis, explained = discover_directions(acts, k=2)

    assert basis.shape == (1, 2, 3)
    assert explained.shape == (1, 2)
    # 1re direction latente colinéaire à x (au signe près).
    assert abs(float(basis[0, 0] @ torch.tensor([1.0, 0.0, 0.0]))) > 0.99
    # Variance expliquée décroissante.
    assert explained[0, 0] > explained[0, 1]


def test_k_clamped_to_rank():
    # k demandé > min(N, H) : on borne au rang disponible.
    acts = torch.randn(1, 4, 3)                          # H=3 -> au plus 3 directions
    basis, explained = discover_directions(acts, k=10)
    assert basis.shape == (1, 3, 3)
    assert explained.shape == (1, 3)


def test_latent_directions_are_unit_norm():
    acts = torch.randn(2, 50, 6)
    basis, _ = discover_directions(acts, k=3)
    norms = basis.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_rejects_non_3d_input():
    import pytest
    with pytest.raises(ValueError):
        discover_directions(torch.randn(10, 3), k=2)
