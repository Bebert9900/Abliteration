"""Tests des hooks d'ablation inference-time : réversibles, retirent la direction projetée."""
import torch
import torch.nn as nn

from src.ablation import register_ablation_hooks


def test_hook_removes_direction_then_is_reversible():
    torch.manual_seed(0)
    layer = nn.Linear(4, 4)
    x = torch.randn(2, 3, 4)
    original = layer(x).clone()

    r = torch.tensor([1.0, 0.0, 0.0, 0.0])
    handles = register_ablation_hooks([layer], r)
    ablated = layer(x)
    # plus aucune composante selon r dans la sortie
    proj = (ablated @ r)
    assert torch.allclose(proj, torch.zeros_like(proj), atol=1e-5)

    for h in handles:
        h.remove()
    restored = layer(x)
    assert torch.allclose(restored, original, atol=1e-6)
