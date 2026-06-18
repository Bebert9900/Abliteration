"""Tests des hooks d'ablation inference-time : réversibles, retirent la direction projetée."""
import torch
import torch.nn as nn

from meridian.ablation import register_ablation_hooks, register_steering_hooks


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


def test_hook_alpha_grades_ablation_strength():
    torch.manual_seed(0)
    layer = nn.Linear(4, 4)
    x = torch.randn(2, 3, 4)
    original = layer(x).clone()
    r = torch.tensor([1.0, 0.0, 0.0, 0.0])
    full_proj = original @ r

    # alpha=0.5 ne retire que la moitié de la composante selon r.
    handles = register_ablation_hooks([layer], r, alpha=0.5)
    half = layer(x)
    for h in handles:
        h.remove()
    assert torch.allclose(half @ r, 0.5 * full_proj, atol=1e-5)


def test_hook_alpha_zero_is_identity():
    torch.manual_seed(1)
    layer = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    original = layer(x).clone()
    handles = register_ablation_hooks([layer], torch.tensor([0.0, 1.0, 0.0, 0.0]), alpha=0.0)
    out = layer(x)
    for h in handles:
        h.remove()
    assert torch.allclose(out, original, atol=1e-6)


def test_steering_hook_adds_direction_and_is_reversible():
    torch.manual_seed(0)
    layer = nn.Linear(4, 4)
    x = torch.randn(2, 3, 4)
    original = layer(x).clone()
    r = torch.tensor([1.0, 0.0, 0.0, 0.0])      # unitaire

    handles = register_steering_hooks([layer], r, alpha=2.0)
    steered = layer(x)
    for h in handles:
        h.remove()
    # la composante selon r est augmentée d'exactement alpha (=2.0)
    assert torch.allclose(steered @ r - original @ r, torch.full((2, 3), 2.0), atol=1e-5)
    # réversible
    assert torch.allclose(layer(x), original, atol=1e-6)


def test_steering_hook_alpha_zero_is_identity():
    torch.manual_seed(3)
    layer = nn.Linear(4, 4)
    x = torch.randn(2, 4)
    original = layer(x).clone()
    handles = register_steering_hooks([layer], torch.tensor([0.0, 0.0, 1.0, 0.0]), alpha=0.0)
    out = layer(x)
    for h in handles:
        h.remove()
    assert torch.allclose(out, original, atol=1e-6)
