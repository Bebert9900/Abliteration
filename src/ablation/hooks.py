"""Hooks d'ablation inference-time : réversibles, pour explorer/sélectionner/évaluer.

On retranche la projection sur la direction (projetée) de la sortie de chaque module ciblé :
`x' = x − (x · r̂) r̂`. Réversible via `handle.remove()` (KB §5a).
"""
from __future__ import annotations

import torch
import torch.nn as nn


def make_ablation_hook(direction: torch.Tensor):
    r = direction / (direction.norm() + 1e-8)

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        proj = (h @ r.to(h.dtype)).unsqueeze(-1) * r.to(h.dtype)
        h = h - proj
        return (h, *output[1:]) if is_tuple else h

    return hook


def register_ablation_hooks(modules: list[nn.Module], direction: torch.Tensor):
    """Pose le hook d'ablation sur chaque module et renvoie les handles (réversible)."""
    hook = make_ablation_hook(direction)
    return [m.register_forward_hook(hook) for m in modules]
