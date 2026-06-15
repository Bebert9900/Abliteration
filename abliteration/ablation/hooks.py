"""Hooks d'ablation inference-time : réversibles, pour explorer/sélectionner/évaluer.

On retranche `alpha` fois la projection sur la direction (projetée) de la sortie de chaque
module ciblé : `x' = x − α (x · r̂) r̂`. `alpha=1.0` = ablation complète (par défaut) ;
`alpha<1.0` = ablation graduée (force partielle, utile pour l'optimisation : le trade-off
refus/préservation se règle en continu). Réversible via `handle.remove()`.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def make_ablation_hook(direction: torch.Tensor, alpha: float = 1.0):
    """Renvoie un forward-hook retranchant `alpha·(x·r̂)r̂` de la sortie du module.

    `direction` est renormalisée. `alpha` : force d'ablation (0 = aucune, 1 = projection
    complète retirée).
    """
    if direction is None:
        raise ValueError("make_ablation_hook : `direction` est None.")
    r = direction / (direction.norm() + 1e-8)

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        rh = r.to(h.dtype)
        proj = (h @ rh).unsqueeze(-1) * rh
        h = h - alpha * proj
        return (h, *output[1:]) if is_tuple else h

    return hook


def register_ablation_hooks(modules: list[nn.Module], direction: torch.Tensor, alpha: float = 1.0):
    """Pose le hook d'ablation (force `alpha`) sur chaque module ; renvoie les handles (réversible)."""
    hook = make_ablation_hook(direction, alpha)
    return [m.register_forward_hook(hook) for m in modules]


def make_steering_hook(direction: torch.Tensor, alpha: float = 1.0):
    """Renvoie un forward-hook qui AJOUTE `alpha·r̂` à la sortie du module (steering).

    Inverse de l'ablation : au lieu de retirer la composante selon r̂, on ajoute le vecteur de
    pilotage unitaire — on pousse les activations vers le comportement encodé par la direction.
    `alpha` est l'intensité du pilotage (0 = aucun effet). Réversible via `handle.remove()`.
    """
    if direction is None:
        raise ValueError("make_steering_hook : `direction` est None.")
    r = direction / (direction.norm() + 1e-8)

    def hook(module, inputs, output):
        is_tuple = isinstance(output, tuple)
        h = output[0] if is_tuple else output
        h = h + alpha * r.to(h.dtype)
        return (h, *output[1:]) if is_tuple else h

    return hook


def register_steering_hooks(modules: list[nn.Module], direction: torch.Tensor, alpha: float = 1.0):
    """Pose le hook de steering (intensité `alpha`) sur chaque module ; renvoie les handles."""
    hook = make_steering_hook(direction, alpha)
    return [m.register_forward_hook(hook) for m in modules]
