"""Découverte non supervisée de directions latentes : SVD/PCA par couche.

On décompose les activations par exemple `(L+1, N, H)` couche par couche pour extraire les `k`
directions qui portent le plus de variance — un jeu de directions « toutes » candidates, SANS
étiquette. Centrer puis SVD = PCA : les vecteurs singuliers droits sont les axes principaux, les
valeurs singulières au carré donnent la variance expliquée. Ces directions latentes sont ensuite
mises en regard des directions de sujets (cf. `atlas.match_subjects_to_latents`).

Ce sont des axes propres à la base de hidden states du modèle : ils ne sont comparables qu'au
sein d'une même lignée (cf. le suivi de fine-tuning, `drift`).
"""
from __future__ import annotations

import torch


def discover_directions(acts: torch.Tensor, k: int, center: bool = True
                        ) -> tuple[torch.Tensor, torch.Tensor]:
    """Top-`k` directions latentes par couche via SVD (PCA si `center`).

    `acts` : activations par exemple `(L+1, N, H)`. Renvoie :
      - `basis` `(L+1, k', H)` : vecteurs unitaires, k' = min(k, N, H) (borné au rang) ;
      - `explained` `(L+1, k')` : fraction de variance expliquée par direction (décroissante).
    """
    if acts.ndim != 3:
        raise ValueError("discover_directions attend des activations (L+1, N, H).")
    n_layers, n, h = acts.shape
    kk = min(k, n, h)
    basis = torch.empty(n_layers, kk, h, dtype=torch.float32)
    explained = torch.empty(n_layers, kk, dtype=torch.float32)
    for l in range(n_layers):
        x = acts[l].to(torch.float32)
        if center:
            x = x - x.mean(dim=0, keepdim=True)
        # full_matrices=False : Vh a la forme (min(N,H), H), lignes = axes principaux unitaires.
        _, s, vh = torch.linalg.svd(x, full_matrices=False)
        basis[l] = vh[:kk]
        var = s**2
        explained[l] = var[:kk] / (var.sum() + 1e-12)
    return basis, explained
