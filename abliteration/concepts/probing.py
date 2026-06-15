"""Probing linéaire : où un concept devient-il décodable, couche par couche ?

On entraîne une sonde linéaire (régression logistique, pure torch — aucune dépendance externe)
à séparer les activations positives des négatives, indépendamment à CHAQUE couche. La courbe
`accuracy_per_layer` révèle la profondeur à laquelle le concept est linéairement représenté :
un classique de l'interprétabilité (linear probing). Pas d'entraînement du modèle, seulement
de la sonde.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


def _split_indices(n: int, train_frac: float, seed: int) -> tuple[list[int], list[int]]:
    """Indices train/test déterministes et disjoints (pas de fuite d'évaluation de la sonde)."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    n_tr = max(1, int(round(n * train_frac)))
    return perm[:n_tr], perm[n_tr:]


def train_linear_probe(x_pos: torch.Tensor, x_neg: torch.Tensor, epochs: int = 200,
                       lr: float = 0.05, train_frac: float = 0.7, seed: int = 0) -> tuple[float, torch.Tensor]:
    """Entraîne une sonde logistique pos vs neg ; renvoie (accuracy_test, poids).

    `x_pos` (Np, H), `x_neg` (Nn, H). Standardisation par dimension, descente de gradient pleine
    batch, split train/test interne déterministe. `accuracy` mesurée sur le test (held-out).
    """
    x = torch.cat([x_pos, x_neg], dim=0).float()
    y = torch.cat([torch.ones(len(x_pos)), torch.zeros(len(x_neg))]).float()
    # Standardisation : stabilise l'optimisation quand les échelles varient selon les dimensions.
    mu, sd = x.mean(0, keepdim=True), x.std(0, keepdim=True) + 1e-6
    x = (x - mu) / sd

    tr, te = _split_indices(len(x), train_frac, seed)
    if not te:                                   # trop peu d'exemples : test = train (dégénéré)
        te = tr
    xtr, ytr = x[tr], y[tr]

    w = torch.zeros(x.shape[1], requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=lr)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(xtr @ w + b, ytr)
        loss.backward()
        opt.step()

    with torch.no_grad():
        pred = ((x[te] @ w + b) > 0).float()
        acc = float((pred == y[te]).float().mean())
    return acc, w.detach()


@dataclass
class ProbeReport:
    accuracy_per_layer: list[float]   # accuracy de la sonde à chaque couche (L+1 valeurs)
    best_layer: int                   # couche la plus décodable

    def to_dict(self) -> dict:
        return {"accuracy_per_layer": self.accuracy_per_layer, "best_layer": self.best_layer}


def probe_per_layer(pos_acts: torch.Tensor, neg_acts: torch.Tensor, epochs: int = 200,
                    lr: float = 0.05, seed: int = 0) -> ProbeReport:
    """Sonde linéaire à chaque couche. `pos_acts`/`neg_acts` = (L+1, N, H).

    Renvoie la courbe de décodabilité et la couche la plus séparable.
    """
    if pos_acts.ndim != 3 or neg_acts.ndim != 3:
        raise ValueError("probe_per_layer attend des activations (L+1, N, H) par exemple.")
    n_layers = pos_acts.shape[0]
    accs = [train_linear_probe(pos_acts[l], neg_acts[l], epochs=epochs, lr=lr, seed=seed)[0]
            for l in range(n_layers)]
    best = int(max(range(n_layers), key=lambda l: accs[l]))
    return ProbeReport(accuracy_per_layer=accs, best_layer=best)
