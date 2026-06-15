"""Séparabilité géométrique entre concepts : matrice cosinus N×N.

Généralise `directions.separability` (refus↔négation/agentique) à un nombre quelconque de
concepts. Deux concepts dont les directions sont colinéaires (|cos| élevé) ne peuvent pas être
modifiés indépendamment : intervenir sur l'un déborde sur l'autre.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .concept import ConceptDirection


@dataclass
class SeparabilityMatrix:
    names: list[str]
    matrix: list[list[float]]      # cosinus N×N (agrégé sur la couche choisie)
    layer: int | None              # couche utilisée (None = moyenne sur toutes les couches)

    def warnings(self, threshold: float = 0.3) -> list[str]:
        """Signale les paires de concepts trop colinéaires (au-dessus du seuil)."""
        msgs = []
        for i in range(len(self.names)):
            for j in range(i + 1, len(self.names)):
                c = abs(self.matrix[i][j])
                if c > threshold:
                    msgs.append(
                        f"|cos({self.names[i]}, {self.names[j]})|={c:.2f} > {threshold} — "
                        f"concepts géométriquement intriqués (intervention non indépendante)."
                    )
        return msgs


def pairwise_separability(directions: dict[str, ConceptDirection],
                          layer: int | None = None) -> SeparabilityMatrix:
    """Matrice cosinus N×N entre directions de concepts.

    `layer=None` : moyenne du cosinus sur toutes les couches ; sinon couche précise. Les
    directions étant unitaires par couche, le cosinus est le produit scalaire. Matrice
    symétrique, diagonale 1.0.
    """
    names = list(directions)
    n = len(names)
    vecs = []
    for name in names:
        d = directions[name].direction.to(torch.float32)
        v = d[layer] if layer is not None else d            # (H,) ou (L+1, H)
        v = v / (v.norm(dim=-1, keepdim=True) + 1e-8)
        vecs.append(v)

    mat = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if layer is not None:
                cos = float(vecs[i] @ vecs[j])
            else:
                cos = float((vecs[i] * vecs[j]).sum(dim=-1).mean())   # moyenne sur les couches
            mat[i][j] = cos
    return SeparabilityMatrix(names, mat, layer)
