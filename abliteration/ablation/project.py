"""Projection de directions : Gram-Schmidt + renormalisation, et résolution par variante."""
from __future__ import annotations

from enum import Enum

import torch

from abliteration.directions import Directions


def _orthonormal_basis(vectors: list[torch.Tensor], eps: float = 1e-8) -> list[torch.Tensor]:
    basis: list[torch.Tensor] = []
    for v in vectors:
        w = v.clone().float()
        for b in basis:
            w = w - (w @ b) * b
        norm = w.norm()
        if norm > eps:
            basis.append(w / norm)
    return basis


def project_out(direction: torch.Tensor, against: list[torch.Tensor], eps: float = 1e-8) -> torch.Tensor:
    """Retire de `direction` ses composantes sur l'espace engendré par `against`, puis renormalise.

    Généralisation de la projected abliteration : `against` = directions à PRÉSERVER (harmless,
    négation, agentique) → l'ablation ne touchera plus ces directions.
    """
    d = direction.clone().float()
    for b in _orthonormal_basis(against, eps):
        d = d - (d @ b) * b
    return d / (d.norm() + eps)


class Variant(str, Enum):
    CONVENTIONAL = "conventional"
    PROJECTED = "projected"                              # contre harmless (KB §3.2)
    PRESERVING = "preserving"                            # contre sous-ensemble [ĥ, n̂, â]
    NORM_PRESERVING_BIPROJECTED = "norm_preserving_biprojected"


def ablation_direction(
    directions: Directions,
    layer: int,
    variant: Variant,
    preserve: list[str] | None = None,
) -> torch.Tensor:
    """Direction (unitaire) effectivement ablatée pour `layer`, selon la variante."""
    r = directions.refusal[layer]
    if variant is Variant.CONVENTIONAL:
        return r / (r.norm() + 1e-8)
    if variant is Variant.PROJECTED:
        return project_out(r, [directions.harmless[layer]])
    if variant is Variant.PRESERVING:
        names = preserve or ["negation", "agentic"]
        return project_out(r, directions.preserve_vectors(names, layer))
    if variant is Variant.NORM_PRESERVING_BIPROJECTED:
        # Simplification documentée : on préserve (négation, agentique) à la projection ; la
        # préservation de norme est appliquée au stade des poids (orthogonalize_weights). La
        # biprojection inter-couches complète est un raffinement ultérieur (KB §3.4).
        names = preserve or ["harmless", "negation", "agentic"]
        return project_out(r, directions.preserve_vectors(names, layer))
    raise ValueError(f"variante inconnue : {variant}")
