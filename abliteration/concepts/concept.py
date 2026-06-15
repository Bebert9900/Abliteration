"""Abstraction « Concept » : un comportement défini par un contraste de prompts.

Généralise « la direction de refus » à un concept arbitraire (sycophantie, véracité, biais…).
La direction du concept par couche est le contraste de moyennes d'activations, exactement comme
la direction de refus canonique :

    d̂_concept[layer] = normalize( μ_positive[layer] − μ_negative[layer] )

Cette direction a la même forme `(L+1, H)` que `directions.Directions.refusal`, donc elle est
directement consommable par `ablation_direction`, les hooks et la séparabilité.
"""
from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field

import torch


def _concept_seed(seed: int, name: str) -> int:
    """Graine déterministe par concept (hashlib, jamais `hash()` randomisé — cf. data/dataset)."""
    digest = hashlib.sha256(name.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], "big") % 1000


def _split(texts: list[str], fraction: float, seed: int) -> tuple[list[str], list[str]]:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"fraction doit être dans [0, 1], reçu {fraction}")
    indices = list(range(len(texts)))
    random.Random(seed).shuffle(indices)
    n_hold = int(round(len(texts) * fraction))
    hold_idx = set(indices[:n_hold])
    holdout = [t for i, t in enumerate(texts) if i in hold_idx]
    train = [t for i, t in enumerate(texts) if i not in hold_idx]
    return train, holdout


@dataclass(frozen=True)
class Concept:
    """Un concept comportemental défini par un contraste positif/négatif."""
    name: str
    positive: list[str]        # prompts qui ACTIVENT le concept
    negative: list[str]        # prompts de référence (concept absent)
    description: str = ""

    def split(self, fraction: float, seed: int = 0) -> tuple["Concept", "Concept"]:
        """Découpe positive/négative en (train, holdout) déterministes et disjoints.

        Le train sert à calculer la direction, le holdout à mesurer son effet (anti-fuite).
        Décalage de graine indépendant pour positive et negative, dérivé du nom du concept.
        """
        s = _concept_seed(seed, self.name)
        pos_tr, pos_ho = _split(self.positive, fraction, s)
        neg_tr, neg_ho = _split(self.negative, fraction, s + 1)
        train = Concept(self.name, pos_tr, neg_tr, self.description)
        holdout = Concept(self.name, pos_ho, neg_ho, self.description)
        return train, holdout


@dataclass(frozen=True)
class ConceptDirection:
    """Direction unitaire d'un concept par couche, `(L+1, H)`."""
    name: str
    direction: torch.Tensor    # (L+1, H), unitaire par couche

    def norms_per_layer(self) -> list[float]:
        """Normes par couche (≈1.0 partout ; sert de garde-fou de normalisation)."""
        return self.direction.norm(dim=-1).tolist()


def direction_from_means(mu_positive: torch.Tensor, mu_negative: torch.Tensor) -> torch.Tensor:
    """`normalize(μ_pos − μ_neg)` par couche. Entrées `(L+1, H)` → sortie `(L+1, H)` unitaire."""
    diff = mu_positive.to(torch.float32) - mu_negative.to(torch.float32)
    return diff / (diff.norm(dim=-1, keepdim=True) + 1e-8)


def concept_direction(concept: Concept, model, formatter, batch_size: int = 8,
                      device=None) -> ConceptDirection:
    """Calcule la direction d'un concept en collectant les moyennes positive/négative.

    Réutilise `directions.collect_means` (générique sur des textes, accumulation float32). Lève
    ValueError si l'un des deux ensembles est vide (contraste impossible).
    """
    from abliteration.directions import collect_means

    if not concept.positive or not concept.negative:
        raise ValueError(f"Concept '{concept.name}' : positive et negative doivent être non vides.")
    mu_pos = collect_means(model, formatter, concept.positive, batch_size=batch_size, device=device)
    mu_neg = collect_means(model, formatter, concept.negative, batch_size=batch_size, device=device)
    return ConceptDirection(concept.name, direction_from_means(mu_pos, mu_neg))
