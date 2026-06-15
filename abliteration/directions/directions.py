"""Directions unitaires des 4 classes + séparabilité.

Formulations (la direction de refus est canonique, KB §2 ; les directions « à préserver » sont
un choix de conception de ce projet — généralisation de la projected abliteration, KB §3.2 —
NON figé dans la KB v.mai-2026, à documenter dans la model card) :

    baseline   = μ_harmless                       (référence neutre/compliante)
    r̂ (refus)  = normalize(μ_harmful  − μ_harmless)   ← canonique
    n̂ (négation) = normalize(μ_negation − μ_harmless)
    â (agentique)= normalize(μ_agentic  − μ_harmless)
    ĥ (harmless) = normalize(μ_harmless − μ̄)          (μ̄ = moyenne des 4 moyennes)

n̂ et â partagent la baseline harmless avec r̂ : la séparabilité cosine(r̂, n̂) mesure alors
proprement si « dévier vers le harmful » et « dévier vers la négation » pointent au même endroit.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from abliteration.data import PromptClass


def _normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


@dataclass
class Directions:
    refusal: torch.Tensor   # (L+1, H), unitaire par couche
    harmless: torch.Tensor
    negation: torch.Tensor
    agentic: torch.Tensor

    _PRESERVE = {"harmless": "harmless", "negation": "negation", "agentic": "agentic"}

    def layer(self, l: int) -> dict[str, torch.Tensor]:
        return {
            "refusal": self.refusal[l],
            "harmless": self.harmless[l],
            "negation": self.negation[l],
            "agentic": self.agentic[l],
        }

    def preserve_vectors(self, names: list[str], l: int) -> list[torch.Tensor]:
        """Vecteurs à préserver (à orthogonaliser contre) pour la couche `l`."""
        layer = self.layer(l)
        return [layer[self._PRESERVE[n]] for n in names]


def compute_directions(means: dict[PromptClass, torch.Tensor]) -> Directions:
    harmful = means[PromptClass.HARMFUL]
    harmless = means[PromptClass.HARMLESS]
    negation = means[PromptClass.LEGITIMATE_NEGATION]
    agentic = means[PromptClass.AGENTIC]
    grand = torch.stack([harmful, harmless, negation, agentic], dim=0).mean(dim=0)
    return Directions(
        refusal=_normalize(harmful - harmless),
        negation=_normalize(negation - harmless),
        agentic=_normalize(agentic - harmless),
        harmless=_normalize(harmless - grand),
    )


def top_k_directions(harmful_acts: torch.Tensor, harmless_acts: torch.Tensor, k: int = 1) -> torch.Tensor:
    """Top-k directions de refus par SVD du contraste (généralisation multi-direction, KB §3.6).

    `harmful_acts` (N_h, H) et `harmless_acts` (N_l, H) sont les activations PAR EXEMPLE (pooled
    au dernier token) d'une couche donnée. La mean-difference classique r̂ = μ_h − μ_l ne capte
    qu'UNE direction ; quand le refus est porté par un petit sous-espace, on veut les `k`
    premières directions. On empile les écarts au centre de la classe opposée et on prend les k
    premiers vecteurs singuliers droits.

    Renvoie (k, H), lignes unitaires orientées « vers le refus ». Pour k=1, colinéaire à r̂.
    """
    if harmful_acts.ndim != 2 or harmless_acts.ndim != 2:
        raise ValueError("top_k_directions attend des matrices (N, H) par exemple.")
    mu_h = harmful_acts.float().mean(dim=0)
    mu_l = harmless_acts.float().mean(dim=0)
    diffs = torch.cat([harmful_acts.float() - mu_l, mu_h - harmless_acts.float()], dim=0)
    _, _, vh = torch.linalg.svd(diffs, full_matrices=False)
    k = max(1, min(k, vh.shape[0]))
    dirs = vh[:k]
    ref = _normalize(mu_h - mu_l)                       # oriente chaque direction vers le refus
    signs = torch.sign(dirs @ ref)
    signs[signs == 0] = 1.0
    return _normalize(dirs * signs.unsqueeze(-1))


@dataclass
class SeparabilityReport:
    cosine_refusal_negation: torch.Tensor   # (L+1,)
    cosine_refusal_agentic: torch.Tensor
    threshold: float

    def warnings(self) -> list[str]:
        msgs: list[str] = []
        for l in range(len(self.cosine_refusal_negation)):
            cn = float(self.cosine_refusal_negation[l])
            ca = float(self.cosine_refusal_agentic[l])
            if abs(cn) > self.threshold:
                msgs.append(
                    f"Couche {l}: |cos(refus, negation)|={abs(cn):.2f} > {self.threshold} — "
                    f"l'ablation risque de déborder sur la négation légitime."
                )
            if abs(ca) > self.threshold:
                msgs.append(
                    f"Couche {l}: |cos(refus, agentic)|={abs(ca):.2f} > {self.threshold} — "
                    f"l'ablation risque de déborder sur les capacités agentiques."
                )
        return msgs


def separability(directions: Directions, threshold: float = 0.3) -> SeparabilityReport:
    """Cosine par couche de r̂ contre n̂ et â (vecteurs déjà unitaires -> produit scalaire)."""
    cos_rn = (directions.refusal * directions.negation).sum(dim=-1)
    cos_ra = (directions.refusal * directions.agentic).sum(dim=-1)
    return SeparabilityReport(cos_rn, cos_ra, threshold)
