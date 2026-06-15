"""Divergence KL sur prompts harmless : préservation des capacités (KB §8.2).

KL scalaire (mesure agrégée) + diagnostic fin : KL PAR POSITION (quels tokens l'ablation
perturbe le plus) pour localiser la casse — complément du module `circuits/`.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F


def per_token_kl(logits_p: torch.Tensor, logits_q: torch.Tensor) -> torch.Tensor:
    """KL(P‖Q) par position : (..., V) -> (...,). P = modèle original, Q = abliteré."""
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    return (log_p.exp() * (log_p - log_q)).sum(dim=-1)


def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """KL(P ‖ Q) moyennée, P/Q = softmax des logits (... , V). P = modèle original."""
    return float(per_token_kl(logits_p, logits_q).mean())


@dataclass
class KLDiagnostic:
    mean: float                 # KL moyenne (= kl_divergence)
    max: float                  # pire position
    p95: float                  # 95e percentile (queue de divergence)
    top_positions: list[int]    # indices des positions les plus divergentes


def kl_diagnostic(logits_p: torch.Tensor, logits_q: torch.Tensor, top_k: int = 5) -> KLDiagnostic:
    """Résumé de la distribution des KL par position : moyenne, max, p95, top-k positions.

    Une KL moyenne basse peut masquer quelques positions très perturbées (la moyenne dilue) :
    ce diagnostic révèle *où* l'ablation abîme le modèle, pas seulement *de combien* en moyenne.
    """
    kl = per_token_kl(logits_p, logits_q).flatten()
    n = kl.numel()
    if n == 0:
        return KLDiagnostic(0.0, 0.0, 0.0, [])
    k = max(1, min(top_k, n))
    top = torch.topk(kl, k).indices.tolist()
    q95 = float(torch.quantile(kl, 0.95)) if n > 1 else float(kl.max())
    return KLDiagnostic(float(kl.mean()), float(kl.max()), q95, top)
