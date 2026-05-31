"""Divergence KL sur prompts harmless : préservation des capacités (KB §8.2)."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def kl_divergence(logits_p: torch.Tensor, logits_q: torch.Tensor) -> float:
    """KL(P ‖ Q) moyennée, P/Q = softmax des logits (... , V). P = modèle original."""
    log_p = F.log_softmax(logits_p.float(), dim=-1)
    log_q = F.log_softmax(logits_q.float(), dim=-1)
    p = log_p.exp()
    kl = (p * (log_p - log_q)).sum(dim=-1)   # (...,)
    return float(kl.mean())
