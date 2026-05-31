"""Objectif composite co-minimisé par l'optimiseur (style Heretic, étendu agentique + négation).

    objectif = refusal_rate
             + λ_kl   · KL(harmless)
             + λ_neg  · (1 − negation_retention)
             + λ_syco · follow_rate                 (sycophantie / capitulation indue)
             + λ_agent· (1 − agentic_score)

Justification des termes étendus : ni la KL ni le taux de refus ne capturent la perte
agentique ou la perte de négation légitime. Sans λ_agent, l'optimiseur peut livrer un modèle
qui hallucine ses tool calls tout en affichant un excellent (refus, KL). Les λ sont exposés en
config ; défauts à 1.0 (compromis neutre, à régler selon le front de Pareto, KB §8.4).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Lambdas:
    kl: float = 1.0
    negation: float = 1.0
    sycophancy: float = 1.0
    agentic: float = 1.0


def _get(metrics, name):
    return getattr(metrics, name) if hasattr(metrics, name) else metrics[name]


def composite_objective(metrics, lambdas: Lambdas) -> float:
    """`metrics` : EvalReport ou dict avec les clés refusal_rate/kl/negation_retention/..."""
    return (
        _get(metrics, "refusal_rate")
        + lambdas.kl * _get(metrics, "kl")
        + lambdas.negation * (1.0 - _get(metrics, "negation_retention"))
        + lambdas.sycophancy * _get(metrics, "follow_rate")
        + lambdas.agentic * (1.0 - _get(metrics, "agentic_score"))
    )
