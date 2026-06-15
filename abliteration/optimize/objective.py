"""Objectif composite co-minimisé par l'optimiseur (style Heretic, étendu agentique + négation).

    objectif = refusal_rate
             + λ_kl   · KL(harmless)
             + λ_neg  · (1 − negation_retention)
             + λ_syco · follow_rate                 (sycophantie / capitulation indue)
             + λ_agent· (1 − agentic_score)

Justification des termes étendus : ni la KL ni le taux de refus ne capturent la perte
agentique ou la perte de négation légitime. Sans λ_agent, l'optimiseur peut livrer un modèle
qui hallucine ses tool calls tout en affichant un excellent (refus, KL). Les λ sont exposés en
config ; défauts à 1.0 (compromis neutre, à régler selon le front de Pareto).
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


def build_objective(eval_fn, candidate_layers, lambdas: Lambdas,
                    alpha_low: float = 0.5, alpha_high: float = 1.0):
    """Construit la fonction-objectif Optuna (à MINIMISER).

    À chaque trial : suggère une couche (`candidate_layers`) et une force d'ablation graduée
    `alpha ∈ [alpha_low, alpha_high]`, appelle `eval_fn(layer, alpha) -> metrics` (EvalReport ou
    dict), et renvoie `composite_objective(metrics, lambdas)`.

    `eval_fn` encapsule l'application réelle (hooks d'ablation réversibles) + l'éval bi-axe sur
    le holdout. Il est INJECTÉ pour que la logique d'optimisation soit testable sans modèle :
    le câblage modèle vit dans la CLI (`cmd_optimize`).
    """
    layers = list(candidate_layers)
    if not layers:
        raise ValueError("build_objective : aucune couche candidate.")

    def objective(trial):
        layer = trial.suggest_categorical("layer", layers)
        alpha = trial.suggest_float("alpha", alpha_low, alpha_high)
        metrics = eval_fn(layer, alpha)
        return composite_objective(metrics, lambdas)

    return objective
