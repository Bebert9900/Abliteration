"""Boucle d'optimisation Optuna (TPE) + checkpoint resume-safe.

Optuna est une dépendance optionnelle (`pip install optuna`). On sauve un checkpoint après
CHAQUE trial (JSONL append) → reprise Ctrl+C-safe. Le câblage de l'espace de recherche
(direction_index, kernel de poids d'ablation par composant — KB §3.6) est passé via `space`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable


def save_trial(path, trial: dict) -> None:
    with Path(path).open("a", encoding="utf-8") as f:
        f.write(json.dumps(trial, ensure_ascii=False) + "\n")


def load_trials(path) -> list[dict]:
    p = Path(path)
    if not p.exists():
        return []
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def best_trial(trials: list[dict], key: str = "objective") -> dict:
    return min(trials, key=lambda t: t[key])


def run_optuna_study(
    objective: Callable,
    space: dict,
    n_trials: int,
    checkpoint_path,
) -> dict:
    """Lance une étude TPE. Lève RuntimeError clair si optuna est absent.

    `objective(trial) -> float` ; chaque trial est checkpointé. Reprend les trials existants
    depuis `checkpoint_path` s'il existe.
    """
    try:
        import optuna
    except ImportError as e:
        raise RuntimeError(
            "L'optimisation automatique requiert optuna (absent). Installer : pip install optuna"
        ) from e

    existing = load_trials(checkpoint_path) if checkpoint_path else []
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler())

    def _wrapped(trial):
        value = objective(trial)
        if checkpoint_path:
            save_trial(checkpoint_path, {"params": dict(trial.params), "objective": value})
        return value

    study.optimize(_wrapped, n_trials=n_trials)
    return {"params": study.best_params, "objective": study.best_value, "resumed": len(existing)}
