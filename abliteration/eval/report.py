"""Rapport d'évaluation : deux axes (refus / préservation) + sérialisation JSON."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class EvalReport:
    """Agrège les métriques des deux axes (KB §8) pour un modèle donné.

    Axe 1 (suppression du refus) : refusal_rate.
    Axe 2 (préservation) : kl, negation_retention, agentic_score.
    Garde-fous anti-gaming : degeneracy_rate, empty_rate, follow_rate (sycophantie).
    """
    refusal_rate: float
    kl: float
    negation_retention: float
    follow_rate: float
    empty_rate: float
    agentic_score: float
    degeneracy_rate: float
    benchmarks: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, ensure_ascii=False))
