"""Sélection de la couche d'ablation.

La sélection réelle pose un hook d'ablation réversible (cf. src.ablation) à chaque couche
candidate et mesure le refus ; on garde la couche qui le minimise (couches milieu→milieu-tardif
en général, KB §2.1). Ici on découple la stratégie de la mesure via une `score_fn` injectée.
"""
from __future__ import annotations

from typing import Callable


def select_layer(candidate_layers: list[int], score_fn: Callable[[int], float]) -> int:
    """Renvoie la couche au score (refus) le plus bas."""
    if not candidate_layers:
        raise ValueError("aucune couche candidate")
    return min(candidate_layers, key=score_fn)
