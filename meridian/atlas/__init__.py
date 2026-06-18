"""Atlas de directions : identifier la direction de n'importe quel sujet, et la suivre.

Sous-système autonome (distinct de l'abliteration) qui combine deux paradigmes :
- supervisé : une direction par sujet, dérivée d'un dataset étiqueté (`subjects`) ;
- non supervisé : un jeu de directions latentes par variance (`discover`) ;
puis un pont entre les deux et la sérialisation d'un `Atlas` (`atlas`), et le suivi de la dérive
de cet atlas à travers les checkpoints d'un fine-tuning (`drift`).
"""
from .atlas import Atlas, build_atlas, build_atlas_from_group_acts, load_atlas, save_atlas
from .callback import AtlasDriftCallback
from .discover import discover_directions
from .drift import atlas_drift, drift_series, subspace_drift
from .subjects import load_labeled, subject_directions_from_means

__all__ = [
    "Atlas",
    "build_atlas",
    "build_atlas_from_group_acts",
    "load_atlas",
    "save_atlas",
    "AtlasDriftCallback",
    "discover_directions",
    "load_labeled",
    "subject_directions_from_means",
    "atlas_drift",
    "drift_series",
    "subspace_drift",
]
