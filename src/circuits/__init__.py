"""Module circuits — analyse circuitielle du refus (Phase 1 : analyse, aucune modif de poids).

DLA (corrélationnel) + activation/attribution patching (causal) + localisation, conformément
au skill `abliteration-circuits`. Règle d'or : une localisation n'est « validée » qu'après
confirmation causale (nécessité + suffisance), jamais sur la DLA seule.
"""
from .backend import (
    CircuitBackend,
    Component,
    ComponentCache,
    ComponentKind,
    ModelInfo,
    NNsightBackend,
    Patch,
    TorchHookBackend,
    make_backend,
)

__all__ = [
    "CircuitBackend",
    "Component",
    "ComponentCache",
    "ComponentKind",
    "ModelInfo",
    "Patch",
    "TorchHookBackend",
    "NNsightBackend",
    "make_backend",
]
