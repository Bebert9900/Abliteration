"""Module concepts : abstraction d'un comportement arbitraire (recherche fondamentale).

Un `Concept` est un contraste de prompts (positif/négatif) ; sa direction par couche généralise
la direction de refus. Registre de concepts prédéfinis + concepts ad hoc, plus la séparabilité
géométrique entre concepts. L'abliteration du refus devient le concept prédéfini `refusal`.
"""
from .concept import (
    Concept,
    ConceptDirection,
    concept_direction,
    direction_from_means,
)
from .probing import ProbeReport, probe_per_layer, train_linear_probe
from .registry import (
    BUILTIN_CONCEPTS,
    available_concepts,
    load_concept,
    load_concept_from_files,
)
from .separability import SeparabilityMatrix, pairwise_separability

__all__ = [
    "ProbeReport",
    "probe_per_layer",
    "train_linear_probe",
    "Concept",
    "ConceptDirection",
    "concept_direction",
    "direction_from_means",
    "BUILTIN_CONCEPTS",
    "available_concepts",
    "load_concept",
    "load_concept_from_files",
    "SeparabilityMatrix",
    "pairwise_separability",
]
