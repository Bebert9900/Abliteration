"""Les quatre classes contrastives du projet.

- HARMFUL / HARMLESS : contraste classique d'Arditi et al. 2024.
- LEGITIMATE_NEGATION : négation logique légitime à PRÉSERVER (« non, ce code est faux »).
- AGENTIC : capacités agentiques à PRÉSERVER (tool use, multi-étapes, schéma strict).

LEGITIMATE_NEGATION et AGENTIC servent de directions « à préserver » contre lesquelles on
orthogonalise la direction de refus (généralisation de la projected abliteration).
"""
from enum import Enum


class PromptClass(str, Enum):
    HARMFUL = "harmful"
    HARMLESS = "harmless"
    LEGITIMATE_NEGATION = "legitimate_negation"
    AGENTIC = "agentic"
