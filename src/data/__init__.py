"""Module de données : 4 classes contrastives, chargement, holdout, mise en forme."""
from .classes import PromptClass
from .dataset import FourClassData, Prompt, load_prompts, split_holdout
from .formatting import PromptFormatter, last_token_index

__all__ = [
    "PromptClass",
    "Prompt",
    "FourClassData",
    "load_prompts",
    "split_holdout",
    "PromptFormatter",
    "last_token_index",
]
