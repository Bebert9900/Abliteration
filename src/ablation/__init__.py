"""Module ablation : projection de directions, orthogonalisation de poids, hooks réversibles."""
from .hooks import make_ablation_hook, register_ablation_hooks
from .project import Variant, ablation_direction, project_out
from .weights import orthogonalize_weights

__all__ = [
    "project_out",
    "Variant",
    "ablation_direction",
    "orthogonalize_weights",
    "make_ablation_hook",
    "register_ablation_hooks",
]
