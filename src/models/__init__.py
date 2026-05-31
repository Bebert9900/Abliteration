"""Module modèles : chargement (import paresseux de transformers) + ArchAdapter."""
from .adapter import ArchAdapter, WriteKind, WriteTarget

__all__ = ["ArchAdapter", "WriteKind", "WriteTarget", "load_model"]


def load_model(model_id: str, dtype: str = "bfloat16", device_map: str = "auto", **kwargs):
    """Charge un modèle HF causal en bf16 pour l'ablation finale (KB : bf16 requis).

    Import paresseux : le paquet reste importable sans transformers (pour les tests unitaires).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=getattr(torch, dtype),
        device_map=device_map,
        output_hidden_states=True,
        **kwargs,
    )
    return model, tok
