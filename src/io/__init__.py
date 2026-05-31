"""Module IO : sauvegarde HF safetensors + model card transparente (export GGUF dans export_gguf)."""
from __future__ import annotations

from pathlib import Path


def save_model(model, tokenizer, out_dir) -> None:
    """Sauvegarde le modèle abliteré en safetensors (jamais des poids mesurés en 4-bit, KB)."""
    out = str(out_dir)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)


def build_model_card(base_model: str, variant: str, preserve, metrics: dict) -> str:
    """Model card transparente : modèle de base + méthode + métriques (exigé par le CLAUDE.md)."""
    preserve = list(preserve or [])
    lines = [
        f"# {base_model} — abliteré ({variant})",
        "",
        "## Méthode",
        f"- Modèle de base : `{base_model}`",
        f"- Variante d'abliteration : `{variant}`",
        f"- Directions préservées (orthogonalisation contre) : {preserve or 'aucune'}",
        "",
        "## Métriques d'évaluation (holdout, protocole honnête)",
    ]
    lines += [f"- `{k}` : {v}" for k, v in metrics.items()]
    lines += [
        "",
        "## Cadre responsable",
        "Modèle produit par abliteration (directional ablation, Arditi et al. 2024). Technique "
        "**dual-use** : outil générique de modification de modèle. Voir les métriques ci-dessus "
        "pour l'impact mesuré sur le refus ET la préservation des capacités.",
    ]
    return "\n".join(lines)


def write_model_card(out_dir, card: str) -> None:
    Path(out_dir).joinpath("README.md").write_text(card, encoding="utf-8")
