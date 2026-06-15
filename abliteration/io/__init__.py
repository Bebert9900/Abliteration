"""Module IO : sauvegarde HF safetensors + model card transparente (export GGUF dans export_gguf)."""
from __future__ import annotations

from pathlib import Path


def save_model(model, tokenizer, out_dir) -> None:
    """Sauvegarde le modèle abliteré en safetensors (jamais des poids mesurés en 4-bit, KB)."""
    out = str(out_dir)
    model.save_pretrained(out, safe_serialization=True)
    tokenizer.save_pretrained(out)


def _axis_table(report) -> list[str]:
    """Tableau Markdown bi-axe (refus / préservation) depuis un EvalReport."""
    g = (lambda n: getattr(report, n, None)) if not isinstance(report, dict) else report.get
    rows = [
        ("Suppression du refus", "Taux de refus (holdout harmful)", g("refusal_rate")),
        ("Préservation", "KL(base‖abliteré) sur harmless", g("kl")),
        ("Préservation", "Rétention de la négation légitime", g("negation_retention")),
        ("Préservation", "Score agentique (validité tool call)", g("agentic_score")),
        ("Garde-fou", "Taux de sycophantie (follow_rate)", g("follow_rate")),
        ("Garde-fou", "Taux de dégénérescence", g("degeneracy_rate")),
        ("Garde-fou", "Taux de réponses vides", g("empty_rate")),
    ]
    out = ["| Axe | Métrique | Valeur |", "|---|---|---|"]
    for axis, name, val in rows:
        if val is not None:
            out.append(f"| {axis} | {name} | {val:.4f} |" if isinstance(val, (int, float))
                       else f"| {axis} | {name} | {val} |")
    return out


def build_model_card(base_model: str, variant: str, preserve, metrics: dict,
                     report=None, run_config: dict | None = None) -> str:
    """Model card transparente : modèle de base + méthode + métriques bi-axe + config repro.

    `report` (EvalReport ou dict) : rendu en tableau lisible refus/préservation/garde-fous.
    `run_config` : hyperparamètres + empreintes de données, pour la reproductibilité.
    Exigée par le cadre du projet : jamais de livraison sans model card.
    """
    preserve = list(preserve or [])
    lines = [
        f"# {base_model} — abliteré ({variant})",
        "",
        "## Méthode",
        f"- Modèle de base : `{base_model}`",
        f"- Variante d'abliteration : `{variant}`",
        f"- Directions préservées (orthogonalisation contre) : {preserve or 'aucune'}",
    ]
    if run_config and run_config.get("selected_layer") is not None:
        lines.append(f"- Couche sélectionnée : {run_config['selected_layer']}")
    if run_config and run_config.get("alpha") is not None:
        lines.append(f"- Force d'ablation (alpha) : {run_config['alpha']}")

    lines += ["", "## Métriques d'évaluation (holdout, protocole honnête)"]
    if report is not None:
        lines += _axis_table(report)
    else:
        lines += [f"- `{k}` : {v}" for k, v in metrics.items()]

    if run_config:
        import json
        lines += [
            "",
            "## Configuration reproductible",
            "```json",
            json.dumps(run_config, indent=2, ensure_ascii=False, default=str),
            "```",
        ]

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
