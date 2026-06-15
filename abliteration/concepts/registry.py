"""Registre de concepts prédéfinis + chargement de concepts ad hoc.

Les concepts prédéfinis ré-expriment les contrastes du dépôt (refus/négation/agentique vs
harmless) ; ils sont versionnés et reproductibles. Un chercheur peut aussi définir un concept
ad hoc en fournissant deux fichiers JSONL (positif/négatif), sans rien enregistrer.
"""
from __future__ import annotations

import json
from pathlib import Path

from .concept import Concept

# nom -> (fichier positif, fichier négatif, description). Fichiers relatifs à --data-dir.
BUILTIN_CONCEPTS: dict[str, tuple[str, str, str]] = {
    "refusal": ("harmful.txt", "harmless.txt",
                "Refus moral : harmful (active) vs harmless (référence). Direction canonique."),
    "negation": ("legitimate_negation.txt", "harmless.txt",
                 "Négation logique légitime (« non, c'est faux ») vs harmless."),
    "agentic": ("agentic.txt", "harmless.txt",
                "Capacités agentiques (tool use, schéma strict) vs harmless."),
}


def available_concepts() -> list[str]:
    """Noms des concepts prédéfinis (pour --concept choices et la découverte)."""
    return sorted(BUILTIN_CONCEPTS)


def _load_texts(path) -> list[str]:
    """Charge les textes d'un JSONL (clé `text` ou `prompt`). Ignore les lignes vides."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Fichier de prompts introuvable : {path}")
    texts: list[str] = []
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        text = obj.get("text") or obj.get("prompt")
        if text is None:
            raise ValueError(f"ligne sans clé 'text' ni 'prompt' : {obj!r}")
        texts.append(text)
    return texts


def load_concept(name: str, data_dir="data") -> Concept:
    """Charge un concept prédéfini depuis le registre. Erreur claire si nom inconnu."""
    if name not in BUILTIN_CONCEPTS:
        raise KeyError(f"Concept inconnu : '{name}'. Disponibles : {available_concepts()}.")
    pos_file, neg_file, desc = BUILTIN_CONCEPTS[name]
    base = Path(data_dir)
    return Concept(name, _load_texts(base / pos_file), _load_texts(base / neg_file), desc)


def load_concept_from_files(name: str, pos_path, neg_path, description="") -> Concept:
    """Charge un concept ad hoc depuis deux fichiers JSONL (positif/négatif)."""
    return Concept(name, _load_texts(pos_path), _load_texts(neg_path),
                   description or f"Concept ad hoc '{name}'.")
