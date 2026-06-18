"""Directions supervisées par sujet, dérivées d'un dataset étiqueté.

Généralise le contraste à deux classes (refus = harmful − harmless) à N sujets. La direction d'un
sujet est la différence de moyennes d'activations entre ce sujet et le reste (one-vs-rest) :

    d̂_s[l] = normalize( μ_s[l] − μ_rest[l] )         (rest = moyenne des AUTRES sujets)

variante `center="grand"` : contre la moyenne globale (μ_grand) plutôt que le complément. On
expose aussi `gap_norm_s[l] = ‖μ_s[l] − μ_rest[l]‖` (force du sujet, AVANT normalisation) : le
signal le plus informatif d'un renforcement/affaiblissement au fil d'un fine-tuning.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch

from meridian.concepts.concept import direction_from_means


def load_labeled(path, label_key: str = "subject") -> dict[str, list[str]]:
    """Charge un dataset étiqueté en `{sujet: [textes]}`. Deux formats acceptés :

    - **fichier JSONL** : une ligne = `{"text"|"prompt": ..., "<label_key>": ...}` ;
    - **dossier** : un fichier `*.txt` (JSONL de textes) par sujet, le nom de fichier = le sujet
      (convention `data/`).
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset étiqueté introuvable : {path}")
    groups: dict[str, list[str]] = {}
    if p.is_dir():
        files = sorted(p.glob("*.txt"))
        if not files:
            raise ValueError(f"Aucun fichier *.txt (un par sujet) sous {path}.")
        for f in files:
            groups[f.stem] = _texts(f)
        return groups
    # Fichier JSONL unique avec clé de label.
    for line in p.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        text = obj.get("text") or obj.get("prompt")
        if text is None:
            raise ValueError(f"ligne sans clé 'text' ni 'prompt' : {obj!r}")
        label = obj.get(label_key)
        if label is None:
            raise ValueError(f"ligne sans clé de label '{label_key}' : {obj!r}")
        groups.setdefault(str(label), []).append(text)
    return groups


def _texts(path: Path) -> list[str]:
    out: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        text = obj.get("text") or obj.get("prompt")
        if text is None:
            raise ValueError(f"ligne sans clé 'text' ni 'prompt' : {obj!r}")
        out.append(text)
    return out


def subject_directions_from_means(means: dict[str, torch.Tensor], center: str = "rest"
                                  ) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Directions one-vs-rest (ou one-vs-grand) à partir des moyennes par sujet.

    `means` : `{sujet: (L+1, H)}`. Renvoie `(names, dirs (S, L+1, H), gap_norms (S, L+1))`,
    `dirs` unitaire par couche, `gap_norms` = norme du contraste avant normalisation.
    """
    names = list(means)
    if len(names) < 2:
        raise ValueError("subject_directions_from_means exige au moins 2 sujets (contraste).")
    stacked = torch.stack([means[n].to(torch.float32) for n in names], dim=0)  # (S, L+1, H)
    grand = stacked.mean(dim=0)                                                 # (L+1, H)
    s = len(names)

    dirs, gaps = [], []
    for i in range(s):
        mu = stacked[i]
        if center == "grand":
            ref = grand
        else:  # rest : moyenne des autres sujets (équilibrée par sujet)
            ref = (stacked.sum(dim=0) - mu) / (s - 1)
        diff = mu - ref
        gaps.append(diff.norm(dim=-1))                 # (L+1,)
        dirs.append(direction_from_means(mu, ref))     # (L+1, H) unitaire
    return names, torch.stack(dirs, dim=0), torch.stack(gaps, dim=0)
