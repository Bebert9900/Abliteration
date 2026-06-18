"""Dérive d'un atlas entre checkpoints : le cœur du suivi de fine-tuning.

Comparer deux atlas d'une **même lignée** (même archi/base → base de hidden states partagée) :
- **sujets** : distance cosinus **signée** `1 − cos(d̂_s^t, d̂_s^ref)` par couche. La direction
  d'un sujet (`μ_s − μ_reste`) est orientée : un retournement de signe est un vrai changement, on
  ne masque donc pas l'orientation (plage [0, 2]) ;
- **sous-espace latent** : les directions latentes ne sont PAS orientées (signe SVD arbitraire)
  et pas identifiables une à une → on compare les **sous-espaces** via les angles principaux
  (distance de Grassmann), invariants au signe et à la base ;
- **force** : variation des `gap_norms` par sujet (renforcement/effondrement du sujet).
"""
from __future__ import annotations

import torch

from .atlas import Atlas, _unit


def subspace_drift(a: torch.Tensor, b: torch.Tensor) -> float:
    """Distance entre les sous-espaces engendrés par les lignes de `a` et `b`.

    `a` `(k1, H)`, `b` `(k2, H)`. Renvoie `1 − moyenne(cos(angles principaux))` ∈ [0, 1] :
    0 = sous-espaces identiques, 1 = orthogonaux. Invariant au choix de base dans chaque
    sous-espace (orthonormalisation QR + valeurs singulières de l'alignement).
    """
    qa = torch.linalg.qr(a.to(torch.float32).T).Q          # (H, k1) colonnes orthonormées
    qb = torch.linalg.qr(b.to(torch.float32).T).Q          # (H, k2)
    cos = torch.linalg.svdvals(qa.T @ qb).clamp(0.0, 1.0)  # cos des angles principaux
    return float(1.0 - cos.mean())


def _subject_drift_per_layer(curr: torch.Tensor, ref: torch.Tensor) -> list[float]:
    """Distance cosinus SIGNÉE `1 − cos` par couche entre deux directions `(L+1, H)` d'un sujet.

    La direction d'un sujet (`μ_s − μ_reste`) est ORIENTÉE : son signe est sémantiquement réel
    (une sonde linéaire le verrait s'inverser). On ne prend donc PAS la valeur absolue, à la
    différence des sous-espaces latents (signe SVD arbitraire). Plage [0, 2] : 0 = identique,
    1 = orthogonal, 2 = retournement à 180°.
    """
    c = _unit(curr.to(torch.float32))
    r = _unit(ref.to(torch.float32))
    cos = (c * r).sum(dim=-1)                              # (L+1,), signé
    return (1.0 - cos).tolist()


def _summary_layers(n_layers: int) -> list[int]:
    """Couches « milieu » pour le résumé : on exclut l'embedding (couche 0) et la dernière, où
    les directions de concepts sont bruitées/dégénérées (gap ≈ 0). Mirror de la bande candidate
    du pipeline d'abliteration (0.4–0.8 de la profondeur). Petits modèles : toutes les couches.
    """
    if n_layers <= 4:
        return list(range(n_layers))
    return sorted({max(1, min(n_layers - 1, round(n_layers * f)))
                   for f in (0.4, 0.5, 0.6, 0.7, 0.8)})


def atlas_drift(curr: Atlas, ref: Atlas, layers: list[int] | None = None) -> dict:
    """Dérive de `curr` par rapport à `ref`. Ne compare que les sujets communs (par nom).

    `per_layer` couvre toutes les couches (transparence) ; `summary` agrège sur les couches
    milieu (`_summary_layers`) afin que l'embedding/les bords dégénérés ne polluent pas la
    métrique. `layers` impose explicitement les couches (report ET résumé).
    """
    n_layers = curr.n_layers
    report = layers if layers is not None else list(range(n_layers))
    summ = layers if layers is not None else _summary_layers(n_layers)

    ref_idx = {name: i for i, name in enumerate(ref.subject_names)}
    subjects: dict[str, dict] = {}
    gap_delta: dict[str, float] = {}
    for i, name in enumerate(curr.subject_names):
        if name not in ref_idx:
            continue
        j = ref_idx[name]
        full = _subject_drift_per_layer(curr.subject_dirs[i], ref.subject_dirs[j])
        subjects[name] = {"per_layer": [full[l] for l in report],
                          "summary": float(sum(full[l] for l in summ) / len(summ))}
        delta = (curr.gap_norms[i] - ref.gap_norms[j])
        gap_delta[name] = float(delta[summ].mean())

    latent_by_layer = {l: subspace_drift(curr.latent_dirs[l], ref.latent_dirs[l])
                       for l in sorted(set(report) | set(summ))}
    latent = {"per_layer": [latent_by_layer[l] for l in report],
              "summary": float(sum(latent_by_layer[l] for l in summ) / len(summ))}

    return {"layers": report, "summary_layers": summ, "subjects": subjects,
            "latent_subspace_drift": latent, "gap_norm_delta": gap_delta}


def drift_series(atlases: list[tuple[str, Atlas]], ref_index: int = 0,
                 layers: list[int] | None = None) -> list[dict]:
    """Série de dérive : chaque atlas comparé à la référence `atlases[ref_index]`.

    `atlases` : liste `(label_checkpoint, Atlas)` dans l'ordre temporel. Renvoie une liste de
    dicts `{checkpoint, ...atlas_drift(...)}`. Le checkpoint de référence a une dérive nulle.
    """
    _, ref = atlases[ref_index]
    return [{"checkpoint": label, **atlas_drift(atlas, ref, layers=layers)}
            for label, atlas in atlases]
