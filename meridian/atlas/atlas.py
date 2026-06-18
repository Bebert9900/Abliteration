"""Objet `Atlas` : collection de directions (sujets supervisés + latents non supervisés).

Réunit les deux paradigmes sur un même modèle : directions par sujet (`subjects`) et directions
latentes (`discover`), plus le **pont** entre les deux (quel latent ≈ quel sujet, et inversement)
et l'**identification** d'une direction quelconque (« la direction de n'importe quel sujet »).

Sérialisé en `.safetensors` (directions seulement, jamais de poids de modèle) : tenseurs + noms
et métadonnées (modèle, dataset, seed, k, méthode) dans les métadonnées du fichier.

Les cosinus utilisent la **valeur absolue** : une direction n'est définie qu'au signe près (vrai
pour les contrastes de moyennes comme pour les vecteurs singuliers de la SVD).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch

from .discover import discover_directions
from .subjects import subject_directions_from_means


@dataclass
class Atlas:
    subject_names: list[str]
    subject_dirs: torch.Tensor          # (S, L+1, H), unitaire par couche
    gap_norms: torch.Tensor             # (S, L+1), force de chaque sujet (avant normalisation)
    latent_dirs: torch.Tensor           # (L+1, k, H), unitaire
    explained_variance: torch.Tensor    # (L+1, k)
    meta: dict = field(default_factory=dict)

    @property
    def n_layers(self) -> int:
        return self.subject_dirs.shape[1]

    @property
    def n_subjects(self) -> int:
        return self.subject_dirs.shape[0]

    @property
    def k(self) -> int:
        return self.latent_dirs.shape[1]

    def _layer(self, layer: int | None) -> int:
        return self.n_layers // 2 if layer is None else layer

    def match_subjects_to_latents(self, layer: int | None = None) -> dict:
        """Pour chaque sujet, la direction latente la plus proche (|cosinus|), et l'inverse."""
        l = self._layer(layer)
        subj = _unit(self.subject_dirs[:, l, :])      # (S, H)
        lat = _unit(self.latent_dirs[l])              # (k, H)
        cos = (subj @ lat.T).abs()                    # (S, k)

        subject_to_latent = {}
        for i, name in enumerate(self.subject_names):
            j = int(cos[i].argmax())
            subject_to_latent[name] = {"latent_index": j, "cosine": float(cos[i, j])}
        latent_to_subject = []
        for j in range(lat.shape[0]):
            i = int(cos[:, j].argmax())
            latent_to_subject.append({"latent_index": j,
                                      "subject": self.subject_names[i],
                                      "cosine": float(cos[i, j])})
        return {"layer": l, "subject_to_latent": subject_to_latent,
                "latent_to_subject": latent_to_subject}

    def identify(self, query: torch.Tensor, k: int = 5, layer: int | None = None) -> list[dict]:
        """Sujets dont la direction est la plus proche de `query` (|cosinus| décroissant).

        `query` : vecteur `(H,)` ou directions par couche `(L+1, H)` (on lit la couche `layer`).
        """
        l = self._layer(layer)
        q = query.to(torch.float32)
        if q.ndim == 2:
            q = q[l]
        q = q / (q.norm() + 1e-8)
        subj = _unit(self.subject_dirs[:, l, :])      # (S, H)
        cos = (subj @ q).abs()                        # (S,)
        order = torch.argsort(cos, descending=True)[:k]
        return [{"name": self.subject_names[int(i)], "cosine": float(cos[int(i)])} for i in order]


def _unit(x: torch.Tensor) -> torch.Tensor:
    return x / (x.norm(dim=-1, keepdim=True) + 1e-8)


def build_atlas_from_group_acts(per_group: dict[str, torch.Tensor], k: int,
                                center: str = "rest", meta: dict | None = None) -> Atlas:
    """Construit un `Atlas` à partir des activations par exemple de chaque sujet (cœur pur).

    `per_group` : `{sujet: (L+1, n_sujet, H)}`. Dérive en UNE passe les directions supervisées
    (moyenne par sujet → one-vs-rest) ET la base latente non supervisée (SVD sur tous les
    exemples concaténés). Aucune dépendance modèle : testable directement.
    """
    means = {name: acts.mean(dim=1) for name, acts in per_group.items()}   # {sujet: (L+1, H)}
    names, subject_dirs, gaps = subject_directions_from_means(means, center=center)
    all_acts = torch.cat([per_group[n] for n in names], dim=1)             # (L+1, N, H)
    latent_dirs, explained = discover_directions(all_acts, k)
    meta = dict(meta or {})
    meta.setdefault("k", int(latent_dirs.shape[1]))
    meta.setdefault("center", center)
    meta.setdefault("n_per_subject", {n: int(per_group[n].shape[1]) for n in names})
    return Atlas(names, subject_dirs, gaps, latent_dirs, explained, meta)


def build_atlas(model, formatter, groups: dict[str, list[str]], k: int, *, batch_size: int = 8,
                device=None, center: str = "rest", limit: int | None = None,
                meta: dict | None = None) -> Atlas:
    """Glue modèle : collecte les activations par sujet puis appelle `build_atlas_from_group_acts`.

    `groups` : `{sujet: [textes]}` (cf. `subjects.load_labeled`). `limit` sous-échantillonne les
    textes par sujet (journalisé par l'appelant). Réutilise `collect_per_example_activations`.
    """
    from meridian.directions import collect_per_example_activations

    per_group = {}
    for name, texts in groups.items():
        ts = texts[:limit] if limit else texts
        per_group[name] = collect_per_example_activations(
            model, formatter, ts, batch_size=batch_size, device=device)
    return build_atlas_from_group_acts(per_group, k, center=center, meta=meta)


def save_atlas(atlas: Atlas, path: str) -> None:
    """Écrit l'atlas en `.safetensors` (tenseurs + noms/méta en métadonnées JSON)."""
    from safetensors.torch import save_file

    tensors = {
        "subject_dirs": atlas.subject_dirs.contiguous().to(torch.float32),
        "gap_norms": atlas.gap_norms.contiguous().to(torch.float32),
        "latent_dirs": atlas.latent_dirs.contiguous().to(torch.float32),
        "explained_variance": atlas.explained_variance.contiguous().to(torch.float32),
    }
    metadata = {"subject_names": json.dumps(atlas.subject_names, ensure_ascii=False),
                "meta": json.dumps(atlas.meta, ensure_ascii=False, default=str)}
    save_file(tensors, path, metadata=metadata)


def load_atlas(path: str) -> Atlas:
    """Recharge un atlas écrit par `save_atlas`."""
    from safetensors import safe_open

    with safe_open(path, framework="pt") as f:
        md = f.metadata() or {}
        tensors = {key: f.get_tensor(key) for key in f.keys()}
    return Atlas(
        subject_names=json.loads(md.get("subject_names", "[]")),
        subject_dirs=tensors["subject_dirs"],
        gap_norms=tensors["gap_norms"],
        latent_dirs=tensors["latent_dirs"],
        explained_variance=tensors["explained_variance"],
        meta=json.loads(md.get("meta", "{}")),
    )
