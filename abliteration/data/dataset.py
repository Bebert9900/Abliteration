"""Prompt, chargement de fichiers, et découpage holdout déterministe par classe."""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass, field
from pathlib import Path

from .classes import PromptClass


def _class_seed(seed: int, klass: PromptClass) -> int:
    """Décalage de graine par classe, DÉTERMINISTE entre processus.

    ⚠️ Ne PAS utiliser le `hash()` builtin : il est randomisé par processus (PYTHONHASHSEED),
    ce qui rendait le holdout non reproductible d'un run à l'autre (splits différents → toute
    comparaison de refus/préservation entre variantes était invalide). On hashe via hashlib.
    """
    digest = hashlib.sha256(klass.value.encode("utf-8")).digest()
    return seed + int.from_bytes(digest[:4], "big") % 1000


@dataclass(frozen=True)
class Prompt:
    """Un prompt étiqueté par classe.

    `meta` porte les infos spécifiques à une classe sans alourdir les autres — p. ex. pour
    AGENTIC : schéma d'outil attendu, appel de fonction de référence, retour d'outil à gérer.
    """
    text: str
    cls: PromptClass
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class FourClassData:
    """Conteneur des 4 classes, chacune découpée en (train, holdout) disjoints.

    `train` sert à calculer les directions ; `holdout` à évaluer le refus sans sur-estimation.
    """
    _train: dict[PromptClass, list[Prompt]]
    _holdout: dict[PromptClass, list[Prompt]]

    @classmethod
    def load(
        cls, paths: dict[PromptClass, object], holdout_fraction: float, seed: int
    ) -> "FourClassData":
        missing = [c for c in PromptClass if c not in paths]
        if missing:
            raise ValueError(f"classe(s) manquante(s) : {[c.value for c in missing]}")
        train, holdout = {}, {}
        for klass in PromptClass:
            prompts = load_prompts(paths[klass], klass)
            # seed décalé par classe : découpages indépendants mais reproductibles.
            tr, ho = split_holdout(prompts, holdout_fraction, _class_seed(seed, klass))
            train[klass], holdout[klass] = tr, ho
        return cls(_train=train, _holdout=holdout)

    def train(self, klass: PromptClass) -> list[Prompt]:
        return self._train[klass]

    def holdout(self, klass: PromptClass) -> list[Prompt]:
        return self._holdout[klass]


def load_prompts(path, cls: PromptClass) -> list[Prompt]:
    """Charge des prompts `.jsonl` pour une classe donnée.

    Chaque ligne est un objet JSON contenant `text` (ou `prompt`) ; toute autre clé est
    conservée dans `meta` (utile pour AGENTIC : outil/arguments attendus). Les lignes vides
    sont ignorées.
    """
    prompts: list[Prompt] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        meta = dict(obj)
        text = meta.pop("text", None) or meta.pop("prompt", None)
        if text is None:
            raise ValueError(f"ligne sans clé 'text' ni 'prompt' : {obj!r}")
        prompts.append(Prompt(text=text, cls=cls, meta=meta))
    return prompts


def split_holdout(
    prompts: list[Prompt], holdout_fraction: float, seed: int
) -> tuple[list[Prompt], list[Prompt]]:
    """Découpe en (train, holdout) de façon déterministe et disjointe.

    Le holdout sert à mesurer le refus sur des prompts JAMAIS utilisés pour calculer la
    direction (sinon on sur-estime le succès). Déterministe pour la reproductibilité.
    """
    if not 0.0 <= holdout_fraction <= 1.0:
        raise ValueError(f"holdout_fraction doit être dans [0, 1], reçu {holdout_fraction}")
    indices = list(range(len(prompts)))
    random.Random(seed).shuffle(indices)
    n_holdout = int(round(len(prompts) * holdout_fraction))
    holdout_idx = set(indices[:n_holdout])
    holdout = [p for i, p in enumerate(prompts) if i in holdout_idx]
    train = [p for i, p in enumerate(prompts) if i not in holdout_idx]
    return train, holdout
