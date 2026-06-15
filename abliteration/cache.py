"""Cache disque pour les calculs coûteux et déterministes (activations, logits de base).

Motivation : `extract` recollecte les activations et `eval --kl` recalcule les logits de base à
chaque run. Pour une boucle Optuna de N essais sur un vrai modèle, c'est rédhibitoire. On met en
cache les tenseurs déterministes (mêmes modèle + données + chat template + dtype → même résultat)
sur disque, clés par un hash de ces composantes.

Le cache est opt-out (`--no-cache`) : il n'altère JAMAIS le résultat (mêmes entrées → même sortie),
seulement le temps de calcul. La clé inclut une signature du chat template car celui-ci change les
activations (le chat template doit toujours être appliqué avant la collecte).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Callable

import torch


def default_cache_dir() -> Path:
    """Dossier de cache : $ABLITERATION_CACHE, sinon ~/.cache/abliteration."""
    env = os.environ.get("ABLITERATION_CACHE")
    if env:
        return Path(env)
    return Path.home() / ".cache" / "abliteration"


def make_key(*parts) -> str:
    """Hash SHA-256 (tronqué) stable d'une liste de composantes hétérogènes."""
    h = hashlib.sha256()
    for p in parts:
        h.update(repr(p).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:32]


def cached_tensor(key: str, compute: Callable[[], torch.Tensor], *,
                  enabled: bool = True, cache_dir=None):
    """Renvoie `compute()`, mis en cache sur disque sous `key` si `enabled`.

    Valeur sérialisée via torch.save (tenseur OU dict de tenseurs). Toujours rechargée sur CPU :
    l'appelant la replace sur le device voulu. Un cache corrompu/illisible est ignoré (recalcul).
    """
    if not enabled:
        return compute()
    root = Path(cache_dir) if cache_dir is not None else default_cache_dir()
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{key}.pt"
    if path.exists():
        try:
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception:  # cache corrompu -> on recalcule et on réécrit
            pass
    value = compute()
    tmp = path.with_suffix(".pt.tmp")
    torch.save(value, tmp)
    tmp.replace(path)   # écriture atomique : pas de demi-fichier en cas d'interruption
    return value
