"""Adaptateur d'architecture : localise les écritures au residual stream.

Règle de conception : **ne jamais coder les noms de modules en dur**. On balaye
`named_modules()`, on matche par suffixe (table par défaut couvrant les familles courantes), et
on récupère l'embedding via `get_input_embeddings()` (fiable cross-archi). Gère MoE (un
`down_proj` par expert + experts partagés, tous captés par suffixe) et Conv1D (GPT-2, poids
transposé → drapeau `is_conv1d` pour que l'orthogonalisation adapte l'axe).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import torch.nn as nn


class WriteKind(Enum):
    ATTENTION_OUT = "attention_out"
    MLP_OUT = "mlp_out"
    EMBEDDING = "embedding"


# Suffixes de noms de modules par catégorie (overridables par famille si besoin).
ATTENTION_OUT_SUFFIXES = ("o_proj", "out_proj", "wo")
MLP_OUT_SUFFIXES = ("down_proj", "w2")
# GPT-2 nomme `c_proj` à la fois dans l'attention et le MLP → désambiguïsé par le contexte.
GPT2_PROJ_SUFFIX = "c_proj"


@dataclass(frozen=True)
class WriteTarget:
    name: str
    module: nn.Module
    kind: WriteKind
    is_conv1d: bool


def _is_conv1d(module: nn.Module) -> bool:
    # On évite d'importer transformers : on teste le nom de classe.
    return type(module).__name__ == "Conv1D"


class ArchAdapter:
    def __init__(self, model: nn.Module):
        self.model = model

    def residual_writers(self) -> list[WriteTarget]:
        targets: list[WriteTarget] = []

        # Embedding : via l'API HF, pas par nom.
        embed = self.model.get_input_embeddings()
        targets.append(WriteTarget("<embedding>", embed, WriteKind.EMBEDDING, _is_conv1d(embed)))

        for name, module in self.model.named_modules():
            kind = self._classify(name, module)
            if kind is not None:
                targets.append(WriteTarget(name, module, kind, _is_conv1d(module)))
        return targets

    def _classify(self, name: str, module: nn.Module) -> WriteKind | None:
        leaf = name.rsplit(".", 1)[-1]
        if leaf in ATTENTION_OUT_SUFFIXES:
            return WriteKind.ATTENTION_OUT
        if leaf in MLP_OUT_SUFFIXES:
            return WriteKind.MLP_OUT
        if leaf == GPT2_PROJ_SUFFIX:
            # Désambiguïser par le module parent.
            if ".attn." in name or name.endswith("attn.c_proj"):
                return WriteKind.ATTENTION_OUT
            if ".mlp." in name or name.endswith("mlp.c_proj"):
                return WriteKind.MLP_OUT
        return None
