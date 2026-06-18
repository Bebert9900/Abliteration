"""Monitoring *en ligne* de la dérive des directions pendant un fine-tuning (TrainerCallback).

Complément du mode hors-ligne (`atlas-monitor`, qui reconstruit un atlas par checkpoint déjà
sauvegardé) : ici on mesure la dérive **pendant** l'entraînement, sans réécrire les checkpoints.
À brancher sur un `transformers.Trainer` via `trainer.add_callback(...)`.

`transformers` est importé paresseusement (comme `peft` dans `heal.py`) : le base class
`TrainerCallback` n'est résolu qu'à la construction, donc `import meridian.atlas` reste léger.
"""
from __future__ import annotations

import json
from pathlib import Path


def AtlasDriftCallback(groups: dict[str, list[str]], formatter, *, k: int = 32,
                       out_path: str = "atlas_drift.json", center: str = "rest",
                       limit: int | None = None):
    """Crée un `TrainerCallback` qui suit la dérive de l'atlas au fil de l'entraînement.

    À chaque sauvegarde (`on_save`) et en fin d'entraînement, construit un atlas sur le modèle en
    cours (`build_atlas`) et réécrit `out_path` avec la série de dérive (`drift_series`, référence =
    premier instantané). `groups` = `{sujet: [textes]}` (cf. `load_labeled`), `formatter` un
    `PromptFormatter` (ou tout objet exposant `.tokenize`). Lève `RuntimeError` si transformers
    est absent.
    """
    try:
        from transformers import TrainerCallback
    except ImportError as e:  # transformers est une dép. core, mais on reste explicite
        raise RuntimeError(
            "AtlasDriftCallback nécessite `transformers` (Trainer/TrainerCallback)."
        ) from e

    from .atlas import build_atlas
    from .drift import drift_series

    class _AtlasDriftCallback(TrainerCallback):
        def __init__(self):
            self.snapshots: list[tuple[str, object]] = []

        def _snapshot(self, model, step: int) -> None:
            label = f"step-{step}"
            atlas = build_atlas(model, formatter, groups, k, center=center, limit=limit,
                                meta={"checkpoint": label})
            self.snapshots.append((label, atlas))
            series = drift_series(self.snapshots, ref_index=0)
            Path(out_path).write_text(
                json.dumps({"ref": series[0]["checkpoint"], "series": series},
                           indent=2, ensure_ascii=False, default=str),
                encoding="utf-8")

        def on_save(self, args, state, control, model=None, **kwargs):
            if model is not None:
                self._snapshot(model, getattr(state, "global_step", len(self.snapshots)))

        def on_train_end(self, args, state, control, model=None, **kwargs):
            if model is not None:
                self._snapshot(model, getattr(state, "global_step", len(self.snapshots)))

    return _AtlasDriftCallback()
