"""AtlasDriftCallback : monitoring en ligne pendant un fine-tuning (TrainerCallback)."""
import json
from types import SimpleNamespace

import torch

from meridian.atlas import AtlasDriftCallback


class _IdFormatter:
    def __init__(self, table):
        self.table = table

    def tokenize(self, texts):
        ids = torch.tensor([[self.table[t]] for t in texts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class _Output:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _PointModel:
    def __call__(self, input_ids, attention_mask, output_hidden_states):
        x = input_ids.float()
        base = torch.cat([x, torch.zeros_like(x)], dim=-1).unsqueeze(1)   # (B, 1, 2)
        return _Output((base,))


def _cb(out_path):
    groups = {"a": ["a1", "a2", "a3"], "b": ["b1", "b2"]}
    table = {"a1": 10, "a2": 12, "a3": 14, "b1": 2, "b2": 4}
    return AtlasDriftCallback(groups, _IdFormatter(table), k=2, out_path=out_path)


def test_callback_accumulates_snapshots_and_writes_zero_drift(tmp_path):
    out = tmp_path / "drift.json"
    cb = _cb(str(out))
    cb.on_save(None, SimpleNamespace(global_step=10), None, model=_PointModel())
    cb.on_save(None, SimpleNamespace(global_step=20), None, model=_PointModel())

    data = json.loads(out.read_text())
    assert data["ref"] == "step-10"
    assert len(data["series"]) == 2
    assert data["series"][0]["checkpoint"] == "step-10"
    # même modèle aux deux étapes -> dérive nulle (canari).
    assert data["series"][1]["subjects"]["a"]["summary"] < 1e-5


def test_callback_ignores_save_without_model(tmp_path):
    out = tmp_path / "drift.json"
    cb = _cb(str(out))
    cb.on_save(None, SimpleNamespace(global_step=10), None)   # pas de model -> no-op
    assert not out.exists()


def test_callback_is_a_trainer_callback():
    from transformers import TrainerCallback
    assert isinstance(_cb("/tmp/_unused.json"), TrainerCallback)
