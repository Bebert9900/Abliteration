"""Cœur pur du build (activations -> Atlas) et de la série de dérive (suivi multi-checkpoints)."""
import torch

from meridian.atlas import Atlas, build_atlas, build_atlas_from_group_acts, drift_series


class _IdFormatter:
    """tokenize(texts) -> un token par texte, id donné par une table (1 couche en aval)."""

    def __init__(self, table):
        self.table = table

    def tokenize(self, texts):
        ids = torch.tensor([[self.table[t]] for t in texts])
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


class _Output:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class _PointModel:
    """hidden_states : 1 couche, H=2 ; hidden[b,0] = [id_b, 0]. La variance d'ids -> SVD non triviale."""

    def __call__(self, input_ids, attention_mask, output_hidden_states):
        x = input_ids.float()                          # (B, 1)
        base = torch.cat([x, torch.zeros_like(x)], dim=-1).unsqueeze(1)  # (B, 1, 2)
        return _Output((base,))


def test_build_atlas_wires_groups_and_respects_limit():
    groups = {"a": ["a1", "a2", "a3"], "b": ["b1", "b2"]}
    table = {"a1": 10, "a2": 12, "a3": 14, "b1": 2, "b2": 4}
    atlas = build_atlas(_PointModel(), _IdFormatter(table), groups, k=2, batch_size=8)
    assert atlas.subject_names == ["a", "b"]
    assert atlas.subject_dirs.shape == (2, 1, 2)
    # 'a' (ids ~12) vs 'b' (ids ~3) -> contraste le long de l'axe x.
    assert abs(float(atlas.subject_dirs[0, 0] @ torch.tensor([1.0, 0.0]))) > 0.99

    limited = build_atlas(_PointModel(), _IdFormatter(table), groups, k=2, limit=1)
    assert limited.meta["n_per_subject"] == {"a": 1, "b": 1}


def _group_acts():
    g = torch.Generator().manual_seed(0)
    # 1 couche, H=2 ; sujet 'a' concentré près de [2,0], 'b' près de [0,0].
    a = (torch.tensor([2.0, 0.0]) + 0.05 * torch.randn(30, 2, generator=g)).unsqueeze(0)  # (1,30,2)
    b = (torch.tensor([0.0, 0.0]) + 0.05 * torch.randn(30, 2, generator=g)).unsqueeze(0)
    return {"a": a, "b": b}


def test_build_atlas_from_group_acts_shapes_and_direction():
    atlas = build_atlas_from_group_acts(_group_acts(), k=2, meta={"model": "toy"})
    assert isinstance(atlas, Atlas)
    assert atlas.subject_names == ["a", "b"]
    assert atlas.subject_dirs.shape == (2, 1, 2)
    assert atlas.latent_dirs.shape == (1, 2, 2)
    assert atlas.explained_variance.shape == (1, 2)
    # 'a' (≈[2,0]) vs rest (≈[0,0]) -> direction ≈ [1,0].
    assert abs(float(atlas.subject_dirs[0, 0] @ torch.tensor([1.0, 0.0]))) > 0.99
    assert atlas.meta["model"] == "toy"


def _toy_atlas(scale_a):
    return build_atlas_from_group_acts({
        "a": (torch.tensor([scale_a, 0.0]) + 0.01 * torch.arange(20).float().reshape(20, 1)).unsqueeze(0),
        "b": (torch.tensor([0.0, 1.0]) + 0.01 * torch.arange(20).float().reshape(20, 1)).unsqueeze(0),
    }, k=2)


def test_drift_series_first_checkpoint_is_reference_with_zero_drift():
    atlases = [("ckpt-0", _toy_atlas(2.0)), ("ckpt-1", _toy_atlas(2.0))]
    series = drift_series(atlases, ref_index=0)
    assert len(series) == 2
    assert series[0]["checkpoint"] == "ckpt-0"
    # checkpoint identique à la référence -> dérive nulle des sujets.
    assert series[1]["subjects"]["a"]["summary"] < 1e-5
