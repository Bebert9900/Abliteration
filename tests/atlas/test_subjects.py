"""Directions supervisées par sujet (one-vs-rest) + chargement d'un dataset étiqueté."""
import json

import pytest
import torch

from meridian.atlas import load_labeled, subject_directions_from_means


def _means(d):
    """dict {sujet: liste de couches} -> {sujet: tenseur (L+1, H)}."""
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in d.items()}


def test_one_vs_rest_direction_and_gap_norm():
    means = _means({"a": [[2.0, 0.0]], "b": [[0.0, 0.0]], "c": [[0.0, 0.0]]})
    names, dirs, gaps = subject_directions_from_means(means, center="rest")

    assert names == ["a", "b", "c"]
    assert dirs.shape == (3, 1, 2)
    # a : rest = moyenne(b, c) = 0 -> direction = [1, 0], gap = 2.
    assert torch.allclose(dirs[0, 0], torch.tensor([1.0, 0.0]), atol=1e-5)
    assert abs(float(gaps[0, 0]) - 2.0) < 1e-5
    # b : rest = moyenne(a, c) = [1, 0] -> direction = [-1, 0], gap = 1.
    assert torch.allclose(dirs[1, 0], torch.tensor([-1.0, 0.0]), atol=1e-5)
    assert abs(float(gaps[1, 0]) - 1.0) < 1e-5


def test_center_grand_uses_global_mean():
    means = _means({"a": [[2.0, 0.0]], "b": [[0.0, 0.0]], "c": [[0.0, 0.0]]})
    _, dirs, gaps = subject_directions_from_means(means, center="grand")
    # grand = [2/3, 0] ; a - grand = [4/3, 0] -> [1, 0], gap = 4/3.
    assert torch.allclose(dirs[0, 0], torch.tensor([1.0, 0.0]), atol=1e-5)
    assert abs(float(gaps[0, 0]) - 4.0 / 3.0) < 1e-5


def test_directions_unit_norm_per_layer():
    means = _means({"a": [[3.0, 4.0], [1.0, 0.0]], "b": [[0.0, 0.0], [0.0, 2.0]]})
    _, dirs, _ = subject_directions_from_means(means)
    norms = dirs.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


def test_requires_at_least_two_subjects():
    with pytest.raises(ValueError):
        subject_directions_from_means(_means({"a": [[1.0, 0.0]]}))


def test_load_labeled_jsonl_groups_by_label(tmp_path):
    p = tmp_path / "ds.jsonl"
    lines = [
        {"text": "deux plus deux", "subject": "math"},
        {"text": "intégrale de x", "subject": "math"},
        {"text": "la photosynthèse", "subject": "bio"},
    ]
    p.write_text("\n".join(json.dumps(o) for o in lines) + "\n", encoding="utf-8")
    groups = load_labeled(str(p))
    assert set(groups) == {"math", "bio"}
    assert len(groups["math"]) == 2 and len(groups["bio"]) == 1


def test_load_labeled_directory_one_file_per_subject(tmp_path):
    (tmp_path / "math.txt").write_text(
        json.dumps({"text": "deux plus deux"}) + "\n", encoding="utf-8")
    (tmp_path / "bio.txt").write_text(
        json.dumps({"text": "la photosynthèse"}) + "\n"
        + json.dumps({"text": "une cellule"}) + "\n", encoding="utf-8")
    groups = load_labeled(str(tmp_path))
    assert set(groups) == {"math", "bio"}
    assert len(groups["bio"]) == 2
