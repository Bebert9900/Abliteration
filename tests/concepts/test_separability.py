"""Tests de la matrice de séparabilité entre concepts."""
import torch

from meridian.concepts import ConceptDirection, pairwise_separability


def _cd(name, vec):
    return ConceptDirection(name, torch.tensor(vec, dtype=torch.float32))


def test_matrix_symmetric_with_unit_diagonal():
    dirs = {
        "a": _cd("a", [[1.0, 0.0, 0.0]]),
        "b": _cd("b", [[0.0, 1.0, 0.0]]),
        "c": _cd("c", [[1.0, 0.0, 0.0]]),   # identique à a
    }
    sm = pairwise_separability(dirs, layer=0)
    m = sm.matrix
    assert m[0][0] == 1.0 and m[1][1] == 1.0          # diagonale = 1
    assert abs(m[0][1]) < 1e-6                          # a ⟂ b
    assert abs(m[0][2] - 1.0) < 1e-6                    # a ≡ c
    assert m[0][1] == m[1][0]                           # symétrique


def test_warnings_flag_collinear_concepts():
    dirs = {"a": _cd("a", [[1.0, 0.0]]), "b": _cd("b", [[0.95, 0.31]])}
    sm = pairwise_separability(dirs, layer=0)
    warns = sm.warnings(threshold=0.3)
    assert len(warns) == 1 and "a" in warns[0] and "b" in warns[0]


def test_layer_none_averages_over_layers():
    # 2 couches : a et b orthogonales en couche 0, identiques en couche 1 -> moyenne = 0.5.
    a = _cd("a", [[1.0, 0.0], [1.0, 0.0]])
    b = _cd("b", [[0.0, 1.0], [1.0, 0.0]])
    sm = pairwise_separability({"a": a, "b": b}, layer=None)
    assert abs(sm.matrix[0][1] - 0.5) < 1e-6
