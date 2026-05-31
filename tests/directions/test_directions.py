"""Tests des directions unitaires (4 classes) et de la séparabilité."""
import torch

from src.data import PromptClass
from src.directions import Directions, compute_directions, separability


def _means(harmful, harmless, negation, agentic):
    """Construit un dict de moyennes (L+1=1, H) à partir de vecteurs."""
    return {
        PromptClass.HARMFUL: torch.tensor([harmful], dtype=torch.float),
        PromptClass.HARMLESS: torch.tensor([harmless], dtype=torch.float),
        PromptClass.LEGITIMATE_NEGATION: torch.tensor([negation], dtype=torch.float),
        PromptClass.AGENTIC: torch.tensor([agentic], dtype=torch.float),
    }


def test_refusal_is_canonical_normalized_harmful_minus_harmless():
    means = _means([2, 0], [0, 0], [0, 3], [0, 0])
    d = compute_directions(means)
    # refus = normalize([2,0]-[0,0]) = [1,0]
    assert torch.allclose(d.refusal[0], torch.tensor([1.0, 0.0]), atol=1e-6)


def test_all_directions_are_unit_vectors():
    means = _means([1, 2], [0, 1], [3, 1], [2, 5])
    d = compute_directions(means)
    for vec in (d.refusal, d.negation, d.agentic, d.harmless):
        assert torch.allclose(vec[0].norm(), torch.tensor(1.0), atol=1e-5)


def test_separability_cosine_zero_for_orthogonal_directions():
    d = Directions(
        refusal=torch.tensor([[1.0, 0.0]]),
        harmless=torch.tensor([[0.0, 1.0]]),
        negation=torch.tensor([[0.0, 1.0]]),   # orthogonal au refus
        agentic=torch.tensor([[0.0, 1.0]]),
    )
    report = separability(d, threshold=0.3)
    assert abs(report.cosine_refusal_negation[0]) < 1e-6
    assert abs(report.cosine_refusal_agentic[0]) < 1e-6
    assert report.warnings() == []


def test_separability_warns_when_refusal_overlaps_preserved_direction():
    d = Directions(
        refusal=torch.tensor([[1.0, 0.0]]),
        harmless=torch.tensor([[0.0, 1.0]]),
        negation=torch.tensor([[1.0, 0.0]]),   # colinéaire au refus -> cos=1
        agentic=torch.tensor([[0.0, 1.0]]),
    )
    report = separability(d, threshold=0.3)
    assert report.cosine_refusal_negation[0] == 1.0
    warns = report.warnings()
    assert len(warns) == 1
    assert "negation" in warns[0].lower() and "0" in warns[0]  # mentionne la couche 0
