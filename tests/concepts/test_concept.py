"""Tests du modèle Concept : direction par contraste, split holdout déterministe, équivalence."""
import pytest
import torch

from abliteration.concepts import Concept, direction_from_means
from abliteration.concepts.concept import ConceptDirection, concept_direction


def test_direction_from_means_is_unit_and_matches_normalized_diff():
    mu_pos = torch.tensor([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]])   # (L+1=2, H=3)
    mu_neg = torch.zeros(2, 3)
    d = direction_from_means(mu_pos, mu_neg)
    assert d.shape == (2, 3)
    assert torch.allclose(d.norm(dim=-1), torch.ones(2), atol=1e-6)
    assert torch.allclose(d[0], torch.tensor([1.0, 0.0, 0.0]), atol=1e-6)


def test_direction_equivalence_with_legacy_refusal():
    # ÉQUIVALENCE : la direction du concept refusal == Directions.refusal de l'ancien chemin,
    # puisque les deux calculent normalize(μ_harmful − μ_harmless). Preuve sans modèle.
    from abliteration.data import PromptClass
    from abliteration.directions import compute_directions

    torch.manual_seed(0)
    harmful = torch.randn(2, 5)
    harmless = torch.randn(2, 5)
    means = {
        PromptClass.HARMFUL: harmful,
        PromptClass.HARMLESS: harmless,
        PromptClass.LEGITIMATE_NEGATION: torch.randn(2, 5),
        PromptClass.AGENTIC: torch.randn(2, 5),
    }
    legacy = compute_directions(means).refusal
    concept = direction_from_means(harmful, harmless)
    assert torch.allclose(legacy, concept, atol=1e-6)


def test_concept_split_is_deterministic_and_disjoint():
    c = Concept("x", positive=[f"p{i}" for i in range(10)], negative=[f"n{i}" for i in range(10)])
    tr1, ho1 = c.split(0.2, seed=0)
    tr2, ho2 = c.split(0.2, seed=0)
    assert tr1.positive == tr2.positive and ho1.positive == ho2.positive   # déterministe
    assert set(tr1.positive).isdisjoint(ho1.positive)                      # disjoint
    assert len(ho1.positive) == 2 and len(tr1.positive) == 8


def test_concept_direction_rejects_empty_side():
    with pytest.raises(ValueError):
        concept_direction(Concept("x", positive=[], negative=["a"]), model=None, formatter=None)


def test_concept_direction_norms_per_layer():
    cd = ConceptDirection("x", torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    assert cd.norms_per_layer() == pytest.approx([1.0, 1.0])
