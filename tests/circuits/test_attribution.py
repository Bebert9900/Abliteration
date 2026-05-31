"""Tests attribution patching : cohérence avec le patching exact + garde-fou caveat.

Sur `ControllableModel` (effet linéaire connu), l'approximation gradient doit être EXACTE :
le modèle est linéaire le long de la direction de refus, donc Δm approximé == Δm exact. On
vérifie aussi que l'attribution classe la tête causale en tête et s'accorde avec le patching
exact (la contre-vérification exigée par le skill).
"""
import torch

from src.circuits.attribution import (
    ATTRIBUTION_CAVEAT,
    AttributionResult,
    agreement_with_exact,
    attribution_patching,
)
from src.circuits.backend import Component, ComponentKind, TorchHookBackend
from src.circuits.patching import RefusalMetric, necessity
from toymodel import (
    ControllableModel,
    controllable_refusal_dir,
    harmful_ids,
    harmless_ids,
)

CAUSAL = Component(ComponentKind.ATTN_HEAD, 0, 0)
NOISE = Component(ComponentKind.ATTN_HEAD, 0, 1)


def _backend():
    return TorchHookBackend(ControllableModel())


def test_attribution_ranks_causal_head_first():
    be = _backend()
    res = attribution_patching(
        be, harmful_ids(), harmless_ids(), refusal_dir=controllable_refusal_dir()
    )
    top_comp, top_score = res.ranked()[0]
    assert top_comp == CAUSAL
    assert abs(top_score) > 0.5


def test_attribution_noise_head_scores_zero():
    be = _backend()
    res = attribution_patching(
        be, harmful_ids(), harmless_ids(), refusal_dir=controllable_refusal_dir()
    )
    assert abs(res.scores[NOISE]) < 1e-5


def test_attribution_matches_exact_patching_on_linear_model():
    """Modèle linéaire le long de r → Δm approx == Δm exact (necessity delta) pour la tête causale."""
    be = _backend()
    metric = RefusalMetric(refusal_dir=controllable_refusal_dir())
    exact = necessity(be, CAUSAL, harmful_ids(), harmless_ids(), metric)
    res = attribution_patching(
        be, harmful_ids(), harmless_ids(), refusal_dir=controllable_refusal_dir()
    )
    # necessity injecte corrupted dans clean : Δm exact = patched - baseline
    assert abs(res.scores[CAUSAL] - exact.delta) < 1e-4


def test_agreement_with_exact_is_one_when_tops_match():
    be = _backend()
    res = attribution_patching(
        be, harmful_ids(), harmless_ids(), refusal_dir=controllable_refusal_dir()
    )
    exact_scores = {CAUSAL: -1.0, NOISE: 0.0, Component(ComponentKind.MLP, 0): 0.0}
    assert agreement_with_exact(res, exact_scores, k=1) == 1.0


def test_result_carries_caveat():
    be = _backend()
    res = attribution_patching(
        be, harmful_ids(), harmless_ids(), refusal_dir=controllable_refusal_dir()
    )
    assert isinstance(res, AttributionResult)
    assert "APPROXIMATION" in res.caveat
    assert res.caveat == ATTRIBUTION_CAVEAT


def test_requires_a_metric_spec():
    import pytest
    be = _backend()
    with pytest.raises(ValueError):
        attribution_patching(be, harmful_ids(), harmless_ids())


def test_token_metric_mode_runs():
    be = _backend()
    res = attribution_patching(
        be, harmful_ids(), harmless_ids(), refusal_token=0, answer_token=1
    )
    # la tête causale reste en tête avec la métrique logit-diff
    assert res.ranked()[0][0] == CAUSAL
