"""Tests attribution patching : cohérence avec le patching exact + garde-fou caveat.

Sur `ControllableModel` (effet linéaire connu), l'approximation gradient doit être EXACTE :
le modèle est linéaire le long de la direction de refus, donc Δm approximé == Δm exact. On
vérifie aussi que l'attribution classe la tête causale en tête et s'accorde avec le patching
exact (la contre-vérification exigée par le skill).
"""
import torch

from abliteration.circuits.attribution import (
    ATTRIBUTION_CAVEAT,
    AttributionResult,
    aggregate_attribution,
    agreement_with_exact,
    attribution_patching,
)
from abliteration.circuits.backend import Component, ComponentKind, TorchHookBackend
from abliteration.circuits.patching import RefusalMetric, necessity
from toymodel import (
    ControllableModel,
    controllable_refusal_dir,
    harmful_ids,
    harmless_ids,
    make_model,
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


# --- RC1 : sélection des candidats agrégée sur TOUTES les paires ----------- #
def _divergent_pairs():
    A = torch.tensor([[1, 5, 3, 9]]); B = torch.tensor([[2, 2, 2, 2]])
    C = torch.tensor([[7, 1, 4, 0]]); D = torch.tensor([[9, 9, 1, 3]])
    return [(A, B, None, None), (C, D, None, None), (B, A, None, None)]


def test_aggregate_attribution_is_mean_over_pairs_and_order_invariant():
    """Corrige RC1 : l'univers de candidats ne doit PAS dépendre de pairs[0].

    On agrège l'attribution sur toutes les paires (moyenne des scores). Le résultat doit (a)
    égaler la moyenne des attributions par paire, (b) être invariant à l'ordre des paires —
    contrairement au choix actuel `top_k(attribution(pairs[0]))`.
    """
    be = TorchHookBackend(make_model(seed=0, hidden=8, n_heads=2, vocab=16, n_layers=3))
    r = torch.randn(8, generator=torch.Generator().manual_seed(1))
    pairs = _divergent_pairs()

    per = [attribution_patching(be, c, k, refusal_dir=r) for (c, k, _, _) in pairs]
    # le bug RC1 n'est réel que si les paires divergent : on l'exige dans la fixture
    tops = [{comp for comp, _ in p.top(4)} for p in per]
    assert not all(t == tops[0] for t in tops), "fixture invalide : les paires ne divergent pas"

    agg = aggregate_attribution(be, pairs, refusal_dir=r)
    assert isinstance(agg, AttributionResult)
    for comp in agg.scores:
        mean = sum(p.scores[comp] for p in per) / len(per)
        assert abs(agg.scores[comp] - mean) < 1e-6

    agg_rev = aggregate_attribution(be, list(reversed(pairs)), refusal_dir=r)
    for comp in agg.scores:
        assert abs(agg.scores[comp] - agg_rev.scores[comp]) < 1e-6


def test_aggregate_attribution_accepts_per_pair_directions():
    """refusal_dir peut être un tenseur partagé OU une liste (une direction par paire)."""
    be = TorchHookBackend(make_model(seed=0, hidden=8, n_heads=2, vocab=16, n_layers=3))
    pairs = _divergent_pairs()
    dirs = [torch.randn(8, generator=torch.Generator().manual_seed(i)) for i in range(len(pairs))]
    agg = aggregate_attribution(be, pairs, refusal_dir=dirs)
    per = [attribution_patching(be, c, k, refusal_dir=dirs[i])
           for i, (c, k, _, _) in enumerate(pairs)]
    for comp in agg.scores:
        mean = sum(p.scores[comp] for p in per) / len(per)
        assert abs(agg.scores[comp] - mean) < 1e-6
