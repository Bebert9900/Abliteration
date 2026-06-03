"""Tests DLA : exactitude de la projection, ratio attn/MLP, classement, et garde-fou caveat.

On réutilise le modèle jouet du backend (decoder-only Llama-like réel). On INJECTE un signal
contrôlé : on choisit une direction r̂ et on vérifie que le score DLA d'un composant égale
exactement la projection de sa contribution residual-espace sur r̂ au dernier token — c'est la
définition, calculée indépendamment du code testé.
"""
import torch

from abliteration.circuits.backend import Component, ComponentKind, TorchHookBackend
from abliteration.circuits.dla import (
    CORRELATIONAL_CAVEAT,
    DLAResult,
    direct_logit_attribution,
    readout_direction,
)
from toymodel import ToyModel, ids as _ids, make_model


def _backend(seed=0):
    return TorchHookBackend(make_model(seed))


def test_dla_score_equals_manual_projection():
    be = _backend()
    ids = _ids()
    torch.manual_seed(1)
    r = torch.randn(be.info.hidden_size)
    r = r / r.norm()

    res = direct_logit_attribution(be, r, ids)

    # recalcul indépendant pour un composant donné
    cache = be.run_with_cache(ids)
    c = Component(ComponentKind.ATTN_HEAD, 1, 0)
    last = cache.component(c)[0, -1, :].to(torch.float32)
    expected = float(last @ r.to(torch.float32))
    assert abs(res.scores[c] - expected) < 1e-5


def test_dla_sum_approximates_final_residual_projection():
    """Σ scores composants ≈ projection du résidu final sur r̂ (aux termes embed/biais près).

    Le modèle jouet n'a pas de biais sur o_proj/down_proj ; le résidu final =
    embed + Σ(attn) + Σ(mlp). La projection de (Σ composants) doit donc égaler la projection de
    (résidu final − embed). On vérifie cette cohérence comptable.
    """
    be = _backend()
    ids = _ids()
    r = torch.zeros(be.info.hidden_size)
    r[0] = 1.0  # lis la coordonnée 0

    res = direct_logit_attribution(be, r, ids)
    total = sum(res.scores.values())

    cache = be.run_with_cache(ids)
    embed = be.model.model.embed_tokens(ids)[0, -1, :].detach().to(torch.float32)
    final = cache.final_resid[0, -1, :].to(torch.float32)
    expected = float((final - embed) @ r.to(torch.float32))
    assert abs(total - expected) < 1e-4


def test_attention_mlp_ratio_sums_to_one():
    be = _backend()
    r = readout_direction_like(be)
    res = direct_logit_attribution(be, r, _ids())
    attn, mlp = res.attention_mlp_ratio()
    assert abs((attn + mlp) - 1.0) < 1e-6
    assert 0.0 <= attn <= 1.0 and 0.0 <= mlp <= 1.0


def test_ranked_is_sorted_by_absolute_value():
    be = _backend()
    res = direct_logit_attribution(be, readout_direction_like(be), _ids())
    ranked = res.ranked(by_abs=True)
    mags = [abs(v) for _, v in ranked]
    assert mags == sorted(mags, reverse=True)


def test_include_mlp_false_excludes_mlps():
    be = _backend()
    res = direct_logit_attribution(be, readout_direction_like(be), _ids(), include_mlp=False)
    assert all(c.kind is ComponentKind.ATTN_HEAD for c in res.scores)
    attn, mlp = res.attention_mlp_ratio()
    assert mlp == 0.0


def test_result_carries_correlational_caveat():
    """Garde-fou : le résultat DLA porte explicitement l'avertissement non-causal."""
    be = _backend()
    res = direct_logit_attribution(be, readout_direction_like(be), _ids())
    assert isinstance(res, DLAResult)
    assert "CORRÉLATIONNEL" in res.caveat
    assert res.caveat == CORRELATIONAL_CAVEAT


def test_readout_direction_reads_from_directions_object_without_recompute():
    """readout_direction extrait r̂_layer normalisé d'un Directions, sans le recalculer."""
    class FakeDirections:
        refusal = torch.tensor([[3.0, 4.0], [0.0, 2.0]])  # (L+1=2, hidden=2)

    r = readout_direction(FakeDirections(), layer=0)
    assert torch.allclose(r, torch.tensor([0.6, 0.8]), atol=1e-6)
    assert abs(float(r.norm()) - 1.0) < 1e-6


# --------------------------------------------------------------------------- #
# Helper local
# --------------------------------------------------------------------------- #
def readout_direction_like(be):
    torch.manual_seed(2)
    r = torch.randn(be.info.hidden_size)
    return r / r.norm()
