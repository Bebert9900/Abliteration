"""Tests du patching causal sur un modèle CONTRÔLABLE à effet connu.

`ControllableModel` est construit pour que la tête (0,0) porte TOUT le signal de refus et que
la tête (0,1) + le MLP n'y contribuent pas. Les valeurs attendues sont donc exactes :
- nécessité/suffisance recovery == 1.0 pour la tête causale ;
- recovery == 0.0 pour les composants non causaux.

C'est l'« effet connu reproduit sur petit modèle » exigé : on vérifie que le knockout d'un
composant top-rank dégrade le refus, et que le patching distingue causal de corrélationnel.
"""

from toymodel import (
    ControllableModel,
    controllable_refusal_dir,
    harmful_ids,
    harmless_ids,
)

from meridian.circuits.backend import Component, ComponentKind, TorchHookBackend
from meridian.circuits.patching import (
    CausalVerdict,
    RefusalMetric,
    necessity,
    sufficiency,
    validate_component,
)


def _setup():
    model = ControllableModel()
    be = TorchHookBackend(model)
    metric = RefusalMetric(refusal_dir=controllable_refusal_dir())
    return be, metric


CAUSAL = Component(ComponentKind.ATTN_HEAD, 0, 0)
NOISE = Component(ComponentKind.ATTN_HEAD, 0, 1)
MLP = Component(ComponentKind.MLP, 0)


def test_metric_separates_clean_and_corrupted():
    """Sanity : le modèle « refuse » (métrique haute) sur harmful, pas sur harmless."""
    be, metric = _setup()
    clean = metric(be.run_with_cache(harmful_ids()))
    corrupted = metric(be.run_with_cache(harmless_ids()))
    assert clean > 0.9 and corrupted < 0.1
    assert clean > corrupted


def test_necessity_of_causal_head_collapses_refusal():
    """Knockout de la tête causale sur le run clean → le refus s'effondre (recovery ≈ 1)."""
    be, metric = _setup()
    eff = necessity(be, CAUSAL, harmful_ids(), harmless_ids(), metric)
    assert eff.test == "necessity"
    assert eff.baseline > 0.9            # refusait
    assert eff.patched < 0.1             # ne refuse plus après knockout
    assert abs(eff.recovery - 1.0) < 1e-4


def test_noise_head_is_not_necessary():
    """Knockout d'une tête non causale → aucun effet sur le refus (recovery ≈ 0)."""
    be, metric = _setup()
    eff = necessity(be, NOISE, harmful_ids(), harmless_ids(), metric)
    assert abs(eff.recovery) < 1e-4
    assert abs(eff.delta) < 1e-4


def test_mlp_is_not_necessary_here():
    be, metric = _setup()
    eff = necessity(be, MLP, harmful_ids(), harmless_ids(), metric)
    assert abs(eff.recovery) < 1e-4


def test_sufficiency_of_causal_head_restores_refusal():
    """Restauration de la tête causale dans le run corrompu → le refus réapparaît (recovery ≈ 1)."""
    be, metric = _setup()
    eff = sufficiency(be, CAUSAL, harmful_ids(), harmless_ids(), metric)
    assert eff.test == "sufficiency"
    assert eff.baseline < 0.1            # ne refusait pas
    assert eff.patched > 0.9             # refuse après restauration
    assert abs(eff.recovery - 1.0) < 1e-4


def test_noise_head_is_not_sufficient():
    be, metric = _setup()
    eff = sufficiency(be, NOISE, harmful_ids(), harmless_ids(), metric)
    assert abs(eff.recovery) < 1e-4


def test_validate_component_marks_causal_head_validated():
    be, metric = _setup()
    verdict = validate_component(be, CAUSAL, harmful_ids(), harmless_ids(), metric, threshold=0.5)
    assert isinstance(verdict, CausalVerdict)
    assert verdict.is_necessary and verdict.is_sufficient
    assert verdict.causally_validated is True


def test_validate_component_rejects_noise_head():
    be, metric = _setup()
    verdict = validate_component(be, NOISE, harmful_ids(), harmless_ids(), metric, threshold=0.5)
    assert verdict.causally_validated is False


def test_logit_diff_metric_mode():
    """La métrique logit-diff (tokens) fonctionne aussi : refus = logit0 − logit1.

    lm_head : logit0 lit r, logit1 = 0. Donc logit-diff = métrique directionnelle ici.
    """
    be, _ = _setup()
    metric = RefusalMetric(refusal_token=0, answer_token=1)
    clean = metric(be.run_with_cache(harmful_ids()))
    corrupted = metric(be.run_with_cache(harmless_ids()))
    assert clean > 0.9 and corrupted < 0.1


def test_refusal_metric_requires_tokens_or_direction():
    import pytest
    with pytest.raises(ValueError):
        RefusalMetric()


def test_targeted_patch_only_changes_last_token():
    """Sur une séquence multi-tokens, le patch ciblé ne modifie QUE la dernière position.

    Vérifie que `targeted_patch_value` renvoie la contribution propre du run-cible partout,
    sauf au dernier token où elle vaut celle du run-source.
    """
    import torch as _t
    from toymodel import make_model

    from meridian.circuits.backend import TorchHookBackend
    from meridian.circuits.patching import targeted_patch_value

    be = TorchHookBackend(make_model(0))
    target_ids = _t.tensor([[1, 5, 3, 9]])
    source_ids = _t.tensor([[2, 7, 4, 8]])
    tcache = be.run_with_cache(target_ids)
    scache = be.run_with_cache(source_ids)
    comp = Component(ComponentKind.ATTN_HEAD, 1, 0)

    val = targeted_patch_value(tcache, scache, comp, None, None)
    target_contrib = tcache.component(comp)
    source_contrib = scache.component(comp)

    # positions 0..2 inchangées (== cible) ; position 3 (dernier token) == source
    assert _t.allclose(val[:, :-1, :], target_contrib[:, :-1, :], atol=1e-6)
    assert _t.allclose(val[:, -1, :], source_contrib[:, -1, :], atol=1e-6)
    # et la position non-dernière diffère bien de la source (sinon test vide)
    assert not _t.allclose(val[:, 0, :], source_contrib[:, 0, :], atol=1e-4)


def test_targeted_patch_handles_unequal_lengths():
    """Cible et source de longueurs différentes : pas d'erreur, seul le dernier token compte."""
    import torch as _t
    from toymodel import make_model

    from meridian.circuits.backend import TorchHookBackend
    from meridian.circuits.patching import targeted_patch_value

    be = TorchHookBackend(make_model(0))
    tcache = be.run_with_cache(_t.tensor([[1, 5, 3, 9]]))      # longueur 4
    scache = be.run_with_cache(_t.tensor([[2, 7]]))            # longueur 2
    comp = Component(ComponentKind.MLP, 0)
    val = targeted_patch_value(tcache, scache, comp, None, None)
    assert val.shape == tcache.component(comp).shape          # forme = celle de la cible
    assert _t.allclose(val[:, -1, :], scache.component(comp)[:, -1, :], atol=1e-6)
