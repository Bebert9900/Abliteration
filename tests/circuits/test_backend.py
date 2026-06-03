"""Tests du backend d'introspection par composant (TorchHookBackend).

Invariants vérifiés sur un modèle jouet Llama-like RÉEL (structure que ArchAdapter comprend) :
1. la somme des contributions par tête == sortie de o_proj (décomposition exacte) ;
2. la contribution MLP capturée == sortie de down_proj ;
3. le patching d'un composant substitue bien sa contribution (et seulement la sienne) ;
4. ModelInfo lit la bonne géométrie (couches/têtes/dim).

Le modèle jouet vit dans `toymodel.py` (module frère, partagé avec les autres tests circuits).
"""
import torch

from abliteration.circuits.backend import (
    Component,
    ComponentKind,
    ModelInfo,
    Patch,
    TorchHookBackend,
)
from toymodel import ToyModel, ids as _ids, make_model as _model


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
def test_model_info_reads_geometry():
    be = TorchHookBackend(_model())
    info = be.info
    assert isinstance(info, ModelInfo)
    assert (info.n_layers, info.n_heads, info.head_dim, info.hidden_size) == (3, 2, 4, 8)


def test_head_contributions_sum_to_o_proj_output():
    """Σ_h contrib_tête_h doit égaler exactement la sortie de o_proj de la couche."""
    model = _model()
    be = TorchHookBackend(model)
    ids = _ids()

    # capture la sortie réelle de o_proj de la couche 1 via un hook indépendant
    captured = {}
    h = model.model.layers[1].self_attn.o_proj.register_forward_hook(
        lambda m, a, o: captured.__setitem__("o", o.detach())
    )
    try:
        cache = be.run_with_cache(ids)
    finally:
        h.remove()

    heads = cache.attn_head_out[1]              # (b, s, n_heads, hidden)
    summed = heads.sum(dim=2)                   # (b, s, hidden)
    assert torch.allclose(summed, captured["o"], atol=1e-5)


def test_mlp_capture_matches_down_proj_output():
    model = _model()
    be = TorchHookBackend(model)
    captured = {}
    h = model.model.layers[2].mlp.down_proj.register_forward_hook(
        lambda m, a, o: captured.__setitem__("d", o.detach())
    )
    try:
        cache = be.run_with_cache(_ids())
    finally:
        h.remove()
    assert torch.allclose(cache.mlp_out[2], captured["d"], atol=1e-5)


def test_component_accessor():
    be = TorchHookBackend(_model())
    cache = be.run_with_cache(_ids())
    head = cache.component(Component(ComponentKind.ATTN_HEAD, 0, 1))
    mlp = cache.component(Component(ComponentKind.MLP, 0))
    assert head.shape == (1, 4, 8)
    assert mlp.shape == (1, 4, 8)


def test_all_components_enumerates_heads_and_mlps():
    be = TorchHookBackend(_model())
    comps = be.all_components()
    # 3 couches * (2 têtes + 1 mlp) = 9
    assert len(comps) == 9
    assert sum(c.kind is ComponentKind.MLP for c in comps) == 3
    assert sum(c.kind is ComponentKind.ATTN_HEAD for c in comps) == 6


def test_patching_a_head_changes_only_that_head_and_logits():
    """Patcher la contribution d'une tête à zéro modifie les logits et reflète le patch en cache."""
    model = _model()
    be = TorchHookBackend(model)
    ids = _ids()

    clean = be.run_with_cache(ids)
    target = Component(ComponentKind.ATTN_HEAD, 1, 0)
    zero = torch.zeros_like(clean.component(target))

    patched = be.run_with_patches(ids, None, [Patch(target, zero)])

    # la contribution patchée est bien nulle dans le cache du run patché
    assert torch.allclose(patched.component(target), zero, atol=1e-6)
    # une autre tête de la même couche est inchangée vs clean
    other = Component(ComponentKind.ATTN_HEAD, 1, 1)
    assert torch.allclose(patched.component(other), clean.component(other), atol=1e-5)
    # les logits changent (le composant contribuait réellement)
    assert not torch.allclose(patched.logits, clean.logits, atol=1e-4)


def test_patching_mlp_replaces_output():
    model = _model()
    be = TorchHookBackend(model)
    ids = _ids()
    clean = be.run_with_cache(ids)
    target = Component(ComponentKind.MLP, 0)
    repl = clean.component(target) + 1.0
    patched = be.run_with_patches(ids, None, [Patch(target, repl)])
    assert torch.allclose(patched.mlp_out[0], repl, atol=1e-5)
    assert not torch.allclose(patched.logits, clean.logits, atol=1e-4)


def test_hooks_are_removed_after_run():
    """Aucun hook résiduel : un forward nu après coup ne doit rien capturer en plus."""
    model = _model()
    be = TorchHookBackend(model)
    be.run_with_cache(_ids())
    # plus aucun forward hook sur o_proj/down_proj
    for layer in model.model.layers:
        assert len(layer.self_attn.o_proj._forward_hooks) == 0
        assert len(layer.self_attn.o_proj._forward_pre_hooks) == 0
        assert len(layer.mlp.down_proj._forward_hooks) == 0
