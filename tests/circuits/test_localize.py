"""Tests de localisation : helpers purs (Jaccard/bootstrap) + intégration sur modèle contrôlable.

Le modèle contrôlable a un core connu = {tête (0,0)}. On vérifie que `localize` :
- ne met dans le core QUE des composants validés causalement (pas la DLA seule) ;
- identifie la bonne tête ;
- atteint une stabilité bootstrap parfaite (le signal est déterministe) ;
- produit faithfulness=1, CPR≈1, CMD≈0 (le core explique tout l'effet).
"""
import torch

from src.circuits.backend import Component, ComponentKind, TorchHookBackend
from src.circuits.localize import (
    ComponentEvidence,
    bootstrap_stability,
    jaccard,
    localize,
)
from src.circuits.patching import RefusalMetric
from toymodel import (
    ControllableModel,
    controllable_refusal_dir,
    harmful_ids,
    harmless_ids,
)

CAUSAL = Component(ComponentKind.ATTN_HEAD, 0, 0)
NOISE = Component(ComponentKind.ATTN_HEAD, 0, 1)
MLP = Component(ComponentKind.MLP, 0)


# ---- helpers purs --------------------------------------------------------- #
def test_jaccard_basics():
    assert jaccard(set(), set()) == 1.0
    assert jaccard({1, 2}, {1, 2}) == 1.0
    assert jaccard({1, 2}, {2, 3}) == 1 / 3
    assert jaccard({1}, {2}) == 0.0


def test_bootstrap_perfect_when_core_is_stable():
    # 3 paires donnant toutes le même core {CAUSAL} → Jaccard bootstrap == 1.0
    row = {CAUSAL: (1.0, 1.0, 1.0), NOISE: (0.0, 0.0, 0.0)}
    per_pair = [dict(row) for _ in range(3)]
    assert bootstrap_stability(per_pair, threshold=0.5, n_boot=50, seed=0) == 1.0


def test_bootstrap_drops_when_core_unstable():
    # une paire soutient CAUSAL, une autre NOISE → cores divergents → Jaccard < 1
    per_pair = [
        {CAUSAL: (1.0, 1.0, 1.0), NOISE: (0.0, 0.0, 0.0)},
        {CAUSAL: (0.0, 0.0, 0.0), NOISE: (0.0, 1.0, 1.0)},
    ]
    j = bootstrap_stability(per_pair, threshold=0.5, n_boot=200, seed=1)
    assert j < 1.0


# ---- intégration ---------------------------------------------------------- #
def _pairs():
    h, n = harmful_ids(), harmless_ids()
    return [(h, n, None, None), (h, n, None, None), (h, n, None, None)]


def test_localize_core_is_only_causal_head():
    be = TorchHookBackend(ControllableModel())
    metric = RefusalMetric(refusal_dir=controllable_refusal_dir())
    loc = localize(be, _pairs(), metric, controllable_refusal_dir(),
                   threshold=0.5, n_boot=50)
    assert CAUSAL in loc.core
    assert NOISE not in loc.core
    assert MLP not in loc.core


def test_localize_does_not_promote_on_dla_alone():
    """Garde-fou règle d'or : un composant à forte DLA mais sans causalité n'entre pas au core.

    On falsifie l'évidence : NOISE reçoit une grosse DLA mais nec/suf nuls.
    """
    from src.circuits.localize import _core_from_evidence
    ev = {
        CAUSAL: ComponentEvidence(CAUSAL, dla=0.01, necessity=1.0, sufficiency=1.0),
        NOISE: ComponentEvidence(NOISE, dla=999.0, necessity=0.0, sufficiency=0.0),
    }
    core = _core_from_evidence(ev, threshold=0.5)
    assert core == {CAUSAL}          # la DLA énorme de NOISE ne le sauve pas


def test_localize_bootstrap_and_circuit_metrics():
    be = TorchHookBackend(ControllableModel())
    metric = RefusalMetric(refusal_dir=controllable_refusal_dir())
    loc = localize(be, _pairs(), metric, controllable_refusal_dir(),
                   threshold=0.5, n_boot=50)
    assert loc.bootstrap_jaccard == 1.0
    assert loc.faithfulness == 1.0
    assert loc.cpr is not None and abs(loc.cpr - 1.0) < 1e-3
    assert loc.cmd is not None and loc.cmd < 1e-3


def test_localize_attention_mlp_split():
    be = TorchHookBackend(ControllableModel())
    metric = RefusalMetric(refusal_dir=controllable_refusal_dir())
    loc = localize(be, _pairs(), metric, controllable_refusal_dir(),
                   threshold=0.5, n_boot=10)
    attn, mlp = loc.attention_mlp_split()
    assert (attn, mlp) == (1, 0)
