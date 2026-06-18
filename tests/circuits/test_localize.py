"""Tests de localisation : helpers purs (Jaccard/bootstrap) + intégration sur modèle contrôlable.

Le modèle contrôlable a un core connu = {tête (0,0)}. On vérifie que `localize` :
- ne met dans le core QUE des composants validés causalement (pas la DLA seule) ;
- identifie la bonne tête ;
- atteint une stabilité bootstrap parfaite (le signal est déterministe) ;
- produit faithfulness=1, CPR≈1, CMD≈0 (le core explique tout l'effet).
"""

from toymodel import (
    ControllableModel,
    controllable_refusal_dir,
    harmful_ids,
    harmless_ids,
)

from meridian.circuits.backend import Component, ComponentKind, TorchHookBackend
from meridian.circuits.localize import (
    ComponentEvidence,
    bootstrap_stability,
    jaccard,
    localize,
)
from meridian.circuits.patching import RefusalMetric

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


# ---- RC2 : sélection par consensus (stability selection) ------------------ #
def test_selection_frequencies_separate_stable_from_borderline():
    """CAUSAL passe le seuil dans toutes les paires → fréquence ~1 ; NOISE seulement la moitié
    → fréquence intermédiaire (instable)."""
    from meridian.circuits.localize import selection_frequencies
    per_pair = [
        {CAUSAL: (1.0, 1.0, 1.0), NOISE: (0.0, 1.0, 1.0)},   # NOISE passe
        {CAUSAL: (1.0, 1.0, 1.0), NOISE: (0.0, 0.0, 0.0)},   # NOISE échoue
        {CAUSAL: (1.0, 1.0, 1.0), NOISE: (0.0, 1.0, 1.0)},
        {CAUSAL: (1.0, 1.0, 1.0), NOISE: (0.0, 0.0, 0.0)},
    ]
    freq = selection_frequencies(per_pair, threshold=0.5, n_boot=500, seed=0)
    assert freq[CAUSAL] > 0.95
    assert 0.1 < freq[NOISE] < 0.9


def test_greedy_faithful_core_grows_to_minimal_faithful_set():
    """RC2 corrigé : le core = plus petit préfixe (par force causale) dont la faithfulness
    atteint la cible. Un seuil dur AND knife-edge scinde un vrai circuit à 2 têtes."""
    from meridian.circuits.localize import greedy_faithful_core
    ranked = [CAUSAL, NOISE, MLP]
    fmap = {(CAUSAL,): 0.44, (CAUSAL, NOISE): 1.0}   # k=1 insuffisant, k=2 explique tout
    core, k = greedy_faithful_core(ranked, lambda c: fmap.get(tuple(c), 1.0), target=0.9)
    assert core == [CAUSAL, NOISE]
    assert k == 2


def test_extend_through_ties_absorbs_near_equal_then_stops_at_gap():
    """Le core faithful minimal coupe parfois au milieu d'une quasi-égalité causale (L15H9≈L15MLP),
    ce qui rend le membership instable. On étend à travers les quasi-égalités jusqu'au prochain
    gap de nécessité → core stable."""
    from meridian.circuits.localize import extend_through_ties
    ranked = ["a", "b", "c", "d"]
    necs = {"a": 0.56, "b": 0.51, "c": 0.46, "d": 0.23}
    # c (0.46) ≥ 0.85·0.51 = 0.43 → absorbé ; d (0.23) < 0.85·0.46 = 0.39 → stop
    assert extend_through_ties(ranked, necs, k_start=2, tie_ratio=0.85) == 3


def test_extend_through_ties_no_extension_past_nonpositive_or_gap():
    from meridian.circuits.localize import extend_through_ties
    ranked = ["a", "b", "c"]
    necs = {"a": 1.0, "b": 0.0, "c": -0.2}   # b n'est pas une quasi-égalité de a
    assert extend_through_ties(ranked, necs, k_start=1, tie_ratio=0.85) == 1


def test_greedy_faithful_core_stops_at_k1_when_enough():
    from meridian.circuits.localize import greedy_faithful_core
    core, k = greedy_faithful_core([CAUSAL, NOISE], lambda c: 1.0, target=0.9)
    assert core == [CAUSAL]
    assert k == 1


def test_localize_faithful_core_reaches_target_and_is_stable():
    from toymodel import ControllableModel, controllable_refusal_dir, harmful_ids, harmless_ids

    from meridian.circuits.backend import TorchHookBackend
    be = TorchHookBackend(ControllableModel())
    metric = RefusalMetric(refusal_dir=controllable_refusal_dir())
    h, n = harmful_ids(), harmless_ids()
    pairs = [(h, n, None, None)] * 3
    loc = localize(be, pairs, metric, controllable_refusal_dir(),
                   threshold=0.5, n_boot=20, target_faithfulness=0.9)
    assert set(loc.core) == {CAUSAL}
    assert loc.faithfulness >= 0.9
    assert loc.bootstrap_jaccard >= 0.9


def test_split_pairs_train_test_disjoint_and_complete():
    """Anti-tautologie : train ∩ test = ∅ (prouvé numériquement) et couvre tous les indices."""
    from meridian.circuits.localize import split_pairs
    tr, te = split_pairs(20, holdout_frac=0.5, seed=0)
    assert set(tr) & set(te) == set()
    assert sorted(tr + te) == list(range(20))
    assert len(te) == 10 and len(tr) == 10


def test_split_pairs_off_when_disabled():
    from meridian.circuits.localize import split_pairs
    assert split_pairs(20, holdout_frac=None) == (list(range(20)), list(range(20)))


def test_localize_reports_heldout_faithfulness_with_negative_control():
    """La faithfulness REPORTÉE est mesurée sur le held-out. Contrôle négatif permanent : le vrai
    circuit y est élevé, un composant non-refus y est bas."""
    from toymodel import ControllableModel, controllable_refusal_dir, harmful_ids, harmless_ids

    from meridian.circuits.backend import TorchHookBackend
    from meridian.circuits.localize import _circuit_metrics
    be = TorchHookBackend(ControllableModel())
    metric = RefusalMetric(refusal_dir=controllable_refusal_dir())
    h, n = harmful_ids(), harmless_ids()
    pairs = [(h, n, None, None)] * 4
    loc = localize(be, pairs, metric, controllable_refusal_dir(),
                   threshold=0.5, n_boot=20, target_faithfulness=0.9,
                   holdout_frac=0.5, min_holdout=5)
    assert loc.n_train >= 1 and loc.n_test >= 1 and loc.n_train + loc.n_test == 4
    assert set(loc.core) == {CAUSAL}
    assert loc.faithfulness >= 0.9                      # vrai circuit, held-out
    # contrôle négatif : un composant non lié au refus a une faithfulness held-out basse
    noise_faith = _circuit_metrics(be, [NOISE], be.all_components(), pairs[:2], metric)[0]
    assert noise_faith < 0.5
    # test-set (2) < min_holdout (5) → WARNING explicite
    assert loc.holdout_warning is not None


def test_core_by_consensus_keeps_stable_lists_marginal():
    from meridian.circuits.localize import core_by_consensus
    freq = {CAUSAL: 1.0, NOISE: 0.45, MLP: 0.0}
    core, marginal = core_by_consensus(freq, consensus_frac=0.8)
    assert core == {CAUSAL}
    assert marginal == {NOISE}        # sélectionné parfois mais pas de façon stable
    assert MLP not in core and MLP not in marginal


def test_localize_consensus_excludes_borderline_and_is_more_stable():
    """Sur des évidences où une tête borderline ferait chuter le Jaccard, le mode consensus
    rend un core stable (Jaccard ~1) et relègue la borderline en `marginal`."""

    from toymodel import ControllableModel, controllable_refusal_dir, harmful_ids, harmless_ids

    from meridian.circuits.backend import TorchHookBackend

    be = TorchHookBackend(ControllableModel())
    metric = RefusalMetric(refusal_dir=controllable_refusal_dir())
    h, n = harmful_ids(), harmless_ids()
    pairs = [(h, n, None, None)] * 3
    loc = localize(be, pairs, metric, controllable_refusal_dir(),
                   threshold=0.5, n_boot=50, consensus_frac=0.8)
    assert CAUSAL in loc.core
    assert NOISE not in loc.core
    assert loc.bootstrap_jaccard >= 0.9
    assert hasattr(loc, "marginal")


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
    from meridian.circuits.localize import _core_from_evidence
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
