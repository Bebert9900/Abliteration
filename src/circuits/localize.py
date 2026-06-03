"""Localisation du circuit de refus : agrège DLA + patching causal sur plusieurs paires.

Sortie = sous-ensemble de composants CAUSALEMENT responsables (le « circuit core »), classé,
caractérisé (motif gate→amplificateur si présent), avec mesures de robustesse. C'est l'entrée
de la (future) Phase 2 — mais ici on ne fait QUE localiser/valider.

Règle d'or appliquée concrètement : un composant n'entre dans le `core` QUE s'il est validé
causalement (nécessité ET suffisance ≥ seuil), jamais sur la DLA seule. La DLA sert à classer
et à révéler le motif (un `gate` a une DLA quasi nulle mais une nécessité causale forte).

Métriques de localisation (définitions conceptuelles : circuit_analysis.md §Métriques,
BlackboxNLP 2025 ; les FORMULES ci-dessous sont notre opérationnalisation explicite, pas des
chiffres importés) :
- **faithfulness** : fraction de paires où l'intervention sur le circuit core produit le
  changement attendu (le knockout du core sur le run clean fait basculer la métrique vers le
  niveau corrompu, au-delà du milieu). **ANTI-TAUTOLOGIE** : avec `holdout_frac`, le circuit est
  SÉLECTIONNÉ sur le train-set (attribution + greedy + nécessité) et la faithfulness REPORTÉE est
  mesurée sur le test-set held-out (paires jamais vues à la sélection). Le chiffre autoritaire et
  la décision de porte = held-out ; l'in-sample n'est gardé que pour comparaison. Comme greedy
  croît JUSQU'À la cible, atteindre la cible n'est pas une preuve (c'est la condition d'arrêt) :
  la preuve = held-out élevé + contrôle négatif (triplet aléatoire ≈ 0).
- **CPR** (circuit performance ratio) : effet causal capté par le core / effet causal de TOUS
  les composants (∈ ~[0,1], plus haut = le core capture l'essentiel ; >1 possible si le core
  surcapte).
- **CMD** (circuit-model distance) : distance normalisée entre le comportement « core restauré »
  et le modèle complet (0 = identique).
- **stabilité bootstrap** : Jaccard des cores re-estimés sur des ré-échantillons de paires
  (viser >0.9).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from .backend import CircuitBackend, Component, ComponentKind, Patch
from .dla import direct_logit_attribution
from .patching import RefusalMetric, validate_component


# --------------------------------------------------------------------------- #
# Données agrégées
# --------------------------------------------------------------------------- #
@dataclass
class ComponentEvidence:
    """Preuves accumulées pour un composant sur l'ensemble des paires."""
    component: Component
    dla: float                    # contribution DLA moyenne (CORRÉLATIONNEL)
    necessity: float              # recovery moyen du test de nécessité (CAUSAL)
    sufficiency: float            # recovery moyen du test de suffisance (CAUSAL)

    def causally_validated(self, threshold: float) -> bool:
        return self.necessity >= threshold and self.sufficiency >= threshold


@dataclass
class Localization:
    evidence: dict[Component, ComponentEvidence]
    core: list[Component]
    threshold: float
    gates: list[Component] = field(default_factory=list)
    amplifiers: list[Component] = field(default_factory=list)
    marginal: list[Component] = field(default_factory=list)
    selection_frequency: dict[Component, float] | None = None
    bootstrap_jaccard: float | None = None
    faithfulness: float | None = None          # AUTORITAIRE = held-out (anti-tautologie)
    faithfulness_insample: float | None = None  # même paires que la sélection (transparence)
    cpr: float | None = None
    cmd: float | None = None
    n_train: int = 0                            # paires de SÉLECTION (attribution + greedy)
    n_test: int = 0                             # paires de MESURE (held-out, jamais sélectionnées)
    held_out: bool = False                      # True si faithfulness mesurée hors échantillon
    holdout_warning: str | None = None          # !=None si le test-set est trop petit

    def ranked_core(self) -> list[Component]:
        """Core trié par nécessité causale décroissante."""
        return sorted(self.core, key=lambda c: self.evidence[c].necessity, reverse=True)

    def attention_mlp_split(self) -> tuple[int, int]:
        attn = sum(c.kind is ComponentKind.ATTN_HEAD for c in self.core)
        mlp = sum(c.kind is ComponentKind.MLP for c in self.core)
        return attn, mlp


# --------------------------------------------------------------------------- #
# Helpers purs (testables isolément)
# --------------------------------------------------------------------------- #
def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def split_pairs(n: int, holdout_frac: float | None, seed: int = 0) -> tuple[list[int], list[int]]:
    """Partition déterministe des indices de paires en (train, test) DISJOINTS.

    `train` sert à SÉLECTIONNER le circuit (attribution + greedy + nécessité) ; `test` à MESURER
    la faithfulness reportée — anti-tautologie. La permutation est seedée (reproductible) et
    indépendante de l'ordre des données. Si holdout désactivé / n trop petit, train == test ==
    tous les indices (comportement legacy, pas de held-out).
    """
    idx = list(range(n))
    if not holdout_frac or n <= 1:
        return idx, idx
    rng = random.Random(seed)
    perm = idx[:]
    rng.shuffle(perm)
    n_test = max(1, round(n * holdout_frac))
    n_test = min(n_test, n - 1)              # garder ≥1 paire de train
    test = sorted(perm[:n_test])
    train = sorted(perm[n_test:])
    return train, test


def _aggregate(per_pair: list[dict[Component, tuple[float, float, float]]],
               indices: list[int]) -> dict[Component, ComponentEvidence]:
    """Moyenne (dla, nec, suf) par composant sur les paires `indices`."""
    if not indices:
        return {}
    comps = per_pair[indices[0]].keys()
    out: dict[Component, ComponentEvidence] = {}
    for c in comps:
        d = sum(per_pair[i][c][0] for i in indices) / len(indices)
        n = sum(per_pair[i][c][1] for i in indices) / len(indices)
        s = sum(per_pair[i][c][2] for i in indices) / len(indices)
        out[c] = ComponentEvidence(c, d, n, s)
    return out


def _core_from_evidence(evidence: dict[Component, ComponentEvidence], threshold: float) -> set:
    return {c for c, e in evidence.items() if e.causally_validated(threshold)}


def _bootstrap_cores(per_pair, threshold, n_boot, seed) -> list[set]:
    """Cores re-estimés (seuil dur) sur `n_boot` ré-échantillons de paires avec remise."""
    n = len(per_pair)
    rng = random.Random(seed)
    cores = []
    for _ in range(n_boot):
        sample = [rng.randrange(n) for _ in range(n)]
        cores.append(_core_from_evidence(_aggregate(per_pair, sample), threshold))
    return cores


def bootstrap_stability(
    per_pair: list[dict[Component, tuple[float, float, float]]],
    threshold: float,
    n_boot: int = 200,
    seed: int = 0,
) -> float:
    """Jaccard moyen entre le core ré-estimé sur ré-échantillons et le core plein échantillon."""
    n = len(per_pair)
    if n == 0:
        return 1.0
    full_core = _core_from_evidence(_aggregate(per_pair, list(range(n))), threshold)
    cores = _bootstrap_cores(per_pair, threshold, n_boot, seed)
    return sum(jaccard(cb, full_core) for cb in cores) / len(cores) if cores else 1.0


def selection_frequencies(
    per_pair: list[dict[Component, tuple[float, float, float]]],
    threshold: float,
    n_boot: int = 200,
    seed: int = 0,
) -> dict[Component, float]:
    """Fréquence de sélection de chaque composant sur les ré-échantillons bootstrap.

    Stability selection : un composant causalement réel passe le seuil quel que soit le
    sous-échantillon (fréquence ~1) ; un composant borderline n'y passe qu'une fraction du temps.
    Cette fréquence — et non le seuil dur sur la moyenne — est le critère stable.
    """
    if not per_pair:
        return {}
    comps = list(per_pair[0].keys())
    cores = _bootstrap_cores(per_pair, threshold, n_boot, seed)
    if not cores:
        return {c: 0.0 for c in comps}
    return {c: sum(c in cb for cb in cores) / len(cores) for c in comps}


def greedy_faithful_core(ranked, faithfulness_fn, target: float, max_k: int | None = None):
    """Plus petit préfixe de `ranked` (composants triés par force causale décroissante) dont la
    faithfulness atteint `target`. Lie le core à la métrique VALIDÉE (knockout explique le
    comportement) plutôt qu'à un seuil knife-edge par score qui scinde les vrais circuits.

    `faithfulness_fn(core: list) -> float` : faithfulness du core candidat.
    Renvoie (core, k). Si la cible n'est jamais atteinte, renvoie le plus grand préfixe testé.
    """
    if not ranked:
        return [], 0
    max_k = max_k or len(ranked)
    core: list = [ranked[0]]
    for k in range(1, min(max_k, len(ranked)) + 1):
        core = list(ranked[:k])
        if faithfulness_fn(core) >= target:
            return core, k
    return core, len(core)


def extend_through_ties(ranked, necessities, k_start: int, tie_ratio: float,
                        max_k: int | None = None) -> int:
    """Étend le core de `k_start` à travers les quasi-égalités de nécessité jusqu'au prochain gap.

    Un core faithful minimal peut couper au milieu de composants causalement quasi-équivalents
    (ex. Qwen3-0.6B : L15H9≈L15MLP), ce qui rend le membership instable au bootstrap. On inclut
    le composant suivant tant que sa nécessité reste ≥ `tie_ratio` × celle du dernier inclus (et
    strictement positive) ; on s'arrête au premier gap. Renvoie le nouveau k.
    """
    n = len(ranked)
    cap = n if max_k is None else min(max_k, n)
    k = max(1, k_start)
    while k < cap:
        prev = necessities[ranked[k - 1]]
        nxt = necessities[ranked[k]]
        if prev <= 0 or nxt < tie_ratio * prev:
            break
        k += 1
    return k


def core_by_consensus(freq: dict[Component, float], consensus_frac: float) -> tuple[set, set]:
    """(core, marginal) à partir des fréquences de sélection.

    core = sélectionné dans ≥ `consensus_frac` des ré-échantillons (stable) ;
    marginal = parfois sélectionné mais sous le seuil de consensus (à signaler, hors core).
    """
    core = {c for c, f in freq.items() if f >= consensus_frac}
    marginal = {c for c, f in freq.items() if 0.0 < f < consensus_frac}
    return core, marginal


# --------------------------------------------------------------------------- #
# Métriques causales sur le circuit complet
# --------------------------------------------------------------------------- #
def _targeted_vals(target_cache, source_cache, comps, target_mask, source_mask):
    """Patchs ciblés au dernier token pour un ENSEMBLE de composants (cf. patching.targeted_patch_value)."""
    from .patching import targeted_patch_value
    return {c: targeted_patch_value(target_cache, source_cache, c, target_mask, source_mask)
            for c in comps}


@torch.no_grad()
def _circuit_metrics(backend, core, all_comps, pairs, metric):
    """faithfulness / CPR / CMD du circuit `core` agrégés sur les paires.

    Patching ciblé au dernier token (cohérent avec patching.py) : on n'altère que la position
    de décision, jamais toute la séquence.
    """
    faith = []
    cpr_num, cpr_den = [], []
    cmd = []
    for (cids, corr_ids, cmask, corrmask) in pairs:
        clean_cache = backend.run_with_cache(cids, cmask)
        corr_cache = backend.run_with_cache(corr_ids, corrmask)
        m_clean = metric(clean_cache, cmask)
        m_corr = metric(corr_cache, corrmask)
        gap = m_clean - m_corr
        if abs(gap) < 1e-9:
            continue

        # knockout du CORE sur clean (injecte corrupted au dernier token) → faithfulness + CPR num
        ko_vals = _targeted_vals(clean_cache, corr_cache, core, cmask, corrmask)
        ko_cache = backend.run_with_patches(cids, cmask, [Patch(c, ko_vals[c]) for c in core])
        m_ko = metric(ko_cache, cmask)
        # attendu : m_ko bascule vers m_corr ; faithful si franchit le milieu
        midpoint = (m_clean + m_corr) / 2
        faith.append(1.0 if (m_ko <= midpoint) == (m_corr < m_clean) else 0.0)
        cpr_num.append(m_clean - m_ko)                       # effet capté par le core

        # knockout de TOUS les composants → effet causal total (dénominateur CPR)
        ko_all_vals = _targeted_vals(clean_cache, corr_cache, all_comps, cmask, corrmask)
        ko_all = backend.run_with_patches(cids, cmask, [Patch(c, ko_all_vals[c]) for c in all_comps])
        m_ko_all = metric(ko_all, cmask)
        cpr_den.append(m_clean - m_ko_all)

        # CMD : restaure le core dans le run corrompu ; distance au modèle complet (clean)
        rs_vals = _targeted_vals(corr_cache, clean_cache, core, corrmask, cmask)
        rs_cache = backend.run_with_patches(corr_ids, corrmask, [Patch(c, rs_vals[c]) for c in core])
        m_rs = metric(rs_cache, corrmask)
        cmd.append(abs(m_rs - m_clean) / abs(gap))

    def _mean(xs):
        return sum(xs) / len(xs) if xs else None

    faithfulness = _mean(faith)
    cpr = (sum(cpr_num) / sum(cpr_den)) if cpr_den and abs(sum(cpr_den)) > 1e-9 else None
    cmd_v = _mean(cmd)
    return faithfulness, cpr, cmd_v


# --------------------------------------------------------------------------- #
# Localisation
# --------------------------------------------------------------------------- #
def localize(
    backend: CircuitBackend,
    pairs: list[tuple],
    metric: RefusalMetric,
    refusal_dirs,
    *,
    candidates: list[Component] | None = None,
    threshold: float = 0.5,
    consensus_frac: float | None = None,
    target_faithfulness: float | None = None,
    tie_ratio: float = 0.85,
    max_core: int | None = None,
    holdout_frac: float | None = None,
    min_holdout: int = 5,
    n_candidates: int = 20,
    dla_gate_quantile: float = 0.25,
    n_boot: int = 200,
    seed: int = 0,
    compute_circuit_metrics: bool = True,
) -> Localization:
    """Localise le circuit de refus.

    `pairs` : liste de tuples (clean_ids, corrupted_ids, clean_mask, corrupted_mask).
    `refusal_dirs` : tenseur (hidden,) OU liste par paire — direction de lecture DLA.
    `candidates` : sous-ensemble de composants à tester (défaut : tous). Limiter le coût.
    `consensus_frac` : si fourni, sélection par CONSENSUS bootstrap (RC2) — un composant entre
        au core ssi il passe le seuil dans ≥ `consensus_frac` des ré-échantillons ; les
        composants instables vont dans `marginal`. Si None, seuil dur sur la moyenne (legacy).
    """
    def dir_for(i):
        if isinstance(refusal_dirs, torch.Tensor) and refusal_dirs.dim() == 1:
            return refusal_dirs
        return refusal_dirs[i]

    # Split train/test déterministe : SÉLECTION sur train, MESURE (faithfulness) sur test.
    train_idx, test_idx = split_pairs(len(pairs), holdout_frac, seed)
    train_pairs = [pairs[i] for i in train_idx]
    test_pairs = [pairs[i] for i in test_idx]
    holdout_on = holdout_frac is not None and test_idx != train_idx
    warn = None
    if holdout_on and len(test_idx) < min_holdout:
        warn = (f"held-out test-set = {len(test_idx)} paires (< min_holdout={min_holdout}) : "
                f"faithfulness held-out peu significative — augmenter le nombre de paires.")

    # Candidats dérivés sur le TRAIN uniquement (sinon fuite de sélection vers la mesure).
    if candidates is None:
        from .attribution import aggregate_attribution
        rd = (refusal_dirs if isinstance(refusal_dirs, torch.Tensor)
              else [refusal_dirs[i] for i in train_idx])
        candidates = [c for c, _ in
                      aggregate_attribution(backend, train_pairs, refusal_dir=rd).top(n_candidates)]

    # collecte per-paire : (dla, necessity, sufficiency) par composant
    per_pair: list[dict[Component, tuple[float, float, float]]] = []
    for i, (cids, corr_ids, cmask, corrmask) in enumerate(pairs):
        dla = direct_logit_attribution(backend, dir_for(i), cids, cmask)
        clean_cache = backend.run_with_cache(cids, cmask)
        corr_cache = backend.run_with_cache(corr_ids, corrmask)
        row: dict[Component, tuple[float, float, float]] = {}
        for c in candidates:
            v = validate_component(backend, c, cids, corr_ids, metric, threshold,
                                   cmask, corrmask, clean_cache, corr_cache)
            row[c] = (dla.scores.get(c, 0.0), v.necessity_recovery, v.sufficiency_recovery)
        per_pair.append(row)

    # Évidence de SÉLECTION : agrégée sur le train uniquement.
    evidence = _aggregate(per_pair, train_idx)

    marginal: list[Component] = []
    freq: dict[Component, float] | None = None
    precomputed_metrics: tuple | None = None
    insample_faith: float | None = None
    if target_faithfulness is not None:
        # RC2 corrigé : core = plus petit ensemble (par nécessité causale décroissante) dont le
        # knockout explique le comportement (faithfulness ≥ cible). Évite le seuil dur knife-edge
        # qui scinde un vrai circuit (ex. Qwen3-0.6B : L15H9 a nec 0.526 mais suf 0.443).
        ranked_all = sorted(evidence, key=lambda c: evidence[c].necessity, reverse=True)

        # SÉLECTION : la faithfulness qui pilote greedy est mesurée sur le TRAIN.
        def _faith(core_list):
            return _circuit_metrics(backend, core_list, candidates, train_pairs, metric)[0] or 0.0

        core_list, kstar = greedy_faithful_core(ranked_all, _faith, target_faithfulness,
                                                max_k=max_core)
        # ne pas couper au milieu d'une quasi-égalité causale → étendre jusqu'au prochain gap.
        necs = {c: evidence[c].necessity for c in ranked_all}
        kstar = extend_through_ties(ranked_all, necs, kstar, tie_ratio, max_k=max_core)
        core_list = list(ranked_all[:kstar])
        core_set = set(core_list)
        # stabilité : ré-échantillonnage du TRAIN ; top-k* par nécessité ; Jaccard vs core retenu.
        per_train = [per_pair[i] for i in train_idx]
        rng = random.Random(seed)
        nt = len(per_train)
        js = []
        for _ in range(n_boot):
            sample = [rng.randrange(nt) for _ in range(nt)]
            ev_b = _aggregate(per_train, sample)
            ranked_b = sorted(ev_b, key=lambda c: ev_b[c].necessity, reverse=True)
            js.append(jaccard(set(ranked_b[:kstar]), core_set))
        boot_j = sum(js) / len(js) if js else 1.0
        freq = selection_frequencies(per_train, threshold, n_boot=n_boot, seed=seed)
        if compute_circuit_metrics and core_list:
            # MESURE AUTORITAIRE : faithfulness sur le held-out (test) ; in-sample en transparence.
            precomputed_metrics = _circuit_metrics(backend, core_list, candidates, test_pairs, metric)
            insample_faith = _circuit_metrics(backend, core_list, candidates, train_pairs, metric)[0]
    elif consensus_frac is not None:
        # RC2 : sélection par consensus bootstrap (stable par construction).
        freq = selection_frequencies(per_pair, threshold, n_boot=n_boot, seed=seed)
        core_set, marginal_set = core_by_consensus(freq, consensus_frac)
        marginal = list(marginal_set)
        # stabilité = accord des ré-échantillons avec le core consensus retenu.
        cores_b = _bootstrap_cores(per_pair, threshold, n_boot, seed)
        boot_j = (sum(jaccard(cb, core_set) for cb in cores_b) / len(cores_b)
                  if cores_b else 1.0)
    else:
        core_set = _core_from_evidence(evidence, threshold)
        boot_j = bootstrap_stability(per_pair, threshold, n_boot=n_boot, seed=seed)
    core = list(core_set)

    # motif gate→amplificateur : parmi le core (donc causal), séparer par la DLA.
    # amplificateur = causal AVEC forte DLA ; gate = causal avec DLA quasi nulle.
    gates, amplifiers = [], []
    if core:
        dlas = sorted(abs(evidence[c].dla) for c in core)
        cut = dlas[max(0, int(len(dlas) * dla_gate_quantile) - 1)] if dlas else 0.0
        for c in core:
            (gates if abs(evidence[c].dla) <= cut else amplifiers).append(c)

    loc = Localization(evidence=evidence, core=core, threshold=threshold,
                       gates=gates, amplifiers=amplifiers, marginal=marginal,
                       selection_frequency=freq,
                       n_train=len(train_idx), n_test=len(test_idx),
                       held_out=holdout_on, holdout_warning=warn)
    loc.bootstrap_jaccard = boot_j

    if precomputed_metrics is not None:
        loc.faithfulness, loc.cpr, loc.cmd = precomputed_metrics
        loc.faithfulness_insample = insample_faith
    elif compute_circuit_metrics and core:
        # branches consensus/legacy : mesure held-out (test) + in-sample (train) en transparence.
        loc.faithfulness, loc.cpr, loc.cmd = _circuit_metrics(backend, core, candidates,
                                                              test_pairs, metric)
        loc.faithfulness_insample = _circuit_metrics(backend, core, candidates,
                                                     train_pairs, metric)[0]
    return loc
