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
  niveau corrompu, au-delà du milieu).
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
    bootstrap_jaccard: float | None = None
    faithfulness: float | None = None
    cpr: float | None = None
    cmd: float | None = None

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


def bootstrap_stability(
    per_pair: list[dict[Component, tuple[float, float, float]]],
    threshold: float,
    n_boot: int = 200,
    seed: int = 0,
) -> float:
    """Jaccard moyen entre le core ré-estimé sur ré-échantillons et le core plein échantillon."""
    n = len(per_pair)
    full_core = _core_from_evidence(_aggregate(per_pair, list(range(n))), threshold)
    rng = random.Random(seed)
    scores = []
    for _ in range(n_boot):
        sample = [rng.randrange(n) for _ in range(n)]
        core_b = _core_from_evidence(_aggregate(per_pair, sample), threshold)
        scores.append(jaccard(core_b, full_core))
    return sum(scores) / len(scores) if scores else 1.0


# --------------------------------------------------------------------------- #
# Métriques causales sur le circuit complet
# --------------------------------------------------------------------------- #
def _last_vals(cache, comps, mask, broadcast_seq):
    from src.data.formatting import last_token_index
    out = {}
    for c in comps:
        contrib = cache.component(c)
        b, s = contrib.shape[0], contrib.shape[1]
        idx = last_token_index(mask) if mask is not None \
            else torch.full((b,), s - 1, dtype=torch.long)
        batch = torch.arange(b)
        last = contrib[batch, idx, :]
        out[c] = last.unsqueeze(1).expand(b, broadcast_seq, contrib.shape[-1]).clone()
    return out


@torch.no_grad()
def _circuit_metrics(backend, core, all_comps, pairs, metric):
    """faithfulness / CPR / CMD du circuit `core` agrégés sur les paires."""
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

        # knockout du CORE sur clean (injecte corrupted) → faithfulness + CPR(numérateur)
        ko_vals = _last_vals(corr_cache, core, corrmask, cids.shape[1])
        ko_cache = backend.run_with_patches(cids, cmask, [Patch(c, ko_vals[c]) for c in core])
        m_ko = metric(ko_cache, cmask)
        # attendu : m_ko bascule vers m_corr ; faithful si franchit le milieu
        midpoint = (m_clean + m_corr) / 2
        faith.append(1.0 if (m_ko <= midpoint) == (m_corr < m_clean) else 0.0)
        cpr_num.append(m_clean - m_ko)                       # effet capté par le core

        # knockout de TOUS les composants → effet causal total (dénominateur CPR)
        ko_all_vals = _last_vals(corr_cache, all_comps, corrmask, cids.shape[1])
        ko_all = backend.run_with_patches(cids, cmask, [Patch(c, ko_all_vals[c]) for c in all_comps])
        m_ko_all = metric(ko_all, cmask)
        cpr_den.append(m_clean - m_ko_all)

        # CMD : restaure le core dans le run corrompu ; distance au modèle complet (clean)
        rs_vals = _last_vals(clean_cache, core, cmask, corr_ids.shape[1])
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
    dla_gate_quantile: float = 0.25,
    n_boot: int = 200,
    seed: int = 0,
    compute_circuit_metrics: bool = True,
) -> Localization:
    """Localise le circuit de refus.

    `pairs` : liste de tuples (clean_ids, corrupted_ids, clean_mask, corrupted_mask).
    `refusal_dirs` : tenseur (hidden,) OU liste par paire — direction de lecture DLA.
    `candidates` : sous-ensemble de composants à tester (défaut : tous). Limiter le coût.
    """
    candidates = candidates or backend.all_components()

    def dir_for(i):
        if isinstance(refusal_dirs, torch.Tensor) and refusal_dirs.dim() == 1:
            return refusal_dirs
        return refusal_dirs[i]

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

    evidence = _aggregate(per_pair, list(range(len(pairs))))
    core_set = _core_from_evidence(evidence, threshold)
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
                       gates=gates, amplifiers=amplifiers)
    loc.bootstrap_jaccard = bootstrap_stability(per_pair, threshold, n_boot=n_boot, seed=seed)

    if compute_circuit_metrics and core:
        loc.faithfulness, loc.cpr, loc.cmd = _circuit_metrics(
            backend, core, candidates, pairs, metric
        )
    return loc
