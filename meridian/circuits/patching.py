"""Activation patching causal — méthode 2/3, CAUSALE (cher).

C'est la méthode qui CONFIRME (ou réfute) les hypothèses de la DLA. La DLA dit « qui
contribue » ; le patching dit « qui contrôle », par intervention.

Protocole counterfactual (skill abliteration-circuits, circuit_analysis.md) :
- run **clean**     : prompt harmful → le modèle refuse.
- run **corrupted** : variante minimalement différente qui NE déclenche pas le refus.
On substitue la contribution residual-espace d'UN composant d'un run dans l'autre, tout le
reste fixe, et on mesure le déplacement de la métrique de refus.

Métrique : **logit-diff de refus** = logit(token_refus) − logit(token_réponse) au dernier
token. Élevée sur le run clean (refus), basse sur corrupted. On peut aussi projeter le résidu
final sur r̂ (métrique directionnelle) — les deux sont fournies via `RefusalMetric`.

Deux tests causaux distincts (à ne pas confondre) :
- **Nécessité** (knockout) : sur le run CLEAN (qui refuse), remplacer la contribution du
  composant par celle du run CORROMPU (ou par zéro). Si le refus s'effondre → le composant est
  NÉCESSAIRE au refus.
- **Suffisance** (restauration) : sur le run CORROMPU (qui ne refuse pas), injecter la
  contribution du composant issue du run CLEAN. Si le refus réapparaît → le composant est
  SUFFISANT (sur ce contexte) pour pousser au refus.

Un composant n'est « validé causalement » que s'il passe le seuil sur ces tests d'intervention,
jamais sur la DLA seule (règle d'or).
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from meridian.data.formatting import last_token_index

from .backend import CircuitBackend, Component, ComponentCache, Patch


# --------------------------------------------------------------------------- #
# Métrique de refus
# --------------------------------------------------------------------------- #
@dataclass
class RefusalMetric:
    """Mesure scalaire « à quel point ça refuse » à partir d'un ComponentCache.

    Deux modes :
    - logit-diff : logits[refusal_token] − logits[answer_token] au dernier token (recommandé,
      circuit_analysis.md) ;
    - directionnel : projection du résidu final sur r̂ (fallback si pas de tokens cibles).
    Plus la valeur est haute, plus le modèle « refuse ».
    """
    refusal_token: int | None = None
    answer_token: int | None = None
    refusal_dir: torch.Tensor | None = None

    def __post_init__(self):
        has_tokens = self.refusal_token is not None and self.answer_token is not None
        if not has_tokens and self.refusal_dir is None:
            raise ValueError("RefusalMetric requiert (refusal_token, answer_token) ou refusal_dir.")

    def __call__(self, cache: ComponentCache, attention_mask: torch.Tensor | None = None) -> float:
        if cache.logits is not None and self.refusal_token is not None:
            logits = cache.logits.to(torch.float32)
            b, s = logits.shape[0], logits.shape[1]
            idx = last_token_index(attention_mask) if attention_mask is not None \
                else torch.full((b,), s - 1, dtype=torch.long)
            batch = torch.arange(b)
            last = logits[batch, idx, :]                              # (b, vocab)
            diff = last[:, self.refusal_token] - last[:, self.answer_token]
            return float(diff.mean())
        # mode directionnel
        resid = cache.final_resid.to(torch.float32)
        b, s = resid.shape[0], resid.shape[1]
        idx = last_token_index(attention_mask) if attention_mask is not None \
            else torch.full((b,), s - 1, dtype=torch.long)
        batch = torch.arange(b)
        r = self.refusal_dir.to(torch.float32)
        r = r / (r.norm() + 1e-8)
        return float((resid[batch, idx, :] @ r).mean())


# --------------------------------------------------------------------------- #
# Résultats
# --------------------------------------------------------------------------- #
@dataclass
class PatchEffect:
    """Effet causal d'une intervention sur un composant, pour UN test (nécessité|suffisance)."""
    component: Component
    test: str                 # "necessity" | "sufficiency"
    baseline: float           # métrique avant intervention
    patched: float            # métrique après intervention
    reference: float          # métrique du run de référence opposé (corrupted|clean)

    @property
    def delta(self) -> float:
        """Variation signée de la métrique due à l'intervention (patched − baseline)."""
        return self.patched - self.baseline

    @property
    def recovery(self) -> float:
        """Fraction de l'écart baseline↔reference que l'intervention fait franchir, ~[0, 1].

        `gap = baseline − reference` est l'écart total à combler (clean↔corrupted) ;
        `baseline − patched` est ce que l'intervention déplace vers la référence.
        Ratio → comparable entre composants. 1.0 = le composant explique tout l'effet ;
        0.0 = aucun effet causal détecté.
        """
        gap = self.baseline - self.reference
        if abs(gap) < 1e-9:
            return 0.0
        return float((self.baseline - self.patched) / gap)


# --------------------------------------------------------------------------- #
# Patching causal
# --------------------------------------------------------------------------- #
def _last_idx(mask, b, s, device):
    return (last_token_index(mask) if mask is not None
            else torch.full((b,), s - 1, dtype=torch.long, device=device))


def targeted_patch_value(target_cache: ComponentCache, source_cache: ComponentCache,
                         c: Component, target_mask, source_mask):
    """Valeur de patch CIBLÉE AU DERNIER TOKEN (activation patching propre).

    Renvoie la contribution PROPRE du run-cible sur toute la séquence, en remplaçant
    UNIQUEMENT la position du dernier token par celle du run-source. Le delta injecté est donc
    nul partout sauf au dernier token → on n'altère que la position de décision du refus, et les
    longueurs des deux runs peuvent différer.

    (Une version antérieure broadcastait la valeur source sur toute la séquence : correct
    seulement pour des séquences de longueur 1, faux sur un vrai modèle — cause de CPR/CMD
    aberrants.)
    """
    base = target_cache.component(c).clone()                    # (b, s_t, hidden) — run cible
    src = source_cache.component(c)                             # (b, s_s, hidden) — run source
    b, s_t, s_s = base.shape[0], base.shape[1], src.shape[1]
    batch = torch.arange(b, device=base.device)
    ti = _last_idx(target_mask, b, s_t, base.device)
    si = _last_idx(source_mask, b, s_s, src.device)
    base[batch, ti, :] = src[batch, si, :].to(base.dtype)
    return base


@torch.no_grad()
def necessity(
    backend: CircuitBackend,
    component: Component,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    metric: RefusalMetric,
    clean_mask: torch.Tensor | None = None,
    corrupted_mask: torch.Tensor | None = None,
    clean_cache: ComponentCache | None = None,
    corrupted_cache: ComponentCache | None = None,
) -> PatchEffect:
    """Knockout : injecte la contribution CORROMPUE du composant dans le run CLEAN.

    Si le refus chute (delta négatif important, recovery élevé), le composant est NÉCESSAIRE.
    """
    clean_cache = clean_cache or backend.run_with_cache(clean_ids, clean_mask)
    corrupted_cache = corrupted_cache or backend.run_with_cache(corrupted_ids, corrupted_mask)

    baseline = metric(clean_cache, clean_mask)
    reference = metric(corrupted_cache, corrupted_mask)

    # cible = clean ; on n'échange que le dernier token par la valeur corrompue
    val = targeted_patch_value(clean_cache, corrupted_cache, component, clean_mask, corrupted_mask)
    patched_cache = backend.run_with_patches(clean_ids, clean_mask, [Patch(component, val)])
    patched = metric(patched_cache, clean_mask)

    return PatchEffect(component, "necessity", baseline, patched, reference)


@torch.no_grad()
def sufficiency(
    backend: CircuitBackend,
    component: Component,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    metric: RefusalMetric,
    clean_mask: torch.Tensor | None = None,
    corrupted_mask: torch.Tensor | None = None,
    clean_cache: ComponentCache | None = None,
    corrupted_cache: ComponentCache | None = None,
) -> PatchEffect:
    """Restauration : injecte la contribution CLEAN du composant dans le run CORROMPU.

    Si le refus réapparaît (delta positif important, recovery élevé), le composant est
    SUFFISant (sur ce contexte) pour pousser au refus.
    """
    clean_cache = clean_cache or backend.run_with_cache(clean_ids, clean_mask)
    corrupted_cache = corrupted_cache or backend.run_with_cache(corrupted_ids, corrupted_mask)

    baseline = metric(corrupted_cache, corrupted_mask)
    reference = metric(clean_cache, clean_mask)

    # cible = corrompu ; on n'échange que le dernier token par la valeur clean
    val = targeted_patch_value(corrupted_cache, clean_cache, component, corrupted_mask, clean_mask)
    patched_cache = backend.run_with_patches(corrupted_ids, corrupted_mask, [Patch(component, val)])
    patched = metric(patched_cache, corrupted_mask)

    return PatchEffect(component, "sufficiency", baseline, patched, reference)


@dataclass
class CausalVerdict:
    """Synthèse des deux tests causaux pour un composant + verdict booléen explicite."""
    component: Component
    necessity: PatchEffect
    sufficiency: PatchEffect
    threshold: float

    @property
    def necessity_recovery(self) -> float:
        return self.necessity.recovery

    @property
    def sufficiency_recovery(self) -> float:
        return self.sufficiency.recovery

    @property
    def is_necessary(self) -> bool:
        return self.necessity.recovery >= self.threshold

    @property
    def is_sufficient(self) -> bool:
        return self.sufficiency.recovery >= self.threshold

    @property
    def causally_validated(self) -> bool:
        """Validé = nécessaire ET suffisant au-dessus du seuil. Le rapport n'affiche « validé »
        que pour ces composants (jamais sur la DLA seule)."""
        return self.is_necessary and self.is_sufficient


@torch.no_grad()
def validate_component(
    backend: CircuitBackend,
    component: Component,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    metric: RefusalMetric,
    threshold: float = 0.5,
    clean_mask: torch.Tensor | None = None,
    corrupted_mask: torch.Tensor | None = None,
    clean_cache: ComponentCache | None = None,
    corrupted_cache: ComponentCache | None = None,
) -> CausalVerdict:
    """Lance nécessité + suffisance sur un composant et rend un verdict causal seuillé."""
    clean_cache = clean_cache or backend.run_with_cache(clean_ids, clean_mask)
    corrupted_cache = corrupted_cache or backend.run_with_cache(corrupted_ids, corrupted_mask)
    nec = necessity(backend, component, clean_ids, corrupted_ids, metric,
                    clean_mask, corrupted_mask, clean_cache, corrupted_cache)
    suf = sufficiency(backend, component, clean_ids, corrupted_ids, metric,
                      clean_mask, corrupted_mask, clean_cache, corrupted_cache)
    return CausalVerdict(component, nec, suf, threshold)
