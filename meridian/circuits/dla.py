"""Direct Logit Attribution (DLA) — méthode 1/3, CORRÉLATIONNELLE.

Projette la contribution residual-espace de CHAQUE composant (tête, MLP) sur la direction de
refus `r̂` (réutilisée depuis `abliteration/directions/`, jamais recalculée ici). Donne « qui contribue »
au refus, par projection scalaire au dernier token de l'instruction.

⚠️ RÈGLE D'OR (skill abliteration-circuits) : la DLA GÉNÈRE DES HYPOTHÈSES, elle ne conclut
jamais seule. Aucune intervention → aucune preuve causale. Un composant haut en DLA doit être
CONFIRMÉ par patching causal (nécessité + suffisance) avant d'être déclaré « validé ». À
l'inverse, une tête `gate` peut avoir un signal DLA quasi nul (<1 %) tout en étant causalement
nécessaire → ne jamais filtrer un composant sur sa seule absence en DLA.

Maths : pour un composant c de contribution residual-espace x_c (au dernier token),
    score_c = ⟨ x_c , r̂ ⟩            (projection signée ; >0 = pousse vers le refus)
La somme des scores sur tous les composants ≈ projection du résidu final sur r̂ (aux termes
embedding/biais/LayerNorm près) — d'où la lecture « part du signal de refus portée par c ».
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from meridian.data.formatting import last_token_index

from .backend import CircuitBackend, Component, ComponentKind

CORRELATIONAL_CAVEAT = (
    "CORRÉLATIONNEL — hypothèse, PAS une conclusion. La DLA mesure « qui contribue » au refus, "
    "pas « qui le contrôle ». À confirmer par patching causal (nécessité + suffisance) avant "
    "toute validation. Une tête gate peut être quasi nulle en DLA mais causalement nécessaire."
)


def readout_direction(directions, layer: int) -> torch.Tensor:
    """Extrait la direction de refus unitaire `r̂_layer` d'un objet `Directions` (meridian.directions).

    On lit, on ne recalcule pas. La direction sert d'axe de lecture (readout) de la DLA.
    """
    r = directions.refusal[layer]
    return r / (r.norm() + 1e-8)


@dataclass
class DLAResult:
    """Scores DLA signés par composant + métadonnées. NON causal (voir `caveat`)."""
    scores: dict[Component, float]
    readout_layer: int
    method: str = "direct_logit_attribution"
    caveat: str = field(default=CORRELATIONAL_CAVEAT)

    def ranked(self, by_abs: bool = True) -> list[tuple[Component, float]]:
        key = (lambda kv: abs(kv[1])) if by_abs else (lambda kv: kv[1])
        return sorted(self.scores.items(), key=key, reverse=True)

    def top(self, k: int, by_abs: bool = True) -> list[tuple[Component, float]]:
        return self.ranked(by_abs=by_abs)[:k]

    def attention_mlp_ratio(self) -> tuple[float, float]:
        """(fraction attention, fraction MLP) de la magnitude totale du signal direct.

        Repère typique documenté (corpus-dépendant, circuit_analysis.md) : attn ~77 %,
        MLP ~23 % sur Qwen3-8B. Ici on rapporte la valeur MESURÉE, pas le repère.
        """
        attn = sum(abs(v) for c, v in self.scores.items() if c.kind is ComponentKind.ATTN_HEAD)
        mlp = sum(abs(v) for c, v in self.scores.items() if c.kind is ComponentKind.MLP)
        total = attn + mlp
        if total == 0:
            return (0.0, 0.0)
        return (attn / total, mlp / total)


@torch.no_grad()
def direct_logit_attribution(
    backend: CircuitBackend,
    refusal_dir: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    include_mlp: bool = True,
) -> DLAResult:
    """Projette la contribution de chaque composant sur `refusal_dir` (unitaire) au dernier token.

    `refusal_dir` : (hidden,) — typiquement `readout_direction(directions, layer)`.
    Le score d'un composant est moyenné sur le batch.
    """
    cache = backend.run_with_cache(input_ids, attention_mask)

    # position du dernier token réel par séquence (robuste au padding gauche/droite)
    if attention_mask is not None:
        idx = last_token_index(attention_mask)              # (b,)
    else:
        idx = torch.full((input_ids.size(0),), input_ids.size(1) - 1, dtype=torch.long)
    batch = torch.arange(input_ids.size(0))

    r = refusal_dir.to(torch.float32)
    r = r / (r.norm() + 1e-8)

    scores: dict[Component, float] = {}
    for c in backend.all_components(include_mlp=include_mlp):
        contrib = cache.component(c).to(torch.float32)      # (b, s, hidden)
        last = contrib[batch, idx, :]                       # (b, hidden)
        proj = last @ r                                     # (b,)
        scores[c] = float(proj.mean())

    readout_layer = getattr(refusal_dir, "_readout_layer", -1)
    return DLAResult(scores=scores, readout_layer=readout_layer)
