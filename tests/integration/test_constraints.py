"""Test d'intégration de bout en bout des CONTRAINTES du projet.

Thèse centrale : la variante `preserving` retire le refus SANS casser
(1) la négation logique légitime ni (2) les capacités agentiques (tool use).

On câble la vraie chaîne : compute_directions → ablation_direction(variant) →
orthogonalize_weights → ArchAdapter, sur un modèle jouet torch. La mesure est
déterministe : un writer résiduel initialisé à l'identité `I` devient, après
orthogonalisation contre la direction unitaire `d`, la projection `I − d·dᵀ`.
La rétention d'un vecteur sonde unitaire `p` à travers ce writer vaut donc
exactement `pᵀ(I − d·dᵀ)p = 1 − (d·p)²`. C'est notre proxy mesurable de la
« capacité à écrire p » dans le residual stream après ablation.
"""
import torch
import torch.nn as nn

from abliteration.ablation import Variant, ablation_direction, orthogonalize_weights
from abliteration.data import PromptClass
from abliteration.directions import compute_directions
from abliteration.models import ArchAdapter

H = 6  # dimension cachée du modèle jouet


# --------------------------------------------------------------------------- #
# Modèle jouet : un seul bloc dense, writers résiduels initialisés à l'identité
# --------------------------------------------------------------------------- #
class IdentityDenseModel(nn.Module):
    """Bloc dense minimal ; o_proj/down_proj/embedding contrôlés pour une mesure exacte."""

    def __init__(self, h=H):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(10, h)
        block = nn.Module()
        block.self_attn = nn.Module()
        block.self_attn.o_proj = nn.Linear(h, h, bias=False)
        block.mlp = nn.Module()
        block.mlp.down_proj = nn.Linear(h, h, bias=False)
        # writers initialisés à l'identité → rétention = 1 − (d·p)²
        block.self_attn.o_proj.weight.data = torch.eye(h)
        block.mlp.down_proj.weight.data = torch.eye(h)
        self.model.layers = nn.ModuleList([block])

    def get_input_embeddings(self):
        return self.model.embed_tokens


def _retention(model: IdentityDenseModel, probe: torch.Tensor) -> float:
    """Fraction de `probe` (unitaire) encore écrite par o_proj après ablation."""
    W = model.model.layers[0].self_attn.o_proj.weight.data.float()
    p = probe.float()
    p = p / (p.norm() + 1e-8)
    return float(p @ (W @ p))


def _make_directions():
    """4 classes avec harmless = 0 → refusal/negation/agentic = vecteurs choisis.

    Le refus CHEVAUCHE volontairement négation (e0) et agentique (e1), plus une
    composante unique (e2) : c'est exactement le cas où l'ablation conventionnelle
    abîme les capacités à préserver.
    """
    e = torch.eye(H)
    negation = e[0]
    agentic = e[1]
    refusal = e[2] + 0.6 * e[0] + 0.5 * e[1]  # unique + chevauchement n̂/â

    # means: [n_layers=1, hidden]; harmless nul, les autres = direction voulue
    zero = torch.zeros(1, H)
    means = {
        PromptClass.HARMLESS: zero,
        PromptClass.HARMFUL: refusal.unsqueeze(0),
        PromptClass.LEGITIMATE_NEGATION: negation.unsqueeze(0),
        PromptClass.AGENTIC: agentic.unsqueeze(0),
    }
    return compute_directions(means), negation, agentic, refusal


def _ablate(variant: Variant, preserve):
    directions, negation, agentic, refusal = _make_directions()
    d = ablation_direction(directions, layer=0, variant=variant, preserve=preserve)
    model = IdentityDenseModel()
    orthogonalize_weights(ArchAdapter(model), d)
    return model, negation, agentic, refusal


# --------------------------------------------------------------------------- #
# Contrainte 1+2 : preserving préserve négation ET agentique
# --------------------------------------------------------------------------- #
def test_preserving_keeps_negation_and_agentic_intact():
    model, negation, agentic, _ = _ablate(Variant.PRESERVING, ["negation", "agentic"])
    assert _retention(model, negation) == _almost(1.0)
    assert _retention(model, agentic) == _almost(1.0)


# --------------------------------------------------------------------------- #
# Contraste : conventional ABÎME les capacités que preserving protège
# --------------------------------------------------------------------------- #
def test_conventional_damages_what_preserving_protects():
    conv, negation, agentic, _ = _ablate(Variant.CONVENTIONAL, None)
    pres, _, _, _ = _ablate(Variant.PRESERVING, ["negation", "agentic"])

    # conventional perd une part mesurable de négation/agentique...
    assert _retention(conv, negation) < 0.95
    assert _retention(conv, agentic) < 0.95
    # ...là où preserving les garde strictement mieux.
    assert _retention(pres, negation) > _retention(conv, negation)
    assert _retention(pres, agentic) > _retention(conv, agentic)


# --------------------------------------------------------------------------- #
# Canari : dans les deux cas, le REFUS est bien réduit (la méthode marche)
# --------------------------------------------------------------------------- #
def test_refusal_is_reduced_in_both_variants():
    conv, _, _, refusal = _ablate(Variant.CONVENTIONAL, None)
    pres, _, _, refusal_p = _ablate(Variant.PRESERVING, ["negation", "agentic"])

    # conventional retire entièrement r̂ (rétention ≈ 0).
    assert _retention(conv, refusal) == _almost(0.0, tol=1e-5)
    # preserving réduit substantiellement le refus, même en protégeant n̂/â.
    assert _retention(pres, refusal_p) < 0.6


# --------------------------------------------------------------------------- #
# La préservation s'applique aussi aux autres writers résiduels (down_proj).
# --------------------------------------------------------------------------- #
def test_all_residual_writers_are_orthogonalized():
    model, negation, _, _ = _ablate(Variant.PRESERVING, ["negation", "agentic"])
    W_down = model.model.layers[0].mlp.down_proj.weight.data.float()
    n = negation / negation.norm()
    # négation préservée à travers down_proj aussi.
    assert float(n @ (W_down @ n)) == _almost(1.0)


# --------------------------------------------------------------------------- #
# Helper de comparaison approchée (lisible dans les assert ci-dessus).
# --------------------------------------------------------------------------- #
class _almost:
    def __init__(self, value, tol=1e-4):
        self.value = value
        self.tol = tol

    def __eq__(self, other):
        return abs(other - self.value) <= self.tol

    def __repr__(self):
        return f"≈{self.value}±{self.tol}"
