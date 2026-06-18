"""Test de la sélection de couche via une fonction de score injectée (hooks réversibles)."""
from meridian.directions import select_layer


def test_select_layer_picks_lowest_refusal_score():
    # score_fn simule : poser le hook d'ablation à la couche L, mesurer le refus.
    scores = {10: 0.8, 14: 0.1, 18: 0.4}
    chosen = select_layer(candidate_layers=[10, 14, 18], score_fn=lambda l: scores[l])
    assert chosen == 14
