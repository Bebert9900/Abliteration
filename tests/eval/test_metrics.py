"""Tests des métriques non-agentiques : dégénérescence, refus, KL, négation, follow/empty."""
import math

import torch

from abliteration.eval import (
    KeywordRefusalJudge,
    empty_rate,
    follow_rate,
    is_degenerate,
    kl_divergence,
    negation_retention,
    refusal_rate,
)


def test_is_degenerate_flags_heavy_repetition():
    assert is_degenerate("spam spam spam spam spam spam spam spam") is True
    assert is_degenerate("La photosynthèse convertit la lumière en énergie chimique.") is False


def test_is_degenerate_flags_empty_or_tiny():
    assert is_degenerate("") is True
    assert is_degenerate("ok") is True  # trop court pour le contrat min_new_tokens


def test_keyword_judge_detects_refusal_phrases():
    judge = KeywordRefusalJudge()
    assert judge("...", "I'm sorry, but I can't help with that request.") is True
    assert judge("...", "Sure, here is a detailed explanation of the topic ...") is False


def test_refusal_rate_counts_judged_refusals():
    judge = lambda prompt, resp: resp.startswith("REFUS")
    responses = ["REFUS a", "ok b", "REFUS c", "ok d"]
    rate = refusal_rate(["p"] * 4, responses, judge)
    assert rate == 0.5


def test_kl_divergence_is_zero_for_identical_distributions():
    logits = torch.randn(3, 10)
    assert kl_divergence(logits, logits.clone()) < 1e-6


def test_kl_divergence_matches_manual_two_point():
    # P=softmax([0, ln3]) = [1/4, 3/4] ; Q uniforme [1/2,1/2]
    p = torch.tensor([[0.0, math.log(3)]])
    q = torch.tensor([[0.0, 0.0]])
    expected = 0.25 * math.log(0.25 / 0.5) + 0.75 * math.log(0.75 / 0.5)
    assert abs(kl_divergence(p, q) - expected) < 1e-5


def test_negation_retention_fraction_of_correct_negations():
    checker = lambda resp: "non" in resp.lower()
    responses = ["Non, ce code est faux.", "Oui c'est correct", "Non, erreur ligne 3"]
    assert abs(negation_retention(responses, checker) - 2 / 3) < 1e-9


def test_follow_rate_and_empty_rate():
    assert follow_rate(["yes", "no", "yes", "yes"], is_sycophantic=lambda r: r == "yes") == 0.75
    assert empty_rate(["", "  ", "contenu"]) == 2 / 3
