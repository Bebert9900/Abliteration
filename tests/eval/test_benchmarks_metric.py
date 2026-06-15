"""Test du sélecteur de métrique lm-eval (extraction de la métrique phare du dict de résultats)."""
import pytest

from abliteration.eval.benchmarks import _pick_metric, available_benchmarks, run_benchmark


def test_pick_metric_prefers_named_metric():
    results = {"acc,none": 0.42, "acc_stderr,none": 0.01}
    key, val = _pick_metric(results, "acc")
    assert key == "acc,none"
    assert val == pytest.approx(0.42)


def test_pick_metric_falls_back_to_first_numeric_non_stderr():
    results = {"exact_match,strict-match": 0.31, "exact_match_stderr,strict-match": 0.02}
    key, val = _pick_metric(results, "acc")  # 'acc' absent -> repli
    assert key == "exact_match,strict-match"
    assert val == pytest.approx(0.31)


def test_run_benchmark_rejects_unknown_name():
    with pytest.raises(ValueError):
        run_benchmark("not_a_benchmark", "some-model")


def test_known_benchmarks_listed():
    assert {"mmlu", "gsm8k"} <= set(available_benchmarks())
