"""Tests des adaptateurs de benchmarks externes (guarded) et de l'agrégation de rapport."""
import json

import pytest

from src.eval import (
    BenchmarkNotInstalled,
    EvalReport,
    available_benchmarks,
    run_benchmark,
)


def test_unknown_benchmark_raises_value_error():
    with pytest.raises(ValueError, match="inconnu"):
        run_benchmark("benchmark_qui_nexiste_pas", model=None)


def test_known_but_uninstalled_benchmark_raises_not_installed():
    # bfcl/ifeval/taubench ne sont pas installés dans cet environnement.
    assert "bfcl" in available_benchmarks()
    with pytest.raises(BenchmarkNotInstalled, match="pip install"):
        run_benchmark("bfcl", model=None)


def test_eval_report_serializes_two_axes_and_pareto_table(tmp_path):
    report = EvalReport(
        refusal_rate=0.12,
        kl=0.18,
        negation_retention=0.95,
        follow_rate=0.05,
        empty_rate=0.0,
        agentic_score=0.88,
        degeneracy_rate=0.01,
    )
    d = report.to_dict()
    assert d["refusal_rate"] == 0.12 and d["agentic_score"] == 0.88
    path = tmp_path / "report.json"
    report.save(path)
    assert json.loads(path.read_text())["kl"] == 0.18
