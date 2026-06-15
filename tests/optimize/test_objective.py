"""Tests de l'objectif composite et du checkpoint resume-safe."""
import pytest

from abliteration.eval import EvalReport
from abliteration.optimize import (
    Lambdas,
    best_trial,
    build_objective,
    composite_objective,
    load_trials,
    run_optuna_study,
    save_trial,
)


class _FakeTrial:
    """Trial Optuna minimal : renvoie des valeurs fixées d'avance pour layer/alpha."""
    def __init__(self, layer, alpha):
        self._layer, self._alpha = layer, alpha
        self.params = {}

    def suggest_categorical(self, name, choices):
        assert self._layer in choices
        self.params[name] = self._layer
        return self._layer

    def suggest_float(self, name, low, high):
        assert low <= self._alpha <= high
        self.params[name] = self._alpha
        return self._alpha


def _report(refusal=0.0, kl=0.0, neg=1.0, follow=0.0, agentic=1.0):
    return EvalReport(
        refusal_rate=refusal, kl=kl, negation_retention=neg, follow_rate=follow,
        empty_rate=0.0, agentic_score=agentic, degeneracy_rate=0.0,
    )


def test_perfect_model_has_zero_objective():
    obj = composite_objective(_report(), Lambdas())
    assert obj == 0.0


def test_objective_sums_all_penalised_terms():
    lam = Lambdas(kl=1.0, negation=2.0, sycophancy=3.0, agentic=4.0)
    obj = composite_objective(
        _report(refusal=0.1, kl=0.5, neg=0.8, follow=0.2, agentic=0.7), lam
    )
    expected = 0.1 + 1.0 * 0.5 + 2.0 * (1 - 0.8) + 3.0 * 0.2 + 4.0 * (1 - 0.7)
    assert abs(obj - expected) < 1e-9


def test_agentic_loss_is_invisible_without_lambda_agent():
    # Justification (commentaire de TASK 5) : sans λ_agent, l'optimiseur ne « voit » pas
    # l'effondrement agentique -> deux modèles très différents ont le même objectif.
    base = Lambdas(agentic=0.0)
    good = composite_objective(_report(agentic=0.95), base)
    collapsed = composite_objective(_report(agentic=0.10), base)
    assert good == collapsed  # l'objectif est aveugle à l'agentique
    with_lambda = Lambdas(agentic=1.0)
    assert composite_objective(_report(agentic=0.10), with_lambda) > composite_objective(
        _report(agentic=0.95), with_lambda
    )


def test_build_objective_calls_eval_fn_and_returns_composite():
    calls = []

    def eval_fn(layer, alpha):
        calls.append((layer, alpha))
        return _report(refusal=0.2, agentic=0.5)   # composite = 0.2 + (1-0.5) = 0.7

    objective = build_objective(eval_fn, candidate_layers=[10, 12, 14], lambdas=Lambdas())
    value = objective(_FakeTrial(layer=12, alpha=0.8))
    assert calls == [(12, 0.8)]                     # eval_fn reçoit bien (layer, alpha) du trial
    assert abs(value - (0.2 + 1.0 * (1 - 0.5))) < 1e-9


def test_build_objective_distinguishes_layers_via_eval_fn():
    # Une couche qui supprime mieux le refus doit donner un objectif plus bas.
    def eval_fn(layer, alpha):
        return _report(refusal=0.0 if layer == 14 else 0.9)

    objective = build_objective(eval_fn, [12, 14], Lambdas())
    assert objective(_FakeTrial(14, 1.0)) < objective(_FakeTrial(12, 1.0))


def test_build_objective_rejects_empty_candidates():
    with pytest.raises(ValueError):
        build_objective(lambda l, a: _report(), [], Lambdas())


def test_checkpoint_save_load_roundtrip_and_best(tmp_path):
    path = tmp_path / "trials.jsonl"
    save_trial(path, {"params": {"layer": 14}, "objective": 0.4})
    save_trial(path, {"params": {"layer": 12}, "objective": 0.2})
    trials = load_trials(path)
    assert len(trials) == 2
    assert best_trial(trials)["params"]["layer"] == 12  # plus petit objectif


def test_run_optuna_study_errors_clearly_when_optuna_missing():
    pytest.importorskip  # noqa
    try:
        import optuna  # noqa: F401
    except ImportError:
        with pytest.raises(RuntimeError, match="optuna"):
            run_optuna_study(objective=lambda t: 0.0, space={}, n_trials=1, checkpoint_path=None)
