"""Tests du rapport : structure JSON, séparation corrélationnel/causal, rendu texte, résumé."""
import json

import torch

from abliteration.circuits.backend import Component, ComponentKind, TorchHookBackend
from abliteration.circuits.localize import localize
from abliteration.circuits.patching import RefusalMetric
from abliteration.circuits.report import CircuitReport
from toymodel import (
    ControllableModel,
    controllable_refusal_dir,
    harmful_ids,
    harmless_ids,
)

CAUSAL = Component(ComponentKind.ATTN_HEAD, 0, 0)


def _report():
    be = TorchHookBackend(ControllableModel())
    metric = RefusalMetric(refusal_dir=controllable_refusal_dir())
    pairs = [(harmful_ids(), harmless_ids(), None, None) for _ in range(3)]
    loc = localize(be, pairs, metric, controllable_refusal_dir(), threshold=0.5, n_boot=30)
    return CircuitReport(model_name="ControllableToy", localization=loc, n_pairs=len(pairs))


def test_to_dict_has_core_and_validation():
    d = _report().to_dict()
    assert d["phase"] == 1
    assert d["core_size"] == 1
    assert d["core_circuit"][0]["component"] == "L0H0"
    assert d["core_circuit"][0]["causally_validated"] is True
    val = d["validation"]
    assert val["bootstrap_stable"] is True
    assert abs(val["faithfulness"] - 1.0) < 1e-6


def test_json_roundtrip(tmp_path):
    rep = _report()
    p = tmp_path / "circuit.json"
    s = rep.to_json(path=p)
    loaded = json.loads(p.read_text())
    assert loaded == json.loads(s)
    assert loaded["model"] == "ControllableToy"


def test_attribution_graph_edges_point_to_refusal_logit():
    d = _report().to_dict()
    edges = d["attribution_graph"]
    assert any(e["source"] == "L0H0" and e["target"] == "refusal_logit" for e in edges)


def test_text_render_marks_causal_core_and_caveats():
    txt = _report().to_text()
    assert "CIRCUIT CORE (validé causalement)" in txt
    assert "L0H0" in txt
    assert "AVERTISSEMENTS" in txt
    assert "CORRÉLATIONNELLE" in txt


def test_short_summary_for_diagnose():
    s = _report().short_summary()
    assert "circuit refus" in s
    assert "L0H0" in s
    assert "Jaccard" in s


def test_report_with_empty_core_is_honest():
    """Si rien n'est validé causalement (seuil impossible), le rapport le dit explicitement."""
    be = TorchHookBackend(ControllableModel())
    metric = RefusalMetric(refusal_dir=controllable_refusal_dir())
    pairs = [(harmful_ids(), harmless_ids(), None, None)]
    loc = localize(be, pairs, metric, controllable_refusal_dir(), threshold=1.5, n_boot=10)
    rep = CircuitReport("ToyNoCore", loc, n_pairs=1)
    d = rep.to_dict()
    assert d["core_size"] == 0
    assert "aucun composant validé" in rep.to_text()
    assert "aucun composant validé" in rep.short_summary()
