"""Tests CLI de l'analyse circuitielle : parseur de `analyze-circuit` et flags `diagnose`.

Niveau parseur (comme tests/cli/test_cli.py) : on vérifie l'interface, pas l'exécution sur un
vrai modèle (faite à la main en fin de phase). On vérifie aussi que analyze-circuit n'expose
AUCUNE option de modification de poids (Phase 1 = lecture seule).
"""
from abliteration.cli import build_parser


def test_analyze_circuit_subcommand_exists():
    ns = build_parser().parse_args(["analyze-circuit", "some-model"])
    assert ns.command == "analyze-circuit"
    assert ns.func.__name__ == "cmd_analyze_circuit"


def test_analyze_circuit_exposes_phase1_flags():
    ns = build_parser().parse_args([
        "analyze-circuit", "Qwen/Qwen3-0.6B",
        "--pairs", "24", "--top-k", "30", "--threshold", "0.6",
        "--n-boot", "100", "--layer", "12", "--backend", "torch", "--out", "rep.json",
    ])
    assert ns.pairs == 24
    assert ns.top_k == 30
    assert ns.threshold == 0.6
    assert ns.n_boot == 100
    assert ns.layer == 12
    assert ns.backend == "torch"
    assert ns.out == "rep.json"


def test_analyze_circuit_has_no_weight_modification_flags():
    """Phase 1 : aucune option d'ablation/écriture de poids ne doit exister sur la commande."""
    ns = build_parser().parse_args(["analyze-circuit", "m"])
    forbidden = ["variant", "preserve", "norm_preserve"]
    for attr in forbidden:
        assert not hasattr(ns, attr), f"analyze-circuit ne doit pas exposer --{attr} en Phase 1"


def test_backend_choice_is_constrained():
    import pytest
    with pytest.raises(SystemExit):
        build_parser().parse_args(["analyze-circuit", "m", "--backend", "transformerlens"])


def test_diagnose_accepts_circuit_flag():
    ns = build_parser().parse_args(["diagnose", "m", "--circuit", "--layer", "10"])
    assert ns.command == "diagnose"
    assert ns.circuit is True
    assert ns.layer == 10


def test_diagnose_circuit_flag_defaults_false():
    ns = build_parser().parse_args(["diagnose", "m"])
    assert ns.circuit is False
