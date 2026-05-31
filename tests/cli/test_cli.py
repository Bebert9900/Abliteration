"""Tests du parseur CLI : sous-commandes, --variant preserving, --preserve, diagnose, heal."""
import pytest

from src.cli import build_parser, parse_preserve


def test_parse_preserve_splits_comma_list():
    assert parse_preserve("negation,agentic") == ["negation", "agentic"]
    assert parse_preserve("harmless") == ["harmless"]
    assert parse_preserve(None) == []


def test_all_expected_subcommands_exist():
    parser = build_parser()
    sub = {a.dest: a for a in parser._subparsers._group_actions}  # noqa: SLF001
    # parse une commande de chaque type sans erreur
    for cmd in ["extract", "select", "apply", "abliterate", "optimize", "eval", "diagnose", "heal"]:
        ns = parser.parse_args([cmd, "some-model"])
        assert ns.command == cmd


def test_abliterate_accepts_preserving_variant_and_preserve_list():
    parser = build_parser()
    ns = parser.parse_args(
        ["abliterate", "meta-llama/Llama-3.1-8B-Instruct",
         "--variant", "preserving", "--preserve", "negation,agentic", "--out", "./out"]
    )
    assert ns.variant == "preserving"
    assert parse_preserve(ns.preserve) == ["negation", "agentic"]
    assert ns.out == "./out"


def test_optimize_exposes_all_lambda_flags():
    parser = build_parser()
    ns = parser.parse_args(
        ["optimize", "m", "--trials", "50",
         "--lambda-kl", "1.0", "--lambda-neg", "2.0", "--lambda-syco", "0.5", "--lambda-agent", "3.0"]
    )
    assert ns.trials == 50
    assert (ns.lambda_kl, ns.lambda_neg, ns.lambda_syco, ns.lambda_agent) == (1.0, 2.0, 0.5, 3.0)


def test_diagnose_does_not_require_output_dir():
    parser = build_parser()
    ns = parser.parse_args(["diagnose", "m"])
    assert ns.command == "diagnose"
    assert not hasattr(ns, "out") or ns.out is None
