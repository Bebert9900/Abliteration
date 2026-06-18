"""Tests du parseur CLI : sous-commandes, --variant preserving, --preserve, diagnose, heal."""
import pytest

from meridian.cli import build_parser, parse_preserve


def test_parse_preserve_splits_comma_list():
    assert parse_preserve("negation,agentic") == ["negation", "agentic"]
    assert parse_preserve("harmless") == ["harmless"]
    assert parse_preserve(None) == []


def test_all_expected_subcommands_exist():
    parser = build_parser()
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


def test_optimize_exposes_alpha_and_cache_flags():
    parser = build_parser()
    ns = parser.parse_args(["optimize", "m", "--alpha-low", "0.3", "--alpha-high", "0.9",
                            "--eval-limit", "8", "--apply-best", "--no-cache"])
    assert (ns.alpha_low, ns.alpha_high, ns.eval_limit) == (0.3, 0.9, 8)
    assert ns.apply_best is True and ns.no_cache is True


def test_default_variant_is_norm_preserving():
    parser = build_parser()
    ns = parser.parse_args(["abliterate", "m"])
    assert ns.variant == "norm_preserving_biprojected"   # variante de production par défaut


def _run_main(argv):
    """Exécute main(argv), capture stdout, renvoie (code, stdout)."""
    import contextlib
    import io

    from meridian.cli import main
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def test_schema_command_emits_valid_json_envelope():
    import json
    rc, out = _run_main(["schema", "--json"])
    env = json.loads(out)
    assert rc == 0
    assert env["status"] == "ok" and env["command"] == "schema"
    assert "extract" in env["data"]["commands"]


def test_json_flag_available_on_every_subcommand():
    parser = build_parser()
    for cmd in ["extract", "select", "apply", "abliterate", "optimize", "eval",
                "diagnose", "analyze-circuit", "heal", "schema"]:
        ns = parser.parse_args([cmd, "m"] if cmd not in ("schema",) else [cmd])
        # le flag --json doit être accepté (défaut False)
        assert hasattr(ns, "json") and ns.json is False


def test_main_error_path_returns_structured_envelope():
    # extract sur un dossier de données absent -> erreur attrapée -> enveloppe + code 1.
    import json
    rc, out = _run_main(["extract", "fake-model", "--data-dir", "/does/not/exist", "--json"])
    env = json.loads(out)
    assert rc == 1
    assert env["status"] == "error" and env["data"] is None
    assert env["error"]["type"]   # un type d'exception est renseigné


def test_usage_error_exits_with_code_2():
    with pytest.raises(SystemExit) as e:
        build_parser().parse_args(["pas-une-commande"])
    assert e.value.code == 2   # argparse : erreur d'usage


def test_concept_commands_registered_with_concept_choices():
    parser = build_parser()
    ns = parser.parse_args(["concept-direction", "m", "--concept", "refusal"])
    assert ns.command == "concept-direction" and ns.concept == "refusal"
    ns = parser.parse_args(["concept-separability", "m", "--concepts", "refusal,negation"])
    assert ns.command == "concept-separability"
    # --concept expose les choix du registre (découvrabilité)
    with pytest.raises(SystemExit):
        parser.parse_args(["concept-direction", "m", "--concept", "inexistant"])


def test_schema_lists_concept_commands_with_output():
    import json
    rc, out = _run_main(["schema", "--json"])
    cmds = json.loads(out)["data"]["commands"]
    assert "concept-direction" in cmds and "concept-separability" in cmds
    assert "matrix" in cmds["concept-separability"]["output"]
    concept_arg = next(a for a in cmds["concept-direction"]["arguments"] if a["name"] == "concept")
    assert "refusal" in concept_arg["choices"]   # concepts du registre découvrables


def test_analyze_circuit_accepts_concept_and_keeps_refusal_default():
    parser = build_parser()
    ns = parser.parse_args(["analyze-circuit", "m", "--concept", "negation"])
    assert ns.concept == "negation"
    ns = parser.parse_args(["analyze-circuit", "m"])      # rétrocompat : pas de concept = refus
    assert ns.concept is None
    with pytest.raises(SystemExit):
        parser.parse_args(["analyze-circuit", "m", "--concept", "inexistant"])


def test_build_concept_pairs_returns_aligned_tuples():
    import torch

    from meridian.cli import _build_concept_pairs

    class FakeFormatter:
        def tokenize(self, texts):
            return {"input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
                    "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]])}

    pairs = _build_concept_pairs(FakeFormatter(), ["p1", "p2"], ["n1", "n2"], n=2)
    assert len(pairs) == 2
    clean_ids, corr_ids, clean_mask, corr_mask = pairs[0]
    assert clean_ids.shape == (1, 3) and corr_ids.shape == (1, 3)


def test_schema_lists_concept_arg_on_analyze_circuit():
    import json
    rc, out = _run_main(["schema", "--json"])
    cmds = json.loads(out)["data"]["commands"]
    arg = next(a for a in cmds["analyze-circuit"]["arguments"] if a["name"] == "concept")
    assert "negation" in arg["choices"]


def test_concept_steer_parser_and_schema():
    import json
    parser = build_parser()
    ns = parser.parse_args(["concept-steer", "m", "--concept", "refusal", "--alpha", "5.0",
                            "--preserve", "negation,agentic"])
    assert ns.command == "concept-steer" and ns.alpha == 5.0 and ns.concept == "refusal"
    rc, out = _run_main(["schema", "--json"])
    cmds = json.loads(out)["data"]["commands"]
    assert "concept-steer" in cmds and "concept-probe" in cmds
    assert "comparisons" in cmds["concept-steer"]["output"]


def test_concept_direction_requires_a_concept_source():
    from types import SimpleNamespace

    from meridian.cli import _resolve_concept
    ns = SimpleNamespace(concept=None, pos=None, neg=None, name=None, data_dir="data")
    with pytest.raises(ValueError):
        _resolve_concept(ns)


def test_dump_run_config_writes_hashes_and_params(tmp_path):
    import json
    from types import SimpleNamespace

    from meridian.cli import _dump_run_config

    data_dir = tmp_path / "data"
    data_dir.mkdir()
    for name in ("harmful", "harmless", "legitimate_negation", "agentic"):
        (data_dir / f"{name}.txt").write_text('{"text": "x"}\n', encoding="utf-8")
    out = tmp_path / "out"
    out.mkdir()
    ns = SimpleNamespace(model="m", dtype="bfloat16", variant="preserving", preserve="negation",
                         layers=None, holdout=0.2, seed=0, batch_size=8, norm_preserve=False,
                         data_dir=str(data_dir))
    cfg = _dump_run_config(ns, out, extra={"selected_layer": 14})
    saved = json.loads((out / "run_config.json").read_text())
    assert saved["model"] == "m"
    assert saved["selected_layer"] == 14
    assert set(saved["data_hashes"]) == {"harmful", "harmless", "legitimate_negation", "agentic"}
    assert cfg == saved
