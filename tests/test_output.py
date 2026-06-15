"""Tests du contrat de sortie machine : enveloppe JSON, erreurs, introspection schema."""
import io
import json
from types import SimpleNamespace

from abliteration.output import (
    SCHEMA_VERSION,
    emit_error,
    emit_result,
    parser_schema,
)


def _capture(fn, *a, **k):
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = fn(*a, **k)
    return rc, buf.getvalue()


def test_emit_result_json_envelope():
    rc, out = _capture(emit_result, SimpleNamespace(json=True), "eval", {"refusal_rate": 0.1})
    env = json.loads(out)
    assert rc == 0
    assert env == {"schema_version": SCHEMA_VERSION, "status": "ok", "command": "eval",
                   "data": {"refusal_rate": 0.1}, "error": None}


def test_emit_result_human_uses_renderer():
    rc, out = _capture(emit_result, SimpleNamespace(json=False), "select",
                       {"selected_layer": 14}, human=lambda d: print(d["selected_layer"]))
    assert rc == 0
    assert out.strip() == "14"           # rendu humain, pas d'enveloppe
    assert "status" not in out


def test_emit_result_human_default_is_plain_json():
    _, out = _capture(emit_result, SimpleNamespace(json=False), "x", {"a": 1})
    assert json.loads(out) == {"a": 1}   # pas d'enveloppe en mode humain par défaut


def test_emit_error_envelope_and_exit_code():
    rc, out = _capture(emit_error, SimpleNamespace(json=True), "extract", FileNotFoundError("nope"))
    env = json.loads(out)
    assert rc == 1
    assert env["status"] == "error" and env["data"] is None
    assert env["error"] == {"type": "FileNotFoundError", "message": "nope"}


def test_emit_error_human_keeps_stdout_clean():
    rc, out = _capture(emit_error, SimpleNamespace(json=False), "extract", ValueError("boom"))
    assert rc == 1
    assert out == ""                     # stdout vide en mode humain (message sur stderr)


def test_parser_schema_lists_all_commands_with_json_and_output():
    from abliteration.cli import build_parser
    schema = parser_schema(build_parser())
    cmds = schema["commands"]
    for expected in ["extract", "select", "apply", "abliterate", "optimize", "eval",
                     "diagnose", "analyze-circuit", "heal", "schema"]:
        assert expected in cmds
        assert any(a["name"] == "json" for a in cmds[expected]["arguments"])
    # les formes de sortie sont déclarées pour les commandes principales
    assert cmds["eval"]["output"] is not None
    assert "refusal_rate" in cmds["eval"]["output"]


def test_parser_schema_captures_arg_metadata():
    from abliteration.cli import build_parser
    cmds = parser_schema(build_parser())["commands"]
    variant = next(a for a in cmds["abliterate"]["arguments"] if a["name"] == "variant")
    assert variant["default"] == "norm_preserving_biprojected"
    assert "norm_preserving_biprojected" in variant["choices"]
    model = next(a for a in cmds["abliterate"]["arguments"] if a["name"] == "model")
    assert model["positional"] is True and model["required"] is True
