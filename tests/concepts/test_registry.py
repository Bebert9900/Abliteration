"""Tests du registre de concepts et du chargement ad hoc."""
import pytest

from meridian.concepts import (
    available_concepts,
    load_concept,
    load_concept_from_files,
)


def test_available_concepts_lists_builtins():
    assert available_concepts() == ["agentic", "negation", "refusal"]


def test_load_builtin_refusal_from_data_dir():
    c = load_concept("refusal", data_dir="data")     # fichiers réels du dépôt
    assert c.name == "refusal"
    assert len(c.positive) > 0 and len(c.negative) > 0
    assert c.description


def test_load_unknown_concept_raises():
    with pytest.raises(KeyError, match="inconnu"):
        load_concept("inexistant")


def test_load_concept_from_files(tmp_path):
    pos = tmp_path / "pos.jsonl"
    neg = tmp_path / "neg.jsonl"
    pos.write_text('{"text": "p1"}\n{"prompt": "p2"}\n', encoding="utf-8")
    neg.write_text('{"text": "n1"}\n', encoding="utf-8")
    c = load_concept_from_files("veracity", pos, neg)
    assert c.positive == ["p1", "p2"] and c.negative == ["n1"]
    assert c.name == "veracity"


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_concept_from_files("x", tmp_path / "absent.jsonl", tmp_path / "absent2.jsonl")
