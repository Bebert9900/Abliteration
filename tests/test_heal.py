"""Tests de la préparation de données du healing (LoRA SFT) — sans GPU ni dépendances lourdes."""

import pytest

from meridian.heal import HealConfig, format_example, heal, iter_training_examples, load_traces


def test_load_traces_accepts_both_schemas(tmp_path):
    p = tmp_path / "traces.jsonl"
    p.write_text(
        '{"prompt": "P", "completion": "C"}\n'
        '\n'  # ligne vide ignorée
        '{"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}\n',
        encoding="utf-8",
    )
    traces = load_traces(p)
    assert len(traces) == 2


def test_load_traces_limit_and_missing_file(tmp_path):
    p = tmp_path / "t.jsonl"
    p.write_text("\n".join('{"prompt": "p", "completion": "c"}' for _ in range(5)), encoding="utf-8")
    assert len(load_traces(p, limit=3)) == 3
    with pytest.raises(FileNotFoundError):
        load_traces(tmp_path / "absent.jsonl")


def test_load_traces_rejects_invalid_schema(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text('{"foo": "bar"}\n', encoding="utf-8")
    with pytest.raises(ValueError):
        load_traces(p)


def test_format_example_prompt_completion():
    assert format_example({"prompt": "P", "completion": "C"}) == ("P", "C")


def test_format_example_messages_targets_last_turn():
    trace = {"messages": [
        {"role": "user", "content": "appelle l'outil"},
        {"role": "assistant", "content": '{"name": "f", "arguments": {}}'},
    ]}
    prompt, completion = format_example(trace)   # sans tokenizer -> concat simple
    assert completion == '{"name": "f", "arguments": {}}'
    assert "appelle l'outil" in prompt
    assert "assistant" not in prompt              # le dernier tour n'est pas dans le prompt


def test_format_example_uses_chat_template_when_available():
    class FakeTok:
        def apply_chat_template(self, messages, tokenize, add_generation_prompt):
            return "TEMPLATED:" + "|".join(m["content"] for m in messages)

    trace = {"messages": [{"role": "user", "content": "u"}, {"role": "assistant", "content": "a"}]}
    prompt, completion = format_example(trace, FakeTok())
    assert prompt.startswith("TEMPLATED:u")
    assert completion == "a"


def test_iter_training_examples_yields_pairs():
    traces = [{"prompt": "p1", "completion": "c1"}, {"prompt": "p2", "completion": "c2"}]
    assert list(iter_training_examples(traces)) == [("p1", "c1"), ("p2", "c2")]


def test_heal_errors_clearly_without_peft(tmp_path, monkeypatch):
    # Simule l'absence de peft : heal() doit lever une RuntimeError explicite, pas un ImportError brut.
    import builtins
    real_import = builtins.__import__

    def fake_import(name, *a, **k):
        if name == "peft":
            raise ImportError("No module named 'peft'", name="peft")
        return real_import(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    cfg = HealConfig(model_dir="m", traces_path=str(tmp_path / "t.jsonl"))
    with pytest.raises(RuntimeError, match="peft"):
        heal(cfg)
