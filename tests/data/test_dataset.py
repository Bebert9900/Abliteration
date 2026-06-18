"""Tests du module de données : découpage holdout, chargement, conteneur 4 classes."""
import json

import pytest

from meridian.data import FourClassData, Prompt, PromptClass, load_prompts, split_holdout


def _prompts(n, cls=PromptClass.HARMFUL):
    return [Prompt(text=f"p{i}", cls=cls) for i in range(n)]


def test_split_holdout_is_disjoint_and_covers_everything():
    prompts = _prompts(10)
    train, holdout = split_holdout(prompts, holdout_fraction=0.3, seed=0)

    train_texts = {p.text for p in train}
    holdout_texts = {p.text for p in holdout}
    assert train_texts.isdisjoint(holdout_texts)
    assert train_texts | holdout_texts == {p.text for p in prompts}


def test_split_holdout_respects_fraction():
    train, holdout = split_holdout(_prompts(10), holdout_fraction=0.3, seed=0)
    assert len(holdout) == 3
    assert len(train) == 7


def test_split_holdout_is_deterministic_for_same_seed():
    a_train, a_hold = split_holdout(_prompts(20), holdout_fraction=0.25, seed=42)
    b_train, b_hold = split_holdout(_prompts(20), holdout_fraction=0.25, seed=42)
    assert [p.text for p in a_hold] == [p.text for p in b_hold]
    assert [p.text for p in a_train] == [p.text for p in b_train]


def test_split_holdout_differs_for_different_seed():
    _, hold1 = split_holdout(_prompts(50), holdout_fraction=0.2, seed=1)
    _, hold2 = split_holdout(_prompts(50), holdout_fraction=0.2, seed=2)
    assert [p.text for p in hold1] != [p.text for p in hold2]


def test_load_prompts_jsonl_accepts_text_or_prompt_key(tmp_path):
    f = tmp_path / "harmless.jsonl"
    f.write_text(
        json.dumps({"text": "explique la photosynthèse"}) + "\n"
        + json.dumps({"prompt": "résume ce texte"}) + "\n"
    )
    prompts = load_prompts(f, PromptClass.HARMLESS)
    assert [p.text for p in prompts] == ["explique la photosynthèse", "résume ce texte"]
    assert all(p.cls is PromptClass.HARMLESS for p in prompts)


def test_load_prompts_keeps_extra_fields_as_meta(tmp_path):
    f = tmp_path / "agentic.jsonl"
    f.write_text(json.dumps({
        "text": "quel temps fait-il à Paris ?",
        "expected_tool": "get_weather",
        "expected_args": {"city": "Paris"},
    }) + "\n")
    (prompt,) = load_prompts(f, PromptClass.AGENTIC)
    assert prompt.text == "quel temps fait-il à Paris ?"
    assert prompt.meta["expected_tool"] == "get_weather"
    assert prompt.meta["expected_args"] == {"city": "Paris"}


def test_load_prompts_skips_blank_lines(tmp_path):
    f = tmp_path / "h.jsonl"
    f.write_text(json.dumps({"text": "a"}) + "\n\n   \n" + json.dumps({"text": "b"}) + "\n")
    assert [p.text for p in load_prompts(f, PromptClass.HARMFUL)] == ["a", "b"]


def _write_jsonl(path, texts):
    path.write_text("".join(json.dumps({"text": t}) + "\n" for t in texts))
    return path


def test_four_class_data_loads_and_splits_each_class(tmp_path):
    paths = {
        PromptClass.HARMFUL: _write_jsonl(tmp_path / "harm.jsonl", [f"x{i}" for i in range(10)]),
        PromptClass.HARMLESS: _write_jsonl(tmp_path / "safe.jsonl", [f"s{i}" for i in range(10)]),
        PromptClass.LEGITIMATE_NEGATION: _write_jsonl(tmp_path / "neg.jsonl", [f"n{i}" for i in range(10)]),
        PromptClass.AGENTIC: _write_jsonl(tmp_path / "ag.jsonl", [f"a{i}" for i in range(10)]),
    }
    data = FourClassData.load(paths, holdout_fraction=0.2, seed=0)

    for cls in PromptClass:
        train, holdout = data.train(cls), data.holdout(cls)
        assert len(train) == 8 and len(holdout) == 2
        assert {p.text for p in train}.isdisjoint({p.text for p in holdout})
        assert all(p.cls is cls for p in train + holdout)


def test_four_class_data_requires_all_four_classes(tmp_path):
    paths = {PromptClass.HARMFUL: _write_jsonl(tmp_path / "h.jsonl", ["a", "b"])}
    with pytest.raises(ValueError, match="classe"):
        FourClassData.load(paths, holdout_fraction=0.2, seed=0)
