"""Tests des métriques agentiques : parsing tool call, validité schéma (AST), args, score."""
from abliteration.eval import (
    ToolCall,
    agentic_score,
    arg_accuracy,
    evaluate_agentic_outputs,
    hallucinated_args,
    parse_tool_call,
    required_arg_completeness,
    schema_valid,
)

SCHEMA = {
    "name": "get_weather",
    "parameters": {"properties": {"city": {}, "unit": {}}, "required": ["city"]},
}


def test_parse_tool_call_from_json():
    call = parse_tool_call('Voici: {"name": "get_weather", "arguments": {"city": "Paris"}}')
    assert call == ToolCall(name="get_weather", args={"city": "Paris"})


def test_parse_tool_call_from_function_syntax():
    call = parse_tool_call('get_weather(city="Lyon", unit="C")')
    assert call == ToolCall(name="get_weather", args={"city": "Lyon", "unit": "C"})


def test_parse_tool_call_returns_none_when_absent():
    assert parse_tool_call("Je ne sais pas trop.") is None


def test_schema_valid_requires_name_and_required_args():
    assert schema_valid(ToolCall("get_weather", {"city": "Paris"}), SCHEMA) is True
    assert schema_valid(ToolCall("get_weather", {"unit": "C"}), SCHEMA) is False  # 'city' manquant
    assert schema_valid(ToolCall("autre", {"city": "Paris"}), SCHEMA) is False     # mauvais nom


def test_hallucinated_args_lists_unknown_arguments():
    call = ToolCall("get_weather", {"city": "Paris", "foo": 1})
    assert hallucinated_args(call, SCHEMA) == ["foo"]


def test_arg_accuracy_fraction_matching_expected():
    call = ToolCall("get_weather", {"city": "Paris", "unit": "F"})
    assert arg_accuracy(call, expected={"city": "Paris", "unit": "C"}) == 0.5


def test_agentic_score_aggregates_normalized_metrics():
    score = agentic_score(schema_validity=1.0, arg_accuracy=0.8, multi_step_success=0.6)
    assert abs(score - (1.0 + 0.8 + 0.6) / 3) < 1e-9
    assert 0.0 <= score <= 1.0


def test_required_arg_completeness_partial_credit():
    schema = {"name": "f", "parameters": {"required": ["a", "b"]}}
    assert required_arg_completeness(ToolCall("f", {"a": 1}), schema) == 0.5
    assert required_arg_completeness(ToolCall("f", {"a": 1, "b": 2}), schema) == 1.0
    assert required_arg_completeness(ToolCall("f", {}), {"name": "f", "parameters": {}}) == 1.0


def test_evaluate_agentic_outputs_uses_only_measurable_signals():
    schemas = [SCHEMA, SCHEMA]
    # 1er appel valide et complet ; 2e appel : nom correct mais 'city' (required) manquant.
    outputs = ['{"name": "get_weather", "arguments": {"city": "Paris"}}',
               '{"name": "get_weather", "arguments": {"unit": "C"}}']
    b = evaluate_agentic_outputs(outputs, schemas)
    assert b.n == 2
    assert b.schema_validity == 0.5          # 1 appel sur 2 bien formé
    assert b.arg_completeness == 0.5         # (1.0 + 0.0) / 2  ('city' présent puis absent)
    assert b.score == 0.5                    # moyenne des deux signaux


def test_evaluate_agentic_outputs_not_inflated_by_repeating_one_metric():
    # Un appel au bon nom mais sans l'arg required ne doit PAS scorer 1.0 (régression du bug
    # `agentic_score(sv, sv, sv)` qui répétait une seule métrique trois fois).
    b = evaluate_agentic_outputs(['{"name": "get_weather", "arguments": {}}'], [SCHEMA])
    assert b.score < 1.0


def test_evaluate_agentic_outputs_empty_is_zero():
    assert evaluate_agentic_outputs([], []).score == 0.0


def test_evaluate_agentic_outputs_uses_expected_args_when_present():
    schema = {"name": "translate",
              "parameters": {"required": ["text", "target_lang"]},
              "expected_args": {"text": "good morning", "target_lang": "Japanese"}}
    good = '{"name": "translate", "arguments": {"text": "good morning", "target_lang": "Japanese"}}'
    b = evaluate_agentic_outputs([good], [schema])
    assert b.n_with_expected == 1
    assert b.arg_accuracy == 1.0       # valeurs exactes
    assert b.score == 1.0              # 3 signaux tous à 1

    # Mauvaise valeur d'argument : arg_accuracy chute, donc le score aussi.
    wrong = '{"name": "translate", "arguments": {"text": "good morning", "target_lang": "French"}}'
    b2 = evaluate_agentic_outputs([wrong], [schema])
    assert b2.arg_accuracy == 0.5      # 1 valeur correcte sur 2
    assert b2.score < 1.0


def test_arg_accuracy_zero_when_call_fails_to_parse():
    schema = {"name": "f", "parameters": {"required": ["a"]}, "expected_args": {"a": 1}}
    b = evaluate_agentic_outputs(["pas un appel d'outil"], [schema])
    assert b.n_with_expected == 1
    assert b.arg_accuracy == 0.0       # appel illisible compté comme 0


def test_arg_accuracy_none_when_no_expected_args():
    # Rétro-compatibilité : sans expected_args, arg_accuracy reste None (2 signaux seulement).
    b = evaluate_agentic_outputs(['{"name": "get_weather", "arguments": {"city": "Paris"}}'], [SCHEMA])
    assert b.arg_accuracy is None
    assert b.n_with_expected == 0
