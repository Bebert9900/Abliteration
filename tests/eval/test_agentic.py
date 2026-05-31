"""Tests des métriques agentiques : parsing tool call, validité schéma (AST), args, score."""
from src.eval import (
    ToolCall,
    agentic_score,
    arg_accuracy,
    hallucinated_args,
    parse_tool_call,
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
