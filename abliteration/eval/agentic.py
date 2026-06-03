"""Métriques agentiques : parsing de tool call, validité de schéma (style AST/BFCL), args.

Cœur custom de l'éval agentique (les harnais externes — BFCL, IFEval, tau-bench — sont branchés
via `benchmarks.py`). On vérifie qu'un appel d'outil prédit est syntaxiquement valide, respecte
le schéma, et que ses arguments sont corrects ; on détecte les arguments hallucinés.
"""
from __future__ import annotations

import ast
import json
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict


def _from_json(text: str) -> ToolCall | None:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(obj, dict) or "name" not in obj:
        return None
    args = obj.get("arguments", obj.get("args", {}))
    return ToolCall(name=obj["name"], args=dict(args) if isinstance(args, dict) else {})


def _from_call_syntax(text: str) -> ToolCall | None:
    match = re.search(r"[A-Za-z_]\w*\s*\([^()]*\)", text)
    if not match:
        return None
    try:
        node = ast.parse(match.group(0), mode="eval").body
    except SyntaxError:
        return None
    if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
        return None
    args = {}
    for kw in node.keywords:
        try:
            args[kw.arg] = ast.literal_eval(kw.value)
        except (ValueError, SyntaxError):
            return None
    return ToolCall(name=node.func.id, args=args)


def parse_tool_call(text: str) -> ToolCall | None:
    """Extrait un appel d'outil d'un texte (JSON ou syntaxe `func(arg=val)`), sinon None."""
    return _from_json(text or "") or _from_call_syntax(text or "")


def schema_valid(call: ToolCall, schema: dict) -> bool:
    if call.name != schema.get("name"):
        return False
    required = schema.get("parameters", {}).get("required", [])
    return all(r in call.args for r in required)


def hallucinated_args(call: ToolCall, schema: dict) -> list[str]:
    """Arguments présents dans l'appel mais absents du schéma (hallucinations)."""
    known = set(schema.get("parameters", {}).get("properties", {}))
    return sorted(a for a in call.args if a not in known)


def arg_accuracy(call: ToolCall, expected: dict) -> float:
    if not expected:
        return 1.0
    matches = sum(1 for k, v in expected.items() if call.args.get(k) == v)
    return matches / len(expected)


def multi_step_success(predicted: list[ToolCall], expected: list[ToolCall]) -> float:
    """1.0 si la séquence d'appels correspond exactement, sinon fraction de bonnes étapes."""
    if not expected:
        return 1.0
    ok = sum(1 for p, e in zip(predicted, expected) if p == e)
    return ok / len(expected)


def agentic_score(schema_validity: float, arg_accuracy: float, multi_step_success: float) -> float:
    """Agrégat normalisé [0,1] des trois métriques agentiques (moyenne)."""
    return (schema_validity + arg_accuracy + multi_step_success) / 3.0
