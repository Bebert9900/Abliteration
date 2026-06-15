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


def required_arg_completeness(call: ToolCall, schema: dict) -> float:
    """Fraction des arguments *required* effectivement présents dans l'appel (crédit partiel)."""
    required = schema.get("parameters", {}).get("required", [])
    if not required:
        return 1.0
    return sum(1 for r in required if r in call.args) / len(required)


@dataclass(frozen=True)
class AgenticBreakdown:
    schema_validity: float          # fraction d'appels bien formés (nom + tous required présents)
    arg_completeness: float         # fraction moyenne des args required présents (crédit partiel)
    score: float                    # moyenne des signaux DISPONIBLES (cf. ci-dessous)
    n: int
    arg_accuracy: float | None = None   # exactitude des valeurs, si `expected_args` fourni (sinon None)
    n_with_expected: int = 0            # nb d'items disposant de valeurs attendues


def evaluate_agentic_outputs(outputs: list[str], schemas: list[dict]) -> AgenticBreakdown:
    """Évalue des sorties brutes contre leurs schémas d'outil (listes alignées).

    Signaux mesurés :
      - `schema_validity` : fraction d'appels bien formés (nom correct + required présents).
      - `arg_completeness` : fraction moyenne des args required présents (crédit partiel).
      - `arg_accuracy` : exactitude des VALEURS d'arguments, calculée UNIQUEMENT pour les items
        dont le schéma porte `expected_args` (clé optionnelle). None si aucun item enrichi.

    Le `score` est la moyenne des signaux effectivement disponibles : il intègre `arg_accuracy`
    dès qu'au moins un item fournit `expected_args`, sinon il se limite aux deux premiers signaux.
    On n'invente jamais une métrique absente (pas de répétition artificielle).
    """
    n = len(outputs)
    if n == 0:
        return AgenticBreakdown(0.0, 0.0, 0.0, 0)
    valid = 0
    comp = 0.0
    acc_sum = 0.0
    n_expected = 0
    for out, schema in zip(outputs, schemas):
        schema = schema or {}
        call = parse_tool_call(out)
        expected = schema.get("expected_args")
        if expected:                       # item enrichi : compte toujours (0 si l'appel échoue)
            n_expected += 1
            acc_sum += arg_accuracy(call, expected) if call is not None else 0.0
        if call is None or not schema:
            continue
        if schema_valid(call, schema):
            valid += 1
        comp += required_arg_completeness(call, schema)
    schema_validity = valid / n
    arg_completeness = comp / n
    signals = [schema_validity, arg_completeness]
    acc = None
    if n_expected:
        acc = acc_sum / n_expected
        signals.append(acc)
    return AgenticBreakdown(schema_validity, arg_completeness, sum(signals) / len(signals), n,
                            arg_accuracy=acc, n_with_expected=n_expected)


def multi_step_success(predicted: list[ToolCall], expected: list[ToolCall]) -> float:
    """1.0 si la séquence d'appels correspond exactement, sinon fraction de bonnes étapes."""
    if not expected:
        return 1.0
    ok = sum(1 for p, e in zip(predicted, expected) if p == e)
    return ok / len(expected)


def agentic_score(schema_validity: float, arg_accuracy: float, multi_step_success: float) -> float:
    """Agrégat normalisé [0,1] des trois métriques agentiques (moyenne)."""
    return (schema_validity + arg_accuracy + multi_step_success) / 3.0
