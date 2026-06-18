"""Contrat de sortie machine de la CLI (mode `--json`) : enveloppe stable + codes de sortie.

Convention (héritée de gh/kubectl/aws) : avec `--json`, chaque commande émet sur stdout UNE
enveloppe versionnée `{schema_version, status, command, data, error}` ; les logs vont sur stderr.
Sans `--json`, on rend la sortie humaine (renderer fourni par le handler, sinon JSON indenté).
"""
from __future__ import annotations

import json
import sys
from typing import Callable

SCHEMA_VERSION = "1"


# Forme des données (`data`) émises par chaque commande en cas de succès. Registre déclaratif
# (les formes de sortie ne sont pas introspectables depuis argparse) consommé par `schema`.
COMMAND_OUTPUT: dict[str, dict] = {
    "extract": {"directions_path": "str — chemin du fichier de directions écrit"},
    "select": {"selected_layer": "int — couche de refus retenue", "scores": "dict[int,float] — refus par couche"},
    "apply": {"out_dir": "str", "selected_layer": "int"},
    "abliterate": {"refusal_rate": "float", "kl": "float", "negation_retention": "float",
                   "follow_rate": "float", "empty_rate": "float", "agentic_score": "float",
                   "degeneracy_rate": "float", "selected_layer": "int", "out_dir": "str"},
    "optimize": {"params": "dict — meilleurs (layer, alpha)", "objective": "float",
                 "resumed": "int", "out_dir": "str|null"},
    "eval": {"refusal_rate": "float", "kl": "float", "negation_retention": "float",
             "follow_rate": "float", "empty_rate": "float", "agentic_score": "float",
             "degeneracy_rate": "float", "benchmarks": "dict|null", "kl_map": "dict|null"},
    "diagnose": {"layers": "list[{layer,cos_refusal_negation,cos_refusal_agentic}]",
                 "warnings": "list[str]", "circuit_summary": "str|null"},
    "analyze-circuit": {"summary": "str", "report": "dict — métriques circuitielles"},
    "heal": {"out_dir": "str — dossier de l'adaptateur LoRA"},
    "concept-direction": {"name": "str", "layers": "int", "direction_path": "str|null",
                          "norms_per_layer": "list[float]"},
    "concept-probe": {"name": "str", "accuracy_per_layer": "list[float] — décodabilité par couche",
                      "best_layer": "int — couche la plus décodable"},
    "concept-separability": {"concepts": "list[str]", "matrix": "list[list[float]] — cosinus N×N",
                             "layer": "int|null", "warnings": "list[str]"},
    "concept-steer": {"name": "str", "alpha": "float", "layer": "int", "preserve": "list[str]",
                      "comparisons": "list[{prompt, baseline, steered}]", "n": "int"},
    "atlas-build": {"atlas_path": "str", "n_subjects": "int", "k": "int — directions latentes",
                    "explained_variance_top": "list[float] — variance du 1er latent par couche",
                    "subject_to_latent": "dict — latent le plus proche de chaque sujet",
                    "separability_warnings": "list[str] — sujets géométriquement intriqués"},
    "atlas-identify": {"query": "str — source de la direction interrogée",
                       "matches": "list[{name, cosine}] — sujets les plus proches (|cos|)"},
    "atlas-monitor": {"checkpoints": "list[str]", "ref": "str — checkpoint de référence",
                      "series": "list[{checkpoint, subjects, latent_subspace_drift, gap_norm_delta}]",
                      "report_path": "str|null"},
    "schema": {"version": "str", "commands": "dict — args + forme de sortie par commande"},
}


def is_json(ns) -> bool:
    return bool(getattr(ns, "json", False))


def _print(obj) -> None:
    json.dump(obj, sys.stdout, indent=2, ensure_ascii=False, default=str)
    sys.stdout.write("\n")


def emit_result(ns, command: str, data, human: Callable | None = None) -> int:
    """Émet le résultat d'une commande. Renvoie le code de sortie (0).

    `--json` : enveloppe `status="ok"` sur stdout. Sinon : `human(data)` si fourni, sinon JSON
    indenté (préserve l'affichage des commandes qui imprimaient déjà du JSON).
    """
    if is_json(ns):
        _print({"schema_version": SCHEMA_VERSION, "status": "ok", "command": command,
                "data": data, "error": None})
    elif human is not None:
        human(data)
    else:
        _print(data)
    return 0


def emit_error(ns, command: str, exc: BaseException) -> int:
    """Émet une erreur structurée. Renvoie le code de sortie (1).

    `--json` : enveloppe `status="error"` sur stdout. Sinon : message sur stderr (l'appelant
    a déjà loggé) — stdout reste vide pour ne pas tromper un parseur.
    """
    if is_json(ns):
        _print({"schema_version": SCHEMA_VERSION, "status": "error", "command": command,
                "data": None, "error": {"type": type(exc).__name__, "message": str(exc)}})
    else:
        print(f"Erreur ({type(exc).__name__}) : {exc}", file=sys.stderr)
    return 1


def parser_schema(parser) -> dict:
    """Introspecte un argparse et renvoie un dict JSON-sérialisable décrivant chaque commande.

    Pour chaque sous-commande : ses arguments (nom, type, défaut, aide, requis, positionnel) et
    la forme de ses données de sortie (`COMMAND_OUTPUT`). Source de vérité machine pour les agents.
    """
    import argparse

    commands: dict[str, dict] = {}
    subparsers_actions = [a for a in parser._actions  # noqa: SLF001
                          if isinstance(a, argparse._SubParsersAction)]
    for sp_action in subparsers_actions:
        for name, subparser in sp_action.choices.items():
            args = []
            for action in subparser._actions:  # noqa: SLF001
                if isinstance(action, argparse._HelpAction):
                    continue
                positional = not action.option_strings
                args.append({
                    "name": action.dest,
                    "flags": list(action.option_strings),
                    "positional": positional,
                    "type": getattr(action.type, "__name__", "str") if action.type else (
                        "bool" if isinstance(action, (argparse._StoreTrueAction,
                                                      argparse._StoreFalseAction)) else "str"),
                    "default": action.default,
                    "required": bool(getattr(action, "required", False)) or positional,
                    "choices": list(action.choices) if action.choices else None,
                    "help": action.help,
                })
            commands[name] = {
                "help": sp_action._choices_actions and next(  # noqa: SLF001
                    (c.help for c in sp_action._choices_actions if c.dest == name), None),
                "arguments": args,
                "output": COMMAND_OUTPUT.get(name),
            }
    return {"version": SCHEMA_VERSION, "commands": commands}
