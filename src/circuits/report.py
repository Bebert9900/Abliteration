"""Rapport d'analyse circuitielle : lisible (texte) + sérialisable (JSON/dict).

Livrable de la PHASE 1. Le rapport sépare visuellement ce qui est CORRÉLATIONNEL (DLA, =
hypothèses) de ce qui est CAUSALEMENT VALIDÉ (patching nécessité+suffisance). Seuls les
composants validés apparaissent comme « circuit core ». Inclut un graphe d'attribution simple
(liste d'arêtes composant→logit-refus pondérées par l'effet causal), exportable.

Aucune modification de modèle ici : c'est un rapport. La décision de passer en Phase 2
appartient à l'humain au vu de ce rapport (composants, stabilité, scores causaux).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

from .dla import DLAResult
from .localize import Localization


@dataclass
class CircuitReport:
    """Rapport structuré pour un modèle donné."""
    model_name: str
    localization: Localization
    dla: DLAResult | None = None
    n_pairs: int = 0

    # -- export structuré --------------------------------------------------- #
    def to_dict(self) -> dict:
        loc = self.localization
        ev = loc.evidence

        def comp_row(c):
            e = ev[c]
            return {
                "component": c.label,
                "kind": c.kind.value,
                "layer": c.layer,
                "head": c.head,
                "dla": round(e.dla, 6),
                "necessity": round(e.necessity, 6),
                "sufficiency": round(e.sufficiency, 6),
                "causally_validated": e.causally_validated(loc.threshold),
            }

        attn, mlp = loc.attention_mlp_split()
        return {
            "model": self.model_name,
            "phase": 1,
            "n_prompt_pairs": self.n_pairs,
            "method": "DLA (correlational) + activation patching (causal necessity+sufficiency)",
            "causal_threshold": loc.threshold,
            "core_circuit": [comp_row(c) for c in loc.ranked_core()],
            "core_size": len(loc.core),
            "core_attention_mlp_split": {"attention_heads": attn, "mlp": mlp},
            "motif": {
                "gates": [c.label for c in loc.gates],
                "amplifiers": [c.label for c in loc.amplifiers],
                "gate_amplifier_detected": bool(loc.gates and loc.amplifiers),
            },
            "validation": {
                "bootstrap_jaccard": _r(loc.bootstrap_jaccard),
                "bootstrap_stable": (loc.bootstrap_jaccard is not None
                                     and loc.bootstrap_jaccard > 0.9),
                # faithfulness AUTORITAIRE = held-out (paires jamais vues à la sélection) ;
                # in-sample fourni pour transparence (anti-tautologie).
                "faithfulness": _r(loc.faithfulness),
                "faithfulness_basis": ("held_out" if loc.held_out else "in_sample"),
                "faithfulness_insample": _r(loc.faithfulness_insample),
                "n_select_pairs": loc.n_train,
                "n_heldout_pairs": loc.n_test,
                "cpr": _r(loc.cpr),
                "cmd": _r(loc.cmd),
            },
            "warnings": ([loc.holdout_warning] if loc.holdout_warning else []),
            "attribution_graph": self._graph(),
            "caveats": [
                "DLA est CORRÉLATIONNELLE : hypothèses, jamais conclusion.",
                "Seuls les composants validés par patching (nécessité+suffisance ≥ seuil) "
                "constituent le circuit core.",
                "Une tête gate peut avoir une DLA quasi nulle mais être causalement nécessaire.",
                "Champ jeune et faillible : ne pas sur-interpréter ; rapporter les négatifs.",
            ],
        }

    def _graph(self) -> list[dict]:
        """Graphe d'attribution simple : arêtes (composant → logit-refus) pondérées causalement.

        Poids = nécessité causale (effet du knockout). On ajoute les arêtes gate→amplificateur
        si le motif est détecté (le gate « route » vers les amplificateurs en aval).
        """
        loc = self.localization
        edges = []
        for c in loc.ranked_core():
            e = loc.evidence[c]
            edges.append({
                "source": c.label,
                "target": "refusal_logit",
                "weight_necessity": round(e.necessity, 6),
                "weight_dla": round(e.dla, 6),
                "role": ("gate" if c in loc.gates else
                         "amplifier" if c in loc.amplifiers else "core"),
            })
        if loc.gates and loc.amplifiers:
            for g in loc.gates:
                for a in loc.amplifiers:
                    if a.layer >= g.layer:   # routage vers l'aval
                        edges.append({
                            "source": g.label, "target": a.label,
                            "weight_necessity": round(loc.evidence[g].necessity, 6),
                            "role": "gate_routes_amplifier",
                        })
        return edges

    def to_json(self, path=None, indent: int = 2) -> str:
        s = json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)
        if path is not None:
            from pathlib import Path
            Path(path).write_text(s, encoding="utf-8")
        return s

    # -- rendu texte -------------------------------------------------------- #
    def to_text(self) -> str:
        d = self.to_dict()
        loc = self.localization
        L = []
        L.append(f"=== Analyse circuitielle du refus — {d['model']} (Phase 1) ===")
        L.append(f"Paires de prompts : {d['n_prompt_pairs']}  |  seuil causal : {d['causal_threshold']}")
        L.append(f"Méthode : {d['method']}")
        L.append("")
        L.append(f"CIRCUIT CORE (validé causalement) — {d['core_size']} composant(s) "
                 f"[{d['core_attention_mlp_split']['attention_heads']} têtes attn / "
                 f"{d['core_attention_mlp_split']['mlp']} MLP] :")
        if not loc.core:
            L.append("  (aucun composant validé causalement au seuil donné)")
        for row in d["core_circuit"]:
            L.append(f"  {row['component']:<10} nec={row['necessity']:+.3f} "
                     f"suf={row['sufficiency']:+.3f} dla={row['dla']:+.3f}")
        L.append("")
        mo = d["motif"]
        if mo["gate_amplifier_detected"]:
            L.append(f"Motif gate→amplificateur DÉTECTÉ : gates={mo['gates']} "
                     f"amplificateurs={mo['amplifiers']}")
        else:
            L.append("Motif gate→amplificateur : non détecté (ou core trop petit).")
        L.append("")
        v = d["validation"]
        L.append("VALIDATION :")
        L.append(f"  stabilité bootstrap (Jaccard) : {_fmt(v['bootstrap_jaccard'])} "
                 f"({'STABLE >0.9' if v['bootstrap_stable'] else 'INSTABLE ≤0.9'})")
        L.append(f"  faithfulness ({v['faithfulness_basis']}, autoritaire) : {_fmt(v['faithfulness'])}"
                 f"  [sélection={v['n_select_pairs']} / held-out={v['n_heldout_pairs']} paires]")
        L.append(f"    (in-sample, pour comparaison) : {_fmt(v['faithfulness_insample'])}")
        L.append(f"  CPR (circuit performance ratio) : {_fmt(v['cpr'])}")
        L.append(f"  CMD (circuit-model distance, 0=identique) : {_fmt(v['cmd'])}")
        for w in d.get("warnings", []):
            L.append(f"  ⚠ {w}")
        L.append("")
        L.append("AVERTISSEMENTS :")
        for c in d["caveats"]:
            L.append(f"  - {c}")
        return "\n".join(L)

    # -- résumé court pour `diagnose` --------------------------------------- #
    def short_summary(self) -> str:
        loc = self.localization
        attn, mlp = loc.attention_mlp_split()
        jac = _fmt(_r(loc.bootstrap_jaccard))
        if not loc.core:
            return f"circuit refus : aucun composant validé causalement (seuil {loc.threshold})."
        core = ", ".join(c.label for c in loc.ranked_core()[:5])
        more = "" if len(loc.core) <= 5 else f" (+{len(loc.core) - 5})"
        motif = " | motif gate→amp" if (loc.gates and loc.amplifiers) else ""
        return (f"circuit refus : {len(loc.core)} comp. validés [{attn} attn/{mlp} mlp] "
                f"core={core}{more} | bootstrap Jaccard={jac}{motif}")


def _r(x):
    return None if x is None else round(x, 6)


def _fmt(x):
    return "n/a" if x is None else f"{x:.3f}"
