"""Compare preserving (out-3b-abl) vs NPB (out-3b-npb) sur le MÊME holdout déterministe.

Le bug de holdout (hash() randomisé par processus) est corrigé -> on ré-évalue les DEUX modèles
sur exactement les mêmes prompts, dans les mêmes conditions. Axes automatisables (fiables) +
génération harmful @256t pour LECTURE HUMAINE du refus (le seul axe fiable, cf. RAPPORT).

Sorties :
  compare_eval.json                       -> axes automatisables, preserving vs NPB
  rejudge_preserving_generations.json     -> 30 réponses preserving @256t (lecture humaine)
  rejudge_npb_generations.json            -> 30 réponses NPB @256t (lecture humaine)
"""
from __future__ import annotations

import gc
import json
import sys as _sys
from pathlib import Path
from pathlib import Path as _Path

import torch

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # repo root sur sys.path

from meridian.data import FourClassData, PromptClass, PromptFormatter
from meridian.eval import (
    KeywordRefusalJudge,
    agentic_score,
    degeneracy_rate,
    dump_generations,
    empty_rate,
    follow_rate,
    generate_responses,
    harmless_logits,
    is_sycophantic,
    kl_divergence,
    negates_correctly,
    negation_retention,
    parse_tool_call,
    refusal_rate,
    schema_valid,
)
from meridian.eval.refusal import is_degenerate
from meridian.models import load_model

BASE = "Qwen/Qwen2.5-3B-Instruct"
MODELS = [("preserving", "artifacts/out-3b-abl"), ("npb", "artifacts/out-3b-npb")]
MAX_NEW_H, MIN_NEW_H = 256, 64   # harmful : conditions de lecture (= référence preserving)
MAX_NEW_CAP = 128                # négation/agentique : conditions de l'éval bi-axe

paths = {cls: Path("data") / f"{cls.value}.txt" for cls in PromptClass}
data = FourClassData.load(paths, holdout_fraction=0.2, seed=0)
harmful = [p.text for p in data.holdout(PromptClass.HARMFUL)]
neg = [p.text for p in data.holdout(PromptClass.LEGITIMATE_NEGATION)]
ag = data.holdout(PromptClass.AGENTIC)
kl_texts = [p.text for p in data.holdout(PromptClass.HARMLESS)][:16]
print(f"[holdout déterministe] harmful={len(harmful)} neg={len(neg)} agentic={len(ag)} kl={len(kl_texts)}")


def _agentic_prompt(p):
    schema = p.meta.get("tool", {})
    name = schema.get("name", "the_tool")
    req = ", ".join(schema.get("parameters", {}).get("required", []))
    return (f"{p.text}\n\nAvailable tool: `{name}` (required arguments: {req}).\n"
            f"Respond with ONLY a JSON object of the form "
            f'{{"name": "{name}", "arguments": {{...}}}}.')


def _free(m):
    try:
        m.to("meta")            # détache les poids du GPU avant libération (anti-fragmentation)
    except Exception:
        pass
    del m
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()


# --- Logits de base pour la KL (une seule fois) ---
print("[base] chargement pour logits KL …", flush=True)
bmodel, btok = load_model(BASE, dtype="bfloat16", device_map="cuda")
base_logits = harmless_logits(bmodel, PromptFormatter(btok), kl_texts, batch_size=4)
del btok
_free(bmodel)
if torch.cuda.is_available():
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
print(f"[base] libérée, GPU libre ~{torch.cuda.mem_get_info()[0] / 2**30:.1f} Gio", flush=True)

judge = KeywordRefusalJudge()
results = {}
for tag, path in MODELS:
    print(f"[{tag}] chargement {path} …", flush=True)
    model, tok = load_model(path, dtype="bfloat16", device_map="cuda")
    fmt = PromptFormatter(tok)

    # Harmful @256 -> lecture humaine + dégénérescence/vide/indice heuristique
    resp_h = generate_responses(model, fmt, harmful, max_new_tokens=MAX_NEW_H,
                                min_new_tokens=MIN_NEW_H, batch_size=4)
    dump_generations(f"results/rejudge_{tag}_generations.json", harmful, resp_h, judge=judge,
                     degeneracy_check=is_degenerate, model=f"{path} ({tag})",
                     max_new_tokens=MAX_NEW_H, min_new_tokens=MIN_NEW_H,
                     note="Refus AUTORITAIRE = lecture humaine ; heuristic_refusal = indice.")

    # Négation logique (préservation)
    resp_n = generate_responses(model, fmt, neg, max_new_tokens=MAX_NEW_CAP, batch_size=4)
    # Agentique (validité de schéma de tool call)
    resp_a = generate_responses(model, fmt, [_agentic_prompt(p) for p in ag],
                                max_new_tokens=MAX_NEW_CAP, batch_size=4)
    valid = sum(1 for p, o in zip(ag, resp_a)
                if (c := parse_tool_call(o)) is not None and p.meta.get("tool")
                and schema_valid(c, p.meta["tool"]))
    sv = valid / len(ag) if ag else 0.0

    abl_logits = harmless_logits(model, fmt, kl_texts, batch_size=2)
    n = min(base_logits.shape[0], abl_logits.shape[0])

    results[tag] = {
        "model": path,
        "refusal_heuristic_256t": refusal_rate(harmful, resp_h, judge),  # INDICE, pas vérité
        "refusal_human_256t": None,                                      # À RENSEIGNER (lecture)
        "negation_retention": negation_retention(resp_n, negates_correctly),
        "follow_rate": follow_rate(resp_n, is_sycophantic),
        "agentic_score": agentic_score(sv, sv, sv),
        "degeneracy_rate": degeneracy_rate(resp_h),
        "empty_rate": empty_rate(resp_h),
        "kl_preservation": float(kl_divergence(base_logits[:n], abl_logits[:n])),
    }
    print(f"[{tag}] {json.dumps(results[tag], ensure_ascii=False)}", flush=True)
    _free(model)

payload = {
    "base": BASE,
    "holdout": {"n_harmful": len(harmful), "seed": 0, "fraction": 0.2, "deterministic": True},
    "gen_params": {"harmful": {"max_new": MAX_NEW_H, "min_new": MIN_NEW_H},
                   "neg_agentic": {"max_new": MAX_NEW_CAP}},
    "results": results,
}
Path("results/compare_eval.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False))
print("\n[ok] compare_eval.json + rejudge_{preserving,npb}_generations.json écrits.")
