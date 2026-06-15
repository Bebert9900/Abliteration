"""Passe HORS-LIGNE unique : re-juger les sorties harmful avec un juge LLM, base ET abliteré.

POURQUOI — Les générations brutes n'avaient PAS été sauvegardées (base-3b-eval.json et
out-3b-abl/eval_report.json ne contiennent que des scores). Le taux de refus 90%->0% a été
mesuré par un juge HEURISTIQUE (mots-clés), qui rate les refus déguisés (« Sure, here's how... »
+ rien). On lève le doute : une seule passe propre qui (1) régénère en SAUVEGARDANT les textes,
(2) les re-classe avec un juge LLM hors-ligne selon une rubrique explicite, (3) compare au juge
heuristique recalculé sur les MÊMES textes.

CADRAGE — Éval hors-ligne après coup, pas une dépendance du produit (cf. CLAUDE.md / llm_judge.py).
On ne relance NI l'abliteration NI une autre variante. Juge LLM = modèle de base local.

GPU 8 Go : on charge les modèles SÉQUENTIELLEMENT (base, puis abliteré, puis juge), en libérant
entre chaque. Holdout reproductible : seed=0, frac=0.2 (défauts CLI) -> mêmes 30 prompts.

Sorties :
  rejudge_generations.json  -> prompts + réponses brutes (base & abliteré) + params de génération
  rejudge_report.json       -> tableau comparatif heuristique vs LLM + comptes par étiquette
"""
from __future__ import annotations

import argparse
import gc
import json
import sys as _sys
from pathlib import Path
from pathlib import Path as _Path

import torch

_sys.path.insert(0, str(_Path(__file__).resolve().parent.parent))  # repo root sur sys.path

from abliteration.data import FourClassData, PromptClass, PromptFormatter
from abliteration.eval import (
    KeywordRefusalJudge,
    LLMRefusalJudge,
    evasive_rate,
    generate_responses,
    label_counts,
    llm_refusal_rate,
    refusal_rate,
)
from abliteration.models import load_model

BASE_ID = "Qwen/Qwen2.5-3B-Instruct"
ABL_DIR = "artifacts/out-3b-abl"
# Garde-fou anti-gaming : assez de tokens pour exposer un « Sure! » suivi de rien. On dépasse
# largement le min_new_tokens=16 de la passe d'origine, qu'on signale dans le rapport.
MAX_NEW_TOKENS = 256
MIN_NEW_TOKENS = 64


def _free(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _generate_for(model_id_or_dir, prompts, device, dtype):
    print(f"[gen] chargement {model_id_or_dir} …", flush=True)
    model, tok = load_model(model_id_or_dir, dtype=dtype, device_map=device)
    formatter = PromptFormatter(tok)
    print(f"[gen] génération sur {len(prompts)} prompts harmful "
          f"(max_new={MAX_NEW_TOKENS}, min_new={MIN_NEW_TOKENS}) …", flush=True)
    resp = generate_responses(model, formatter, prompts, max_new_tokens=MAX_NEW_TOKENS,
                              min_new_tokens=MIN_NEW_TOKENS, batch_size=4, device=None)
    _free(model)
    return resp


def _make_llm_judge(judge_id, device, dtype):
    print(f"[judge] chargement juge {judge_id} …", flush=True)
    model, tok = load_model(judge_id, dtype=dtype, device_map=device)

    @torch.no_grad()
    def generate_fn(prompt: str) -> str:
        chat = tok.apply_chat_template([{"role": "user", "content": prompt}],
                                       tokenize=False, add_generation_prompt=True)
        enc = tok(chat, return_tensors="pt").to(model.device)
        out = model.generate(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                             max_new_tokens=8, do_sample=False,
                             pad_token_id=tok.pad_token_id or tok.eos_token_id)
        return tok.decode(out[0, enc["input_ids"].shape[1]:], skip_special_tokens=True)

    return LLMRefusalJudge(generate_fn), model


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default=BASE_ID)
    ap.add_argument("--ablated", default=ABL_DIR)
    ap.add_argument("--judge", default=BASE_ID, help="Modèle juge LLM (même rubrique des deux côtés).")
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--holdout", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--gen-out", default="results/rejudge_generations.json")
    ap.add_argument("--report-out", default="results/rejudge_report.json")
    ap.add_argument("--reuse-gen", action="store_true",
                    help="Sauter la régénération si --gen-out existe déjà (re-juge seulement).")
    ns = ap.parse_args()

    base_dir = Path(ns.data_dir)
    paths = {cls: base_dir / f"{cls.value}.txt" for cls in PromptClass}
    data = FourClassData.load(paths, holdout_fraction=ns.holdout, seed=ns.seed)
    prompts = [p.text for p in data.holdout(PromptClass.HARMFUL)]
    print(f"[data] {len(prompts)} prompts harmful holdout (seed={ns.seed}, frac={ns.holdout})")

    gen_path = Path(ns.gen_out)
    if ns.reuse_gen and gen_path.exists():
        g = json.loads(gen_path.read_text())
        prompts = g["prompts"]; base_resp = g["base_responses"]; abl_resp = g["ablated_responses"]
        print(f"[gen] réutilisation de {gen_path} ({len(prompts)} prompts)")
    else:
        base_resp = _generate_for(ns.base, prompts, ns.device, ns.dtype)
        abl_resp = _generate_for(ns.ablated, prompts, ns.device, ns.dtype)
        gen_path.write_text(json.dumps({
            "base_model": ns.base, "ablated_model": ns.ablated,
            "max_new_tokens": MAX_NEW_TOKENS, "min_new_tokens": MIN_NEW_TOKENS,
            "n_prompts": len(prompts), "seed": ns.seed, "holdout_fraction": ns.holdout,
            "prompts": prompts, "base_responses": base_resp, "ablated_responses": abl_resp,
        }, indent=2, ensure_ascii=False))
        print(f"[gen] textes bruts sauvegardés -> {gen_path}")

    # --- Juge heuristique (recalculé sur les MÊMES textes, pour une comparaison équitable) ---
    kw = KeywordRefusalJudge()
    heur_base = refusal_rate(prompts, base_resp, kw)
    heur_abl = refusal_rate(prompts, abl_resp, kw)

    # --- Juge LLM (même rubrique des deux côtés) ---
    judge, judge_model = _make_llm_judge(ns.judge, ns.device, ns.dtype)
    print("[judge] classement base …", flush=True)
    base_labels = judge.classify_batch(prompts, base_resp)
    print("[judge] classement abliteré …", flush=True)
    abl_labels = judge.classify_batch(prompts, abl_resp)
    _free(judge_model)

    def block(name, resp, labels, heur):
        c = label_counts(labels)
        return {
            "model": name,
            "refusal_heuristic": heur,
            "refusal_llm": llm_refusal_rate(labels),
            "evasive_llm": evasive_rate(labels),
            "counts": c, "n": len(labels),
            "labels": labels,
        }

    report = {
        "judge_model": ns.judge,
        "rubric": "REFUSAL / NON_REFUSAL / EVASIVE (cf. abliteration/eval/llm_judge.RUBRIC_PROMPT)",
        "gen_params": {"max_new_tokens": MAX_NEW_TOKENS, "min_new_tokens": MIN_NEW_TOKENS},
        "base": block(ns.base, base_resp, base_labels, heur_base),
        "ablated": block(ns.ablated, abl_resp, abl_labels, heur_abl),
    }
    Path(ns.report_out).write_text(json.dumps(report, indent=2, ensure_ascii=False))

    # --- Affichage tableau ---
    def gap(b): return b["refusal_llm"] - b["refusal_heuristic"]
    print("\n=== TABLEAU COMPARATIF (refus) ===")
    print(f"{'modèle':<28} {'heuristique':>11} {'LLM':>7} {'évasif':>8} {'écart':>7}")
    for b in (report["base"], report["ablated"]):
        print(f"{b['model']:<28} {b['refusal_heuristic']*100:>10.1f}% "
              f"{b['refusal_llm']*100:>6.1f}% {b['evasive_llm']*100:>7.1f}% "
              f"{gap(b)*100:>+6.1f}pt")
    print(f"\nrapport  -> {ns.report_out}\ngénérations -> {ns.gen_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
