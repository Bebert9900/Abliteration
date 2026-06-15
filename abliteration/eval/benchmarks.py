"""Adaptateurs vers les harnais d'évaluation externes.

Ces harnais ne sont pas réimplémentés : on les branche s'ils sont installés, sinon on lève une
erreur claire avec l'indication d'installation. Cela couvre :
- capacité : lm-evaluation-harness (MMLU/GSM8K/BBH/GPQA/IFEval), bigcode-evaluation-harness
  (HumanEval/MBPP/Pro) ;
- agentique : BFCL (function calling, éval AST), tau-bench (multi-tours), Noisy-ToolBench
  (robustesse aux instructions imparfaites — set custom côté projet).

NB : ne jamais livrer des poids mesurés en 4-bit ; la quantification ne sert qu'à
la mesure.
"""
from __future__ import annotations


class BenchmarkNotInstalled(RuntimeError):
    pass


# nom -> (module pip à importer, indice d'installation)
_BENCHMARKS = {
    "mmlu": ("lm_eval", "pip install lm-eval"),
    "gsm8k": ("lm_eval", "pip install lm-eval"),
    "bbh": ("lm_eval", "pip install lm-eval"),
    "gpqa": ("lm_eval", "pip install lm-eval"),
    "ifeval": ("lm_eval", "pip install lm-eval"),
    "humaneval": ("bigcode_eval", "pip install bigcode-eval"),
    "mbpp": ("bigcode_eval", "pip install bigcode-eval"),
    "bfcl": ("bfcl", "pip install bfcl-eval"),
    "taubench": ("tau_bench", "pip install tau-bench"),
}


def available_benchmarks() -> list[str]:
    return sorted(_BENCHMARKS)


# Métrique « phare » à extraire du dict de résultats lm-eval, par benchmark.
_LM_EVAL_METRIC = {
    "mmlu": "acc",
    "gsm8k": "exact_match",
    "bbh": "exact_match",
    "gpqa": "acc",
    "ifeval": "prompt_level_strict_acc",
}


def _pick_metric(results: dict, prefer: str) -> tuple[str, float]:
    """Choisit la clé métrique (lm-eval suffixe par filtre, ex 'acc,none').

    Pour gsm8k on privilégie le filtre 'flexible-extract' : 'strict-match' exige le format
    '#### <nb>' que les modèles chat ne produisent pas (donne 0 à tort).
    """
    # priorité au bon filtre d'extraction quand plusieurs existent pour la même métrique.
    for key, val in results.items():
        base, _, flt = key.partition(",")
        if base == prefer and flt == "flexible-extract" and isinstance(val, (int, float)):
            return key, float(val)
    for key, val in results.items():
        if key.split(",")[0] == prefer and isinstance(val, (int, float)):
            return key, float(val)
    # repli : première métrique numérique non-stderr.
    for key, val in results.items():
        if "stderr" not in key and isinstance(val, (int, float)):
            return key, float(val)
    raise RuntimeError(f"aucune métrique numérique dans {results!r}")


def run_benchmark(name: str, model, *, device=None, batch_size="auto", limit=None, **kwargs):
    """Lance un benchmark externe et renvoie {benchmark, task, metric, score, n}.

    `model` est un chemin/identifiant HF (lm-eval recharge le modèle ; quantif. mesure seulement,
    poids livrés en bf16). `limit` restreint le nombre d'exemples (sous-ensemble rapide).
    """
    if name not in _BENCHMARKS:
        raise ValueError(f"benchmark inconnu : {name!r}. Connus : {available_benchmarks()}")
    module, hint = _BENCHMARKS[name]
    try:
        __import__(module)
    except ImportError as e:
        raise BenchmarkNotInstalled(
            f"Le benchmark '{name}' requiert le paquet '{module}' (absent). Installer : {hint}"
        ) from e

    if module != "lm_eval":
        raise NotImplementedError(
            f"Adaptateur '{name}' : runner pour '{module}' non câblé (seul lm-eval l'est)."
        )

    import lm_eval

    # `apply_chat_template=True` : indispensable pour un modèle instruct (sinon GSM8K/MMLU
    # sous-estimés). Même réglage des deux côtés -> comparaison base vs abliteré valide.
    model_args = f"pretrained={model},dtype=bfloat16,trust_remote_code=True"
    out = lm_eval.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks=[name],
        batch_size=batch_size,
        device=device,
        limit=limit,
        apply_chat_template=kwargs.get("apply_chat_template", True),
        fewshot_as_multiturn=kwargs.get("fewshot_as_multiturn", False),
    )
    task_results = out["results"][name]
    metric_key, score = _pick_metric(task_results, _LM_EVAL_METRIC.get(name, "acc"))
    n = out.get("n-samples", {}).get(name, {}).get("effective")
    return {"benchmark": name, "task": name, "metric": metric_key, "score": score, "n": n}
