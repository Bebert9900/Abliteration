"""Adaptateurs vers les harnais d'évaluation externes.

Ces harnais ne sont pas réimplémentés : on les branche s'ils sont installés, sinon on lève une
erreur claire avec l'indication d'installation. Cela couvre :
- capacité : lm-evaluation-harness (MMLU/GSM8K/BBH/GPQA/IFEval), bigcode-evaluation-harness
  (HumanEval/MBPP/Pro) ;
- agentique : BFCL (function calling, éval AST), tau-bench (multi-tours), Noisy-ToolBench
  (robustesse aux instructions imparfaites — set custom côté projet).

NB (CLAUDE.md) : ne jamais livrer des poids mesurés en 4-bit ; la quantification ne sert qu'à
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


def run_benchmark(name: str, model, **kwargs):
    """Lance un benchmark externe. Lève ValueError si inconnu, BenchmarkNotInstalled si absent."""
    if name not in _BENCHMARKS:
        raise ValueError(f"benchmark inconnu : {name!r}. Connus : {available_benchmarks()}")
    module, hint = _BENCHMARKS[name]
    try:
        __import__(module)
    except ImportError as e:
        raise BenchmarkNotInstalled(
            f"Le benchmark '{name}' requiert le paquet '{module}' (absent). Installer : {hint}"
        ) from e
    # Câblage réel à compléter selon l'API du harnais ; on échoue explicitement tant que non fait.
    raise NotImplementedError(
        f"Adaptateur '{name}' : '{module}' est installé mais le câblage runner n'est pas encore fait."
    )
