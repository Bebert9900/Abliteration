"""Module éval : refus, KL, négation, agentique, benchmarks externes, rapport bi-axe."""
from .agentic import (
    ToolCall,
    agentic_score,
    arg_accuracy,
    hallucinated_args,
    multi_step_success,
    parse_tool_call,
    schema_valid,
)
from .benchmarks import BenchmarkNotInstalled, available_benchmarks, run_benchmark
from .generate import generate_responses, harmless_logits
from .judges import is_sycophantic, negates_correctly
from .kl import kl_divergence
from .llm_judge import (
    EVASIVE,
    LABELS,
    NON_REFUSAL,
    REFUSAL,
    RUBRIC_PROMPT,
    LLMRefusalJudge,
    evasive_rate,
    label_counts,
    llm_refusal_rate,
    parse_label,
)
from .refusal import (
    KeywordRefusalJudge,
    degeneracy_rate,
    empty_rate,
    follow_rate,
    is_degenerate,
    negation_retention,
    refusal_rate,
)
from .report import EvalReport

__all__ = [
    # refus & métriques
    "KeywordRefusalJudge",
    "is_degenerate",
    "degeneracy_rate",
    "refusal_rate",
    "negation_retention",
    "follow_rate",
    "empty_rate",
    "kl_divergence",
    "generate_responses",
    "harmless_logits",
    "negates_correctly",
    "is_sycophantic",
    # juge LLM hors-ligne (re-classement des refus)
    "LLMRefusalJudge",
    "parse_label",
    "label_counts",
    "llm_refusal_rate",
    "evasive_rate",
    "RUBRIC_PROMPT",
    "LABELS",
    "REFUSAL",
    "NON_REFUSAL",
    "EVASIVE",
    # agentique
    "ToolCall",
    "parse_tool_call",
    "schema_valid",
    "hallucinated_args",
    "arg_accuracy",
    "multi_step_success",
    "agentic_score",
    # benchmarks & rapport
    "BenchmarkNotInstalled",
    "available_benchmarks",
    "run_benchmark",
    "EvalReport",
]
