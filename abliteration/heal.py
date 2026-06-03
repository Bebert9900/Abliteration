"""Récupération agentique (« healing ») — STUB documenté, interface seulement.

Si l'éval révèle un effondrement agentique RÉSIDUEL après abliteration (tool calls invalides,
arguments hallucinés, échec multi-tours) malgré la variante `preserving`, un court fine-tuning
de récupération sur ~100–300 traces de tool use restaure typiquement les capacités agentiques
sans réintroduire le refus.
(Source : fournie par l'utilisateur, arXiv:2604.08388 — HORS KB v.mai-2026, à confirmer/intégrer.)

Volontairement NON implémenté (cf. TASK 6) : on pose l'interface et la doc. Le câblage réel
brancherait un LoRA SFT (peft/trl) sur un jeu de traces (succès ET erreurs d'outils).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class HealConfig:
    model_dir: str
    traces_path: str            # jsonl de traces de tool use (succès et erreurs)
    n_traces: int = 200         # ~100–300 recommandé
    method: str = "lora_sft"    # LoRA SFT par défaut (peft/trl)
    out_dir: str = "./out-healed"


def heal(config: HealConfig):
    """Point d'entrée du fine-tuning de récupération agentique. NON implémenté (stub)."""
    raise NotImplementedError(
        "heal() est un stub documenté (TASK 6). À câbler : LoRA SFT sur ~100–300 traces de tool "
        "use via peft/trl, puis ré-évaluer l'agentique (abliteration.eval). Lancer uniquement si l'éval "
        "montre un effondrement agentique résiduel après abliteration `preserving`."
    )
