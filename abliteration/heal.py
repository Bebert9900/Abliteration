"""Récupération agentique (« healing ») post-abliteration : LoRA SFT.

Si l'éval révèle un effondrement agentique RÉSIDUEL après abliteration (tool calls invalides,
arguments hallucinés, échec multi-tours) malgré la variante préservante, un court fine-tuning de
récupération sur ~100–300 traces de tool use restaure typiquement les capacités sans réintroduire
le refus (on n'entraîne JAMAIS sur des refus). On utilise LoRA (peft) pour ne toucher qu'un petit
adaptateur, ce qui limite le risque de réintroduire la direction de refus dans les poids de base.
(Source : fournie par l'utilisateur, arXiv:2604.08388 — HORS KB v.mai-2026, à confirmer.)

Découpage : préparation des données (`load_traces`, `format_example`, `iter_training_examples`)
PURE et testable sans GPU ; l'entraînement (`heal`) importe peft/transformers paresseusement et
lève une erreur claire si ces dépendances optionnelles manquent.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class HealConfig:
    model_dir: str
    traces_path: str                 # jsonl de traces de tool use (succès et erreurs)
    n_traces: int = 200              # ~100–300 recommandé
    method: str = "lora_sft"         # LoRA SFT par défaut (peft)
    out_dir: str = "./out-healed"
    epochs: int = 1
    lr: float = 1e-4
    batch_size: int = 4
    max_len: int = 1024
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )


def load_traces(path: str, limit: int | None = None) -> list[dict]:
    """Charge des traces de tool use depuis un JSONL. Ignore les lignes vides.

    Schémas acceptés par trace :
      - {"messages": [{"role": ..., "content": ...}, ...]}  (conversation, format préféré)
      - {"prompt": "...", "completion": "..."}              (paire simple)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Traces introuvables : {path}")
    traces = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "messages" not in obj and not ("prompt" in obj and "completion" in obj):
            raise ValueError(
                f"Trace invalide (attendu 'messages' ou 'prompt'+'completion') : {obj!r}"
            )
        traces.append(obj)
        if limit is not None and len(traces) >= limit:
            break
    return traces


def format_example(trace: dict, tokenizer=None) -> tuple[str, str]:
    """Renvoie (prompt, completion) en texte pour une trace.

    Pour le format conversation, le dernier message (assistant) est la cible (`completion`) et le
    préfixe est rendu via le chat template du tokenizer (règle : chat template toujours).
    Sans tokenizer, on retombe sur une concaténation simple `role: content`.
    """
    if "messages" in trace:
        messages = trace["messages"]
        if not messages:
            raise ValueError("Trace 'messages' vide.")
        *context, last = messages
        completion = last["content"]
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(context, tokenize=False,
                                                   add_generation_prompt=True)
        else:
            prompt = "".join(f"{m['role']}: {m['content']}\n" for m in context)
        return prompt, completion
    return trace["prompt"], trace["completion"]


def iter_training_examples(traces: list[dict], tokenizer=None):
    """Génère les paires (prompt, completion) formatées prêtes pour le SFT (masquage du prompt)."""
    for t in traces:
        yield format_example(t, tokenizer)


def heal(config: HealConfig):
    """LoRA SFT de récupération agentique : entraîne un adaptateur LoRA sur les traces.

    Dépendances optionnelles : `peft` (LoRA) + `transformers`. Lève une RuntimeError claire si
    absentes. À lancer UNIQUEMENT si l'éval montre un effondrement agentique résiduel après
    abliteration préservante. Ré-évaluer ensuite l'agentique (abliteration.eval).
    """
    try:
        import torch
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "heal() (LoRA SFT) requiert `peft` et `transformers` : pip install peft transformers. "
            f"Dépendance manquante : {e.name}."
        ) from e

    tok = AutoTokenizer.from_pretrained(config.model_dir)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(config.model_dir, torch_dtype=torch.bfloat16)
    lora = LoraConfig(
        r=config.lora_r, lora_alpha=config.lora_alpha, lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules, task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora)
    model.train()

    examples = list(iter_training_examples(load_traces(config.traces_path, config.n_traces), tok))
    if not examples:
        raise ValueError("Aucune trace exploitable pour le healing.")

    device = next(model.parameters()).device
    opt = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.lr)

    def _encode(prompt: str, completion: str):
        """Tokenise prompt+completion ; masque les tokens du prompt dans les labels (-100)."""
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        c_ids = tok(completion, add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
        ids = (p_ids + c_ids)[: config.max_len]
        labels = ([-100] * len(p_ids) + c_ids)[: config.max_len]
        return ids, labels

    for _epoch in range(config.epochs):
        for i in range(0, len(examples), config.batch_size):
            batch = [_encode(p, c) for p, c in examples[i : i + config.batch_size]]
            maxlen = max(len(ids) for ids, _ in batch)
            input_ids, labels, attn = [], [], []
            for ids, labs in batch:
                pad = maxlen - len(ids)
                input_ids.append(ids + [tok.pad_token_id] * pad)
                labels.append(labs + [-100] * pad)
                attn.append([1] * len(ids) + [0] * pad)
            out = model(
                input_ids=torch.tensor(input_ids, device=device),
                attention_mask=torch.tensor(attn, device=device),
                labels=torch.tensor(labels, device=device),
            )
            out.loss.backward()
            opt.step()
            opt.zero_grad()

    Path(config.out_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(config.out_dir)   # sauvegarde l'adaptateur LoRA
    tok.save_pretrained(config.out_dir)
    return config.out_dir
