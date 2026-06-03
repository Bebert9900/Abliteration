"""Génération batchée pour l'évaluation (axe refus + axe préservation).

L'éval honnête (KB §8.3) impose `min_new_tokens` pour empêcher le gaming par réponses vides,
et le chat template + padding gauche (déjà gérés par `PromptFormatter`). On ne décode QUE les
tokens générés (pas le prompt) pour que les juges (refus, négation, tool-call) voient la réponse.
"""
from __future__ import annotations

import torch


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


@torch.no_grad()
def generate_responses(
    model,
    formatter,
    texts: list[str],
    max_new_tokens: int = 128,
    min_new_tokens: int = 16,
    batch_size: int = 8,
    device=None,
) -> list[str]:
    """Renvoie une réponse (str) par prompt, dans l'ordre. Décode seulement les tokens générés."""
    tok = formatter.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    out_texts: list[str] = []
    for batch in _chunks(texts, batch_size):
        enc = formatter.tokenize(batch)
        if device is not None and hasattr(enc, "to"):
            enc = enc.to(device)
        else:
            enc = {k: v.to(model.device) for k, v in enc.items()}
        gen = model.generate(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        # padding gauche -> le prompt occupe les premières colonnes ; on coupe à la longueur d'entrée.
        prompt_len = enc["input_ids"].shape[1]
        new_tokens = gen[:, prompt_len:]
        out_texts.extend(tok.batch_decode(new_tokens, skip_special_tokens=True))
    return out_texts


@torch.no_grad()
def harmless_logits(model, formatter, texts: list[str], batch_size: int = 4, device=None):
    """Logits next-token (concaténés sur les prompts) pour la mesure de KL de préservation.

    Renvoie un tenseur CPU (N_tokens, V). Utilisé pour KL(base ‖ abliteré) sur prompts harmless.
    """
    chunks_out = []
    for batch in _chunks(texts, batch_size):
        enc = formatter.tokenize(batch)
        enc = {k: v.to(model.device) for k, v in enc.items()}
        out = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        # on ne garde que les positions réelles (masque=1) pour ne pas polluer avec le padding.
        mask = enc["attention_mask"].bool()
        logits = out.logits[mask]                       # (sum_tokens, V)
        chunks_out.append(logits.float().cpu())
    return torch.cat(chunks_out, dim=0)
