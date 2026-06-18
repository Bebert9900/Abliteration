"""Génération batchée pour l'évaluation (axe refus + axe préservation).

L'éval honnête impose `min_new_tokens` pour empêcher le gaming par réponses vides,
et le chat template + padding gauche (déjà gérés par `PromptFormatter`). On ne décode QUE les
tokens générés (pas le prompt) pour que les juges (refus, négation, tool-call) voient la réponse.
"""
from __future__ import annotations

import json
from pathlib import Path

import torch


def dump_generations(path, prompts, responses, *, judge=None, degeneracy_check=None, **meta):
    """Persiste les générations brutes (prompt + réponse) pour relecture/re-jugement hors-ligne.

    Recommandation de la passe de re-jugement : ne JAMAIS jeter les textes bruts (on ne peut pas
    re-juger des scores). Un enregistrement par cas, numéroté, aligné sur l'ordre du holdout.
    `judge`/`degeneracy_check` (optionnels) ajoutent un INDICE heuristique par cas — jamais une
    vérité (cf. RAPPORT : le 0 % heuristique sous-estime, le juge LLM 3B est cassé).
    """
    records = []
    for i, (p, r) in enumerate(zip(prompts, responses)):
        rec = {"idx": i, "prompt": p, "response": r}
        if judge is not None:
            rec["heuristic_refusal"] = bool(judge(p, r))
        if degeneracy_check is not None:
            rec["degenerate"] = bool(degeneracy_check(r))
        records.append(rec)
    payload = {"n": len(records), **meta, "records": records}
    Path(path).write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    return payload


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


@torch.no_grad()
def generate_responses(
    model,
    formatter,
    texts: list[str],
    max_new_tokens: int = 128,
    min_new_tokens: int = 100,
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
