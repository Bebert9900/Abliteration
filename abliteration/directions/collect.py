"""Collecte d'activations : residual stream par couche, à la position du dernier token."""
from __future__ import annotations

import torch

from abliteration.data.formatting import last_token_index


def pooled_last_token(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """(L+1, B, T, H) + masque (B, T) -> (L+1, B, H) au dernier token réel de chaque séquence."""
    idx = last_token_index(attention_mask)              # (B,)
    batch = torch.arange(hidden_states.size(1))
    return hidden_states[:, batch, idx, :]              # (L+1, B, H)


def _chunks(seq, size):
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


@torch.no_grad()
def collect_means(model, formatter, texts: list[str], batch_size: int = 8, device=None):
    """Moyenne, par couche, des activations du dernier token sur `texts` -> (L+1, H).

    `model(**enc, output_hidden_states=True)` doit exposer `.hidden_states` (tuple L+1).
    Le chat template et le padding gauche sont gérés par `formatter` (cf. abliteration.data).
    """
    sums = None
    n = 0
    for batch in _chunks(texts, batch_size):
        enc = formatter.tokenize(batch)
        target = device if device is not None else getattr(model, "device", None)
        if target is not None and hasattr(enc, "to"):
            enc = enc.to(target)
        out = model(
            input_ids=enc["input_ids"],
            attention_mask=enc["attention_mask"],
            output_hidden_states=True,
        )
        hs = torch.stack(tuple(out.hidden_states), dim=0)      # (L+1, B, T, H)
        pooled = pooled_last_token(hs, enc["attention_mask"])  # (L+1, B, H)
        batch_sum = pooled.sum(dim=1)                          # (L+1, H)
        sums = batch_sum if sums is None else sums + batch_sum
        n += pooled.size(1)
    return sums / n
