"""Mise en forme des prompts pour la collecte d'activations.

Deux règles dures vivent ici :
- **Chat template systématique** : la direction de refus vit dans l'espace du format instruct,
  pas du texte brut.
- **Padding à gauche** : pour que le « dernier token » de l'instruction soit aligné en fin de
  séquence sur tout le batch.
"""
from __future__ import annotations

import torch


class PromptFormatter:
    """Enveloppe un tokenizer HF : applique le chat template et force le padding à gauche."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        # Règle dure : padding à gauche pour l'indexation du dernier token.
        self.tokenizer.padding_side = "left"

    def format_chat(self, text: str) -> str:
        return self.tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
        )

    def tokenize(self, texts: list[str]):
        chats = [self.format_chat(t) for t in texts]
        return self.tokenizer(chats, return_tensors="pt", padding=True)


def last_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """Index du dernier token réel par séquence, robuste au côté de padding.

    Position du dernier 1 dans chaque ligne du masque : `T-1 - argmax(flip(mask))`. Fonctionne
    que le padding soit à gauche (cas normal) ou à droite.
    """
    seq_len = attention_mask.shape[1]
    last_one_from_end = attention_mask.flip(dims=[1]).argmax(dim=1)
    return seq_len - 1 - last_one_from_end
