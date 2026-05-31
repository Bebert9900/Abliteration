"""Tests de la mise en forme : chat template, padding à gauche, index du dernier token."""
import torch

from src.data import PromptFormatter, last_token_index


class FakeTokenizer:
    """Tokenizer minimal pour tester notre logique sans transformers."""

    def __init__(self):
        self.padding_side = "right"  # défaut HF ; le formatter doit forcer "left"
        self.template_calls = []
        self.call_args = None

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        self.template_calls.append((messages, tokenize, add_generation_prompt))
        return f"<chat>{messages[0]['content']}</chat>"

    def __call__(self, texts, return_tensors, padding):
        self.call_args = {"texts": texts, "return_tensors": return_tensors, "padding": padding}
        return {"received": texts}


def test_format_chat_wraps_prompt_in_user_turn_with_generation_prompt():
    tok = FakeTokenizer()
    fmt = PromptFormatter(tok)
    out = fmt.format_chat("bonjour")
    assert out == "<chat>bonjour</chat>"
    messages, tokenize, add_gen = tok.template_calls[0]
    assert messages == [{"role": "user", "content": "bonjour"}]
    assert tokenize is False
    assert add_gen is True


def test_formatter_forces_left_padding():
    tok = FakeTokenizer()
    PromptFormatter(tok)
    assert tok.padding_side == "left"


def test_tokenize_applies_chat_template_to_each_prompt():
    tok = FakeTokenizer()
    fmt = PromptFormatter(tok)
    fmt.tokenize(["a", "b"])
    assert tok.call_args["texts"] == ["<chat>a</chat>", "<chat>b</chat>"]
    assert tok.call_args["padding"] is True


def test_last_token_index_with_left_padding():
    # 0 = padding (à gauche), 1 = token réel -> dernier réel toujours en position T-1
    mask = torch.tensor([[0, 0, 1, 1], [0, 1, 1, 1]])
    assert last_token_index(mask).tolist() == [3, 3]


def test_last_token_index_with_right_padding():
    # robustesse : si jamais padding à droite, l'index reste correct (somme-1)
    mask = torch.tensor([[1, 1, 0, 0], [1, 1, 1, 0]])
    assert last_token_index(mask).tolist() == [1, 2]
