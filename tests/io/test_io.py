"""Tests IO : sauvegarde du modèle et génération de model card transparente."""
from src.io import build_model_card, save_model


class FakeModel:
    def __init__(self):
        self.saved = None

    def save_pretrained(self, out_dir, safe_serialization):
        self.saved = (out_dir, safe_serialization)


class FakeTok:
    def __init__(self):
        self.saved = None

    def save_pretrained(self, out_dir):
        self.saved = out_dir


def test_save_model_uses_safetensors(tmp_path):
    m, t = FakeModel(), FakeTok()
    save_model(m, t, tmp_path)
    assert m.saved == (str(tmp_path), True)   # safe_serialization=True (safetensors)
    assert t.saved == str(tmp_path)


def test_model_card_documents_base_method_and_metrics():
    card = build_model_card(
        base_model="meta-llama/Llama-3.1-8B-Instruct",
        variant="preserving",
        preserve=["negation", "agentic"],
        metrics={"refusal_rate": 0.12, "kl": 0.18, "agentic_score": 0.88},
    )
    assert "meta-llama/Llama-3.1-8B-Instruct" in card
    assert "preserving" in card
    assert "negation" in card and "agentic" in card
    assert "0.12" in card and "0.88" in card
    assert "dual-use" in card.lower()  # cadre responsable (CLAUDE.md)
