"""Tests IO : sauvegarde du modèle et génération de model card transparente."""
from abliteration.io import build_model_card, save_model


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


def test_model_card_renders_report_as_bi_axis_table():
    from abliteration.eval import EvalReport
    report = EvalReport(refusal_rate=0.05, kl=0.2, negation_retention=0.9, follow_rate=0.1,
                        empty_rate=0.0, agentic_score=0.85, degeneracy_rate=0.0)
    card = build_model_card("base/model", "norm_preserving_biprojected",
                            preserve=["negation", "agentic"], metrics={}, report=report,
                            run_config={"selected_layer": 14, "alpha": 0.8, "seed": 0})
    assert "| Axe | Métrique | Valeur |" in card        # tableau bi-axe
    assert "Taux de refus" in card and "0.0500" in card
    assert "Score agentique" in card and "0.8500" in card
    assert "Couche sélectionnée : 14" in card
    assert "Force d'ablation (alpha) : 0.8" in card
    assert "## Configuration reproductible" in card     # config repro embarquée
    assert '"seed": 0' in card
