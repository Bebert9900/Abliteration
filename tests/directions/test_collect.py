"""Tests de collecte d'activations : pooling dernier token + moyenne par couche."""
import torch

from src.directions import collect_means, pooled_last_token


def test_pooled_last_token_selects_last_real_position_per_layer():
    # (L+1=2, B=2, T=3, H=2)
    hs = torch.zeros(2, 2, 3, 2)
    hs[0, 0, 2] = torch.tensor([1.0, 1.0])   # couche0, seq0, dernier token
    hs[0, 1, 2] = torch.tensor([3.0, 3.0])   # couche0, seq1, dernier token
    hs[1, 0, 2] = torch.tensor([5.0, 5.0])   # couche1, seq0
    hs[1, 1, 2] = torch.tensor([7.0, 7.0])   # couche1, seq1
    mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
    pooled = pooled_last_token(hs, mask)      # (L+1, B, H)
    assert pooled.shape == (2, 2, 2)
    assert torch.allclose(pooled[0, 0], torch.tensor([1.0, 1.0]))
    assert torch.allclose(pooled[1, 1], torch.tensor([7.0, 7.0]))


class FakeFormatter:
    def tokenize(self, texts):
        # left-padded ; ids encodent une valeur, pad = 0
        ids = torch.tensor([[5, 6, 7], [0, 8, 9]])
        mask = torch.tensor([[1, 1, 1], [0, 1, 1]])
        return {"input_ids": ids, "attention_mask": mask}


class FakeOutput:
    def __init__(self, hidden_states):
        self.hidden_states = hidden_states


class FakeModel:
    """hidden_states[l][b,t,:] = id(b,t) * (l+1), H=4, 2 couches."""

    def __call__(self, input_ids, attention_mask, output_hidden_states):
        H = 4
        base = input_ids.float().unsqueeze(-1).expand(-1, -1, H)  # (B,T,H)
        return FakeOutput((base, base * 2))


def test_collect_means_averages_last_token_over_batch_per_layer():
    means = collect_means(FakeModel(), FakeFormatter(), ["a", "b"], batch_size=8)
    assert means.shape == (2, 4)  # (L+1, H)
    # dernier token : seq0->7, seq1->9 ; moyenne 8 (couche0), 16 (couche1)
    assert torch.allclose(means[0], torch.full((4,), 8.0))
    assert torch.allclose(means[1], torch.full((4,), 16.0))
