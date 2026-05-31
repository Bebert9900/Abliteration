"""Tests de l'orthogonalisation de poids (Linear, Conv1D, embedding) + norm-preserving."""
import torch
import torch.nn as nn

from src.models import ArchAdapter, WriteKind
from src.ablation import orthogonalize_weights


class Conv1D(nn.Module):
    def __init__(self, nx, nf):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(nx, nf))  # (in, out) transposé
        self.bias = nn.Parameter(torch.zeros(nf))


class TinyModel(nn.Module):
    def __init__(self, h=8, layers=2):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(20, h)
        self.model.layers = nn.ModuleList()
        for _ in range(layers):
            b = nn.Module()
            b.self_attn = nn.Module()
            b.self_attn.o_proj = nn.Linear(h, h, bias=False)
            b.mlp = nn.Module()
            b.mlp.down_proj = nn.Linear(h, h, bias=False)
            self.model.layers.append(b)

    def get_input_embeddings(self):
        return self.model.embed_tokens


def test_orthogonalize_kills_refusal_component_in_all_writers():
    torch.manual_seed(0)
    model = TinyModel(h=8)
    r = torch.randn(8)
    r = r / r.norm()
    orthogonalize_weights(ArchAdapter(model), r, norm_preserve=False)

    for name, p in model.named_parameters():
        W = p.data
        if W.ndim != 2:
            continue
        # La sortie (le long du residual stream) ne doit plus avoir de composante selon r.
        if W.shape[0] == 8 and "embed" not in name:   # Linear (out=8, in)
            assert torch.allclose(r @ W, torch.zeros(W.shape[1]), atol=1e-5)
        else:                                          # embedding (vocab, 8): sortie = colonnes
            assert torch.allclose(W @ r, torch.zeros(W.shape[0]), atol=1e-5)


def test_orthogonalize_handles_conv1d_axes():
    torch.manual_seed(1)
    model = TinyModel(h=8, layers=1)
    model.model.layers[0].self_attn.o_proj = Conv1D(8, 8)  # sortie = colonnes
    r = torch.randn(8)
    r = r / r.norm()
    orthogonalize_weights(ArchAdapter(model), r, norm_preserve=False)
    W = model.model.layers[0].self_attn.o_proj.weight.data  # (in, out)
    assert torch.allclose(W @ r, torch.zeros(8), atol=1e-5)


def test_norm_preserve_restores_slice_norms():
    torch.manual_seed(2)
    model = TinyModel(h=8, layers=1)
    before = model.model.layers[0].mlp.down_proj.weight.data.norm(dim=0).clone()
    r = torch.randn(8)
    r = r / r.norm()
    orthogonalize_weights(ArchAdapter(model), r, norm_preserve=True)
    after = model.model.layers[0].mlp.down_proj.weight.data.norm(dim=0)
    assert torch.allclose(before, after, atol=1e-4)
