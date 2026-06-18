"""Tests de l'ArchAdapter : découverte des écritures au residual stream sans noms en dur."""
import torch
import torch.nn as nn

from meridian.models import ArchAdapter, WriteKind


class Conv1D(nn.Module):
    """Mime transformers.Conv1D (GPT-2) : poids transposé (in, out) vs nn.Linear (out, in)."""

    def __init__(self, nf, nx):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))


def _dense_block(h):
    block = nn.Module()
    block.self_attn = nn.Module()
    block.self_attn.o_proj = nn.Linear(h, h, bias=False)
    block.mlp = nn.Module()
    block.mlp.down_proj = nn.Linear(h, h, bias=False)
    return block


class DenseModel(nn.Module):
    def __init__(self, h=8, layers=3):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(100, h)
        self.model.layers = nn.ModuleList([_dense_block(h) for _ in range(layers)])

    def get_input_embeddings(self):
        return self.model.embed_tokens


class MoEModel(nn.Module):
    """Chaque couche a N experts, chacun son down_proj, + une couche partagée."""

    def __init__(self, h=8, layers=2, experts=4):
        super().__init__()
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(100, h)
        self.model.layers = nn.ModuleList()
        for _ in range(layers):
            block = nn.Module()
            block.self_attn = nn.Module()
            block.self_attn.o_proj = nn.Linear(h, h, bias=False)
            block.mlp = nn.Module()
            block.mlp.experts = nn.ModuleList()
            for _ in range(experts):
                exp = nn.Module()
                exp.down_proj = nn.Linear(h, h, bias=False)
                block.mlp.experts.append(exp)
            block.mlp.shared_expert = nn.Module()
            block.mlp.shared_expert.down_proj = nn.Linear(h, h, bias=False)
            self.model.layers.append(block)

    def get_input_embeddings(self):
        return self.model.embed_tokens


def test_dense_finds_attn_mlp_and_embedding():
    adapter = ArchAdapter(DenseModel(layers=3))
    writers = adapter.residual_writers()
    kinds = [w.kind for w in writers]
    assert kinds.count(WriteKind.ATTENTION_OUT) == 3
    assert kinds.count(WriteKind.MLP_OUT) == 3
    assert kinds.count(WriteKind.EMBEDDING) == 1


def test_no_module_name_hardcoded_uses_get_input_embeddings():
    adapter = ArchAdapter(DenseModel())
    emb = [w for w in adapter.residual_writers() if w.kind is WriteKind.EMBEDDING]
    assert len(emb) == 1
    assert emb[0].module is adapter.model.get_input_embeddings()


def test_moe_collects_every_expert_down_proj_plus_shared():
    adapter = ArchAdapter(MoEModel(layers=2, experts=4))
    mlp_out = [w for w in adapter.residual_writers() if w.kind is WriteKind.MLP_OUT]
    # 2 couches * (4 experts + 1 partagé) = 10
    assert len(mlp_out) == 10


def test_conv1d_is_flagged():
    model = DenseModel(h=8, layers=1)
    model.model.layers[0].self_attn.o_proj = Conv1D(8, 8)  # remplace par un Conv1D
    adapter = ArchAdapter(model)
    attn = [w for w in adapter.residual_writers() if w.kind is WriteKind.ATTENTION_OUT][0]
    assert attn.is_conv1d is True
    mlp = [w for w in adapter.residual_writers() if w.kind is WriteKind.MLP_OUT][0]
    assert mlp.is_conv1d is False
