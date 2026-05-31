"""Modèle jouet partagé par les tests circuits : decoder-only Llama-like minimal mais RÉEL.

Structure (`model.layers[i].self_attn.o_proj`, `...mlp.down_proj`, `get_input_embeddings`)
comprise telle quelle par `ArchAdapter` → le backend l'introspecte exactement comme un vrai
modèle HF. L'« attention » est factice (pas de softmax/causalité) : on teste la PLOMBERIE des
composants et la MATH causale, pas la qualité d'attention.

Importé en module frère (pytest mode "prepend" met le dossier du test sur sys.path), conforme
à la convention du repo (aucun __init__.py dans tests/).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Cfg:
    def __init__(self, hidden, n_heads, vocab, n_layers):
        self.hidden_size = hidden
        self.num_attention_heads = n_heads
        self.head_dim = hidden // n_heads
        self.vocab_size = vocab
        self.num_hidden_layers = n_layers


class ToyAttn(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = hidden // n_heads
        self.v_proj = nn.Linear(hidden, hidden, bias=False)
        self.o_proj = nn.Linear(hidden, hidden, bias=False)

    def forward(self, x):
        v = self.v_proj(x)
        z = v.cumsum(dim=1) / torch.arange(1, x.size(1) + 1, device=x.device).view(1, -1, 1)
        return self.o_proj(z)


class ToyMLP(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.up_proj = nn.Linear(hidden, 4 * hidden, bias=False)
        self.down_proj = nn.Linear(4 * hidden, hidden, bias=False)

    def forward(self, x):
        return self.down_proj(torch.relu(self.up_proj(x)))


class ToyBlock(nn.Module):
    def __init__(self, hidden, n_heads):
        super().__init__()
        self.self_attn = ToyAttn(hidden, n_heads)
        self.mlp = ToyMLP(hidden)

    def forward(self, x):
        x = x + self.self_attn(x)
        x = x + self.mlp(x)
        return x


class ToyOutput:
    def __init__(self, logits, hidden_states):
        self.logits = logits
        self.hidden_states = hidden_states


class ToyModel(nn.Module):
    def __init__(self, hidden=8, n_heads=2, vocab=16, n_layers=3):
        super().__init__()
        self.config = Cfg(hidden, n_heads, vocab, n_layers)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        self.model.layers = nn.ModuleList([ToyBlock(hidden, n_heads) for _ in range(n_layers)])
        self.model.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        x = self.model.embed_tokens(input_ids)
        hidden_states = [x]
        for blk in self.model.layers:
            x = blk(x)
            hidden_states.append(x)
        x = self.model.norm(x)
        logits = self.lm_head(x)
        return ToyOutput(logits, tuple(hidden_states) if output_hidden_states else None)


def make_model(seed: int = 0, **kwargs) -> ToyModel:
    torch.manual_seed(seed)
    return ToyModel(**kwargs)


def ids():
    return torch.tensor([[1, 5, 3, 9]])


# --------------------------------------------------------------------------- #
# Modèle CONTRÔLABLE : un seul composant porte causalement le "refus".
#
# Construit pour que les tests de patching aient des valeurs EXACTES attendues.
# Géométrie : hidden=4, 2 têtes (head_dim=2), 1 couche, vocab=2.
# Direction de refus r = e0 = [1,0,0,0].
#
# - token 1 = "harmful" → embed = [0,1,0,0] (feature en dim1, ORTHOGONALE à r).
# - token 0 = "harmless" → embed = [0,0,0,0].
# - v_proj route la feature dim1 vers la tranche z de la tête 0 (z[0] = x[1]).
# - o_proj : tête 0 mappe sa tranche vers r (dim0) ; tête 1 vers dim3 (bruit, ⊥ r).
# - MLP nul (poids 0) → n'interfère pas.
#
# Conséquence (projection du résidu final sur r) :
#   harmful  → métrique = 1   (tête 0 écrit 1·r ; l'embed ne contribue pas à r)
#   harmless → métrique = 0
# La tête 0 explique donc TOUT l'écart : necessity/sufficiency recovery == 1.0,
# tandis que la tête 1 (bruit) a recovery == 0.0. Le MLP aussi.
# --------------------------------------------------------------------------- #
class ControllableModel(nn.Module):
    def __init__(self):
        super().__init__()
        hidden, n_heads, head_dim, vocab = 4, 2, 2, 2
        self.config = Cfg(hidden, n_heads, vocab, 1)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(vocab, hidden)
        block = ToyBlock(hidden, n_heads)
        self.model.layers = nn.ModuleList([block])
        self.model.norm = nn.Identity()
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

        with torch.no_grad():
            # embed : harmless=0, harmful=e1 (⊥ r=e0)
            self.model.embed_tokens.weight.zero_()
            self.model.embed_tokens.weight[1, 1] = 1.0

            # attention sans mixage de positions : on remplace par une attn "identité"
            attn = block.self_attn
            # v_proj : z[0] = x[1]  (route la feature harmful vers la tranche tête 0)
            attn.v_proj.weight.zero_()
            attn.v_proj.weight[0, 1] = 1.0
            # o_proj reshape (hidden=4, n_heads=2, head_dim=2)
            # tête 0 : z[0:2] -> dim0 (=r) ; tête 1 : z[2:4] -> dim3 (bruit)
            W = torch.zeros(hidden, n_heads, head_dim)
            W[0, 0, 0] = 1.0     # head0, head_dim0 -> hidden dim0 (r)
            W[3, 1, 1] = 1.0     # head1, head_dim1 -> hidden dim3 (noise)
            attn.o_proj.weight.copy_(W.reshape(hidden, n_heads * head_dim))

            # MLP nul
            block.mlp.up_proj.weight.zero_()
            block.mlp.down_proj.weight.zero_()

            self.lm_head.weight.zero_()
            self.lm_head.weight[0, 0] = 1.0   # logit 0 lit la direction r

        # attention identité (pas de cumsum) : z = v_proj(x) position par position.
        # IMPORTANT : on appelle o_proj comme MODULE (pas F.linear) pour que les hooks du
        # backend (pré/post sur o_proj) se déclenchent → décomposition par tête disponible.
        def identity_attn(x, _attn=attn):
            z = _attn.v_proj(x)
            return _attn.o_proj(z)

        attn.forward = identity_attn

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(self, input_ids, attention_mask=None, output_hidden_states=False):
        x = self.model.embed_tokens(input_ids)
        hidden_states = [x]
        for blk in self.model.layers:
            x = blk(x)
            hidden_states.append(x)
        logits = self.lm_head(self.model.norm(x))
        return ToyOutput(logits, tuple(hidden_states) if output_hidden_states else None)


# direction de refus de ControllableModel
def controllable_refusal_dir():
    return torch.tensor([1.0, 0.0, 0.0, 0.0])


def harmful_ids():     # "clean" : déclenche le refus
    return torch.tensor([[1]])


def harmless_ids():    # "corrupted" : ne déclenche pas le refus
    return torch.tensor([[0]])
