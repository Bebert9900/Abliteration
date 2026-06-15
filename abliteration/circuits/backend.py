"""Backend d'introspection par composant (têtes d'attention, MLP) — Phase 1.

Décision du skill `abliteration-circuits` : le backend de référence est **NNsight / nnterp**
(pas TransformerLens), car NNsight préserve l'implémentation HuggingFace EXACTE — on analyse
exactement les poids qu'on ablate, au lieu d'une réimplémentation (raison du rejet de
TransformerLens).

On expose une interface `CircuitBackend` minimale et DEUX implémentations :

- `NNsightBackend` — backend de production (import paresseux ; échoue BRUYAMMENT si nnsight est
  absent ou incompatible). C'est lui qu'utilise `analyze-circuit <model>`.
- `TorchHookBackend` — hooks `torch` posés sur les MODULES HF RÉELS (réutilise `ArchAdapter`
  pour trouver `o_proj`/`down_proj` sans coder les noms en dur). Opère donc sur les poids HF
  exacts, comme NNsight, mais sans dépendance externe → rend la MATH de DLA/patching testable
  hors-ligne sur des modèles jouets (discipline TDD du repo).

Les modules en aval (`dla`, `patching`, `attribution`, `localize`) ne dépendent QUE de
l'interface : ils sont donc validés hors-ligne avec `TorchHookBackend`, et tournent en prod
avec l'un ou l'autre.

Décomposition par tête (nn.Linear `o_proj`, forme (hidden, n_heads·head_dim)) :
    z              = entrée de o_proj           (batch, seq, n_heads·head_dim)
    contrib_tête_h = z[..., h] @ W_O[:, h]^T    (batch, seq, hidden)
    sortie o_proj  = Σ_h contrib_tête_h (+ biais)
C'est la « contribution au residual stream » de chaque tête (résidu-espace), entrée de la DLA.
Conv1D (GPT-2, axes transposés) n'est PAS géré pour la découpe par tête v1 → échec explicite.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch

from abliteration.models import ArchAdapter, WriteKind


class ComponentKind(str, Enum):
    ATTN_HEAD = "attn_head"
    MLP = "mlp"


@dataclass(frozen=True)
class Component:
    """Identité d'un composant analysable. `head` est None pour un MLP."""
    kind: ComponentKind
    layer: int
    head: int | None = None

    @property
    def label(self) -> str:
        if self.kind is ComponentKind.ATTN_HEAD:
            return f"L{self.layer}H{self.head}"
        return f"L{self.layer}MLP"

    def __repr__(self) -> str:  # lisibilité dans les rapports/tests
        return self.label


@dataclass(frozen=True)
class ModelInfo:
    n_layers: int
    n_heads: int
    head_dim: int
    hidden_size: int
    device: torch.device
    dtype: torch.dtype


@dataclass
class ComponentCache:
    """Contributions residual-espace capturées sur un forward, + logits/résidu final.

    attn_head_out[layer] : (batch, seq, n_heads, hidden)  — contribution de CHAQUE tête.
    mlp_out[layer]       : (batch, seq, hidden)           — contribution du MLP (sortie down_proj).
    final_resid          : (batch, seq, hidden)           — dernier hidden state (pré-unembed).
    logits               : (batch, seq, vocab).
    """
    attn_head_out: dict[int, torch.Tensor] = field(default_factory=dict)
    mlp_out: dict[int, torch.Tensor] = field(default_factory=dict)
    final_resid: torch.Tensor | None = None
    logits: torch.Tensor | None = None

    def component(self, c: Component) -> torch.Tensor:
        """Contribution residual-espace d'un composant : (batch, seq, hidden)."""
        if c.kind is ComponentKind.MLP:
            return self.mlp_out[c.layer]
        return self.attn_head_out[c.layer][:, :, c.head, :]


def decompose_heads(o_proj_weight: torch.Tensor, z: torch.Tensor,
                    n_heads: int, head_dim: int, hidden: int) -> torch.Tensor:
    """z (b,s,n_heads·head_dim) + W_O (hidden, n_heads·head_dim) -> (b,s,n_heads,hidden).

    Contribution residual-espace de chaque tête : Σ_d z[...,h,d]·W_O[:,h,d]. La somme sur les
    têtes égale exactement la sortie de o_proj (hors biais). Géométrie lue depuis la config :
    gère head_dim ≠ hidden/n_heads (cas Qwen3 : 16×128=2048 ≠ hidden=1024).
    """
    W = o_proj_weight
    zf = z.to(W.dtype)
    b, s, _ = zf.shape
    z_heads = zf.reshape(b, s, n_heads, head_dim)
    W_heads = W.reshape(hidden, n_heads, head_dim)
    return torch.einsum("bshd,fhd->bshf", z_heads, W_heads)


@dataclass(frozen=True)
class Patch:
    """Remplace la contribution residual-espace de `component` par `value` (batch, seq, hidden).

    `value` provient typiquement d'un autre run (counterfactual) → activation patching.
    """
    component: Component
    value: torch.Tensor


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class CircuitBackend:
    """Interface commune. Les sous-classes implémentent run_with_cache / run_with_patches."""

    info: ModelInfo

    def all_components(self, include_mlp: bool = True) -> list[Component]:
        comps: list[Component] = []
        for l in range(self.info.n_layers):
            for h in range(self.info.n_heads):
                comps.append(Component(ComponentKind.ATTN_HEAD, l, h))
            if include_mlp:
                comps.append(Component(ComponentKind.MLP, l))
        return comps

    def run_with_cache(self, input_ids, attention_mask=None) -> ComponentCache:  # pragma: no cover
        raise NotImplementedError

    def run_with_patches(self, input_ids, attention_mask, patches: list[Patch]) -> ComponentCache:  # pragma: no cover
        raise NotImplementedError


# --------------------------------------------------------------------------- #
# Backend torch (hooks sur les modules HF réels) — testable hors-ligne
# --------------------------------------------------------------------------- #
def _layer_index(module_name: str) -> int | None:
    """Extrait l'indice de couche d'un nom type 'model.layers.12.self_attn.o_proj'."""
    parts = module_name.split(".")
    for i, p in enumerate(parts):
        if p in ("layers", "h", "blocks") and i + 1 < len(parts) and parts[i + 1].isdigit():
            return int(parts[i + 1])
    return None


class TorchHookBackend(CircuitBackend):
    """Hooks forward sur o_proj (par tête) et down_proj (MLP) des modules HF réels."""

    def __init__(self, model, adapter: ArchAdapter | None = None, config=None):
        self.model = model
        adapter = adapter or ArchAdapter(model)
        cfg = config if config is not None else getattr(model, "config", None)
        if cfg is None:
            raise ValueError("TorchHookBackend requiert un model.config (ou config=...).")

        self._o_proj: dict[int, torch.nn.Module] = {}
        self._down_proj: dict[int, torch.nn.Module] = {}
        for w in adapter.residual_writers():
            if w.kind is WriteKind.EMBEDDING:
                continue
            if w.is_conv1d:
                raise NotImplementedError(
                    f"Découpe par tête non gérée pour Conv1D ({w.name}). "
                    "v1 cible les archis denses type Llama (nn.Linear)."
                )
            li = _layer_index(w.name)
            if li is None:
                continue
            if w.kind is WriteKind.ATTENTION_OUT:
                self._o_proj[li] = w.module
            elif w.kind is WriteKind.MLP_OUT:
                self._down_proj[li] = w.module

        n_layers = len(self._o_proj)
        n_heads = int(getattr(cfg, "num_attention_heads"))
        hidden = int(getattr(cfg, "hidden_size"))
        head_dim = int(getattr(cfg, "head_dim", hidden // n_heads))
        p = next(model.parameters())
        self.info = ModelInfo(n_layers, n_heads, head_dim, hidden, p.device, p.dtype)

    # -- décomposition par tête à partir de z (entrée de o_proj) -------------- #
    def _heads_from_z(self, layer: int, z: torch.Tensor) -> torch.Tensor:
        """z (batch, seq, n_heads·head_dim) -> contributions (batch, seq, n_heads, hidden)."""
        return decompose_heads(self._o_proj[layer].weight, z,
                               self.info.n_heads, self.info.head_dim, self.info.hidden_size)

    @torch.no_grad()
    def run_with_cache(self, input_ids, attention_mask=None) -> ComponentCache:
        return self._run(input_ids, attention_mask, patches=None)

    @torch.no_grad()
    def run_with_patches(self, input_ids, attention_mask, patches: list[Patch]) -> ComponentCache:
        return self._run(input_ids, attention_mask, patches=patches)

    def _run(self, input_ids, attention_mask, patches):
        cache = ComponentCache()
        captured_z: dict[int, torch.Tensor] = {}
        handles = []

        # index des patches par (kind, layer[, head])
        attn_patch: dict[tuple[int, int], torch.Tensor] = {}
        mlp_patch: dict[int, torch.Tensor] = {}
        for pt in patches or []:
            c = pt.component
            if c.kind is ComponentKind.ATTN_HEAD:
                attn_patch[(c.layer, c.head)] = pt.value
            else:
                mlp_patch[c.layer] = pt.value

        def make_attn_hooks(layer):
            def pre_hook(module, args, kwargs):
                z = args[0] if args else kwargs.get("input")
                captured_z[layer] = z.detach()
                return None

            def post_hook(module, args, output):
                z = captured_z[layer]
                heads = self._heads_from_z(layer, z)           # (b,s,nh,hidden)
                cache.attn_head_out[layer] = heads.detach()
                out = output[0] if isinstance(output, tuple) else output
                # applique les patches de têtes : out += Σ (value - contrib_orig)
                for (pl, ph), val in attn_patch.items():
                    if pl != layer:
                        continue
                    delta = (val.to(out.dtype) - heads[:, :, ph, :].to(out.dtype))
                    out = out + delta
                    heads[:, :, ph, :] = val.to(heads.dtype)   # cache reflète le patch
                cache.attn_head_out[layer] = heads.detach()
                return (out, *output[1:]) if isinstance(output, tuple) else out

            return pre_hook, post_hook

        def make_mlp_hook(layer):
            def post_hook(module, args, output):
                out = output[0] if isinstance(output, tuple) else output
                if layer in mlp_patch:
                    out = mlp_patch[layer].to(out.dtype)
                cache.mlp_out[layer] = out.detach()
                return (out, *output[1:]) if isinstance(output, tuple) else out

            return post_hook

        for layer, mod in self._o_proj.items():
            pre, post = make_attn_hooks(layer)
            handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
            handles.append(mod.register_forward_hook(post))
        for layer, mod in self._down_proj.items():
            handles.append(mod.register_forward_hook(make_mlp_hook(layer)))

        try:
            kw = {"output_hidden_states": True}
            if attention_mask is not None:
                kw["attention_mask"] = attention_mask
            out = self.model(input_ids=input_ids, **kw)
        finally:
            for h in handles:
                h.remove()

        cache.logits = out.logits.detach()
        hs = getattr(out, "hidden_states", None)
        cache.final_resid = hs[-1].detach() if hs is not None else None
        return cache


# --------------------------------------------------------------------------- #
# Backend NNsight (production) — import paresseux, échec bruyant
# --------------------------------------------------------------------------- #
class NNsightBackend(CircuitBackend):
    """Backend de production via nnsight/nnterp. NON câblé tant que la dépendance n'est pas
    validée sur l'environnement (py3.14 / torch 2.10 / transformers 5.9, à confirmer).

    Échoue BRUYAMMENT plutôt que de fournir une analyse silencieusement fausse (règle du repo).
    """

    def __init__(self, model, *args, **kwargs):
        try:
            import nnsight  # noqa: F401
        except Exception as e:  # pragma: no cover - dépend de l'env
            raise NotImplementedError(
                "NNsightBackend requiert `nnsight`/`nnterp` (cf. skill abliteration-circuits). "
                f"Import impossible : {e!r}. Installer `nnsight nnterp` puis câbler ce backend, "
                "ou utiliser TorchHookBackend (même introspection sur les poids HF exacts)."
            ) from e
        raise NotImplementedError(
            "NNsightBackend : câblage nnsight à finaliser une fois la compat env confirmée. "
            "TorchHookBackend couvre l'introspection par composant en attendant."
        )


# --------------------------------------------------------------------------- #
# Fabrique
# --------------------------------------------------------------------------- #
def make_backend(model, adapter: ArchAdapter | None = None, prefer: str = "auto") -> CircuitBackend:
    """Construit un backend. prefer ∈ {auto, nnsight, torch}.

    - "torch"   : TorchHookBackend (hors-ligne, poids HF exacts).
    - "nnsight" : NNsightBackend (échoue si indisponible).
    - "auto"    : NNsight si disponible et câblé, sinon repli torch (loggé par l'appelant).
    """
    if prefer == "torch":
        return TorchHookBackend(model, adapter)
    if prefer == "nnsight":
        return NNsightBackend(model)
    try:
        return NNsightBackend(model)
    except NotImplementedError:
        return TorchHookBackend(model, adapter)
