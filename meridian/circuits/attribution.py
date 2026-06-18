"""Attribution patching — méthode 3/3, approximation GRADIENT scalable.

Linéarisation au 1er ordre de l'activation patching (Nanda 2023, « activation patching at
industrial scale », circuit_analysis.md) : au lieu d'un forward par site, on estime l'effet de
TOUS les sites en ~2-3 passes.

Principe : l'effet de remplacer la contribution `a` d'un composant par `a'` (du run corrompu)
sur une métrique `m` s'approxime par
    Δm ≈ ⟨ ∂m/∂a , (a' − a) ⟩
le gradient étant pris au point `a` (run clean). Une seule rétropropagation de `m` donne
∂m/∂a pour TOUS les composants simultanément → scalable.

⚠️ L'approximation DÉCROCHE sur les gros effets (non-linéarités, saturation). Donc :
- on l'utilise pour SCORER/CLASSER vite tous les sites ;
- on CONTRE-VÉRIFIE les top sites par patching exact (`patching.py`) — c'est non négociable
  (piège documenté dans le skill).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from meridian.data.formatting import last_token_index

from .backend import Component, TorchHookBackend

ATTRIBUTION_CAVEAT = (
    "APPROXIMATION GRADIENT (1er ordre) — décroche sur les gros effets. Sert à classer vite ; "
    "les top sites DOIVENT être contre-vérifiés par patching exact (patching.py) avant conclusion."
)


@dataclass
class AttributionResult:
    """Scores d'attribution approximés par composant (Δm estimé clean→corrupted)."""
    scores: dict[Component, float]
    method: str = "attribution_patching"
    caveat: str = field(default=ATTRIBUTION_CAVEAT)

    def ranked(self, by_abs: bool = True) -> list[tuple[Component, float]]:
        key = (lambda kv: abs(kv[1])) if by_abs else (lambda kv: kv[1])
        return sorted(self.scores.items(), key=key, reverse=True)

    def top(self, k: int, by_abs: bool = True) -> list[tuple[Component, float]]:
        return self.ranked(by_abs=by_abs)[:k]


def _metric_from_resid_logits(final_resid, logits, attention_mask, metric_dir, refusal_token,
                              answer_token):
    """Métrique différentiable (garde le graphe) : logit-diff si tokens, sinon proj sur r̂."""
    b = logits.shape[0]
    s = logits.shape[1]
    idx = last_token_index(attention_mask) if attention_mask is not None \
        else torch.full((b,), s - 1, dtype=torch.long)
    batch = torch.arange(b)
    if refusal_token is not None:
        last = logits[batch, idx, :]
        return (last[:, refusal_token] - last[:, answer_token]).mean()
    r = metric_dir.to(final_resid.dtype)
    r = r / (r.norm() + 1e-8)
    return (final_resid[batch, idx, :] @ r).mean()


def attribution_patching(
    backend: TorchHookBackend,
    clean_ids: torch.Tensor,
    corrupted_ids: torch.Tensor,
    *,
    refusal_dir: torch.Tensor | None = None,
    refusal_token: int | None = None,
    answer_token: int | None = None,
    clean_mask: torch.Tensor | None = None,
    corrupted_mask: torch.Tensor | None = None,
    include_mlp: bool = True,
) -> AttributionResult:
    """Estime Δmétrique pour tous les composants en une rétropropagation (gradient × écart).

    Δm_c ≈ ⟨ ∂m/∂a_c |clean , (a_c^corrupted − a_c^clean) ⟩, au dernier token.
    Requiert un backend torch (gradient) — c'est l'usage normal hors-ligne ; en prod NNsight
    expose aussi les gradients (.grad), câblage ultérieur.
    """
    if refusal_dir is None and refusal_token is None:
        raise ValueError("attribution_patching requiert refusal_dir ou (refusal_token, answer_token).")

    # 1) run corrompu (sans grad) : valeurs de référence des contributions au dernier token
    corr_cache = backend.run_with_cache(corrupted_ids, corrupted_mask)

    # 2) run clean AVEC capture des activations gardant le graphe (grad activé)
    grads, clean_vals = _clean_run_with_grads(
        backend, clean_ids, clean_mask, refusal_dir, refusal_token, answer_token, include_mlp
    )

    # position du dernier token (clean / corrupted)
    def last_idx(ids, mask):
        b, s = ids.shape[0], ids.shape[1]
        return (last_token_index(mask) if mask is not None
                else torch.full((b,), s - 1, dtype=torch.long))

    ci = last_idx(corrupted_ids, corrupted_mask)
    cb = torch.arange(corrupted_ids.shape[0])

    scores: dict[Component, float] = {}
    for c in backend.all_components(include_mlp=include_mlp):
        a_clean = clean_vals[c]                                   # (b, hidden) au dernier token
        corr_contrib = corr_cache.component(c)[cb, ci, :].to(a_clean.dtype)
        delta_input = (corr_contrib - a_clean)                   # a' − a
        g = grads[c]                                             # ∂m/∂a au dernier token
        scores[c] = float((g * delta_input).sum(dim=-1).mean())

    return AttributionResult(scores=scores)


def aggregate_attribution(
    backend: TorchHookBackend,
    pairs: list[tuple],
    *,
    refusal_dir=None,
    refusal_token: int | None = None,
    answer_token: int | None = None,
    include_mlp: bool = True,
) -> AttributionResult:
    """Attribution moyennée sur TOUTES les paires (corrige RC1).

    Le scan de candidats ne doit pas dépendre d'une seule paire : `attribution(pairs[0])` donne
    un univers qui varie fortement selon la paire (mesuré : Jaccard ~0.37 entre paires). On
    moyenne donc le score d'attribution de chaque composant sur l'ensemble des paires →
    classement invariant à l'ordre et représentatif du corpus.

    `pairs` : liste de (clean_ids, corrupted_ids, clean_mask, corrupted_mask).
    `refusal_dir` : tenseur (hidden,) partagé OU liste (une direction par paire).
    """
    if not pairs:
        raise ValueError("aggregate_attribution requiert au moins une paire.")

    def dir_for(i):
        if refusal_dir is None or isinstance(refusal_dir, torch.Tensor):
            return refusal_dir
        return refusal_dir[i]

    totals: dict[Component, float] = {}
    for i, (cids, corr_ids, cmask, corrmask) in enumerate(pairs):
        res = attribution_patching(
            backend, cids, corr_ids, refusal_dir=dir_for(i),
            refusal_token=refusal_token, answer_token=answer_token,
            clean_mask=cmask, corrupted_mask=corrmask, include_mlp=include_mlp,
        )
        for c, s in res.scores.items():
            totals[c] = totals.get(c, 0.0) + s

    n = len(pairs)
    return AttributionResult(scores={c: v / n for c, v in totals.items()})


def _clean_run_with_grads(backend, clean_ids, clean_mask, refusal_dir, refusal_token,
                          answer_token, include_mlp):
    """Forward clean en gardant le graphe ; rétroprop la métrique ; renvoie (gradients, valeurs)
    au dernier token par composant.

    Astuce exacte : la sortie de o_proj est la SOMME des contributions de têtes
    (out = Σ_h head_h). Le residual stream en aval ne voit que `out`, donc
        ∂m/∂head_h = ∂m/∂out   (identique pour toutes les têtes de la couche).
    On capture donc le gradient sur la SORTIE du module (qui est dans le graphe), et on le
    réutilise pour chaque tête ; les VALEURS par tête viennent de la décomposition de z.
    """
    model = backend.model
    info = backend.info

    captured_z: dict[int, torch.Tensor] = {}
    attn_out: dict[int, torch.Tensor] = {}     # sortie o_proj (retain_grad)
    mlp_out: dict[int, torch.Tensor] = {}      # sortie down_proj (retain_grad)
    handles = []

    def make_attn_hooks(layer):
        def pre_hook(module, args, kwargs):
            z = args[0] if args else kwargs.get("input")
            captured_z[layer] = z.detach()
            return None

        def post_hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            out.retain_grad()
            attn_out[layer] = out
            return output

        return pre_hook, post_hook

    def make_mlp_hook(layer):
        def post_hook(module, args, output):
            out = output[0] if isinstance(output, tuple) else output
            out.retain_grad()
            mlp_out[layer] = out
            return output

        return post_hook

    for layer, mod in backend._o_proj.items():
        pre, post = make_attn_hooks(layer)
        handles.append(mod.register_forward_pre_hook(pre, with_kwargs=True))
        handles.append(mod.register_forward_hook(post))
    if include_mlp:
        for layer, mod in backend._down_proj.items():
            handles.append(mod.register_forward_hook(make_mlp_hook(layer)))

    try:
        kw = {"output_hidden_states": True}
        if clean_mask is not None:
            kw["attention_mask"] = clean_mask
        out = model(input_ids=clean_ids, **kw)
        m = _metric_from_resid_logits(
            out.hidden_states[-1], out.logits, clean_mask, refusal_dir, refusal_token, answer_token
        )
        model.zero_grad(set_to_none=True)
        m.backward()
    finally:
        for h in handles:
            h.remove()

    b, s = clean_ids.shape[0], clean_ids.shape[1]
    idx = (last_token_index(clean_mask) if clean_mask is not None
           else torch.full((b,), s - 1, dtype=torch.long))
    batch = torch.arange(b)

    grads: dict[Component, torch.Tensor] = {}
    vals: dict[Component, torch.Tensor] = {}
    from .backend import Component as C
    from .backend import ComponentKind as K
    for layer in backend._o_proj:
        g_out = attn_out[layer].grad[batch, idx, :].detach()      # (b, hidden) — partagé
        heads = backend._heads_from_z(layer, captured_z[layer])   # valeurs par tête
        for h in range(info.n_heads):
            comp = C(K.ATTN_HEAD, layer, h)
            grads[comp] = g_out
            vals[comp] = heads[batch, idx, h, :].detach()
    if include_mlp:
        for layer in backend._down_proj:
            comp = C(K.MLP, layer)
            mo = mlp_out[layer]
            grads[comp] = mo.grad[batch, idx, :].detach()
            vals[comp] = mo[batch, idx, :].detach()
    return grads, vals


def agreement_with_exact(approx: AttributionResult, exact_scores: dict[Component, float],
                         k: int) -> float:
    """Fraction des top-k de l'attribution présents dans les top-k exacts (contre-vérification).

    Sert à dire honnêtement à quel point l'approximation tient sur CE modèle/corpus.
    """
    approx_top = {c for c, _ in approx.top(k)}
    exact_top = {c for c, _ in sorted(exact_scores.items(), key=lambda kv: abs(kv[1]),
                                      reverse=True)[:k]}
    if not exact_top:
        return 0.0
    return len(approx_top & exact_top) / len(exact_top)
