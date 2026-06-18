"""Orthogonalisation permanente des poids écrivant dans le residual stream.

`W' = W − r̂ (r̂ᵀ W)` appliqué à toutes les écritures (o_proj, down_proj de chaque couche — y
compris chaque expert MoE — et l'embedding). On gère l'axe de sortie selon le type :
- Linear attn/mlp out (out, in) : sortie = lignes  -> W -= outer(r̂, r̂ᵀW)
- Conv1D (in, out) et Embedding (vocab, hidden) : sortie = colonnes -> W -= outer(W r̂, r̂)

Le calcul se fait en float32 pour la stabilité puis est réécrit dans le dtype du poids (bf16 en
prod). `norm_preserve` réimpose la norme des tranches d'entrée (variante norm-preserving).
"""
from __future__ import annotations

import logging

import torch

from meridian.models import ArchAdapter, WriteKind

log = logging.getLogger(__name__)


def _orthogonalize_tensor(W: torch.Tensor, r: torch.Tensor, output_is_rows: bool, norm_preserve: bool,
                          alpha: float = 1.0):
    dtype = W.dtype
    # Calcul float32 sur CPU : la copie float32 d'une grosse matrice (embedding/lm_head, ~1 Go)
    # ne tient pas dans la VRAM résiduelle d'un GPU 8 Go déjà occupé par un modèle 3B. On travaille
    # une matrice à la fois sur CPU, puis on réécrit en place dans le tenseur GPU (copy_ cross-device).
    Wf = W.detach().to("cpu", torch.float32)
    rf = r.detach().to("cpu", torch.float32)
    slice_dim = 0 if output_is_rows else 1
    orig_norm = Wf.norm(dim=slice_dim, keepdim=True) if norm_preserve else None

    # `alpha` grave la force d'ablation (graduée) trouvée par l'optimiseur : α=1 retire toute la
    # composante de refus, α<1 n'en retire qu'une fraction (équivalent poids du hook inference-time).
    if output_is_rows:                       # (out, in), sortie le long des lignes
        Wf = Wf - alpha * torch.outer(rf, rf @ Wf)
    else:                                    # (*, out), sortie le long des colonnes
        Wf = Wf - alpha * torch.outer(Wf @ rf, rf)

    if norm_preserve:
        scale = (orig_norm / (Wf.norm(dim=slice_dim, keepdim=True) + 1e-8)).clamp(max=10.0)
        Wf = Wf * scale
    W.copy_(Wf.to(dtype))


@torch.no_grad()
def orthogonalize_weights(adapter: ArchAdapter, direction: torch.Tensor, norm_preserve: bool = False,
                          alpha: float = 1.0):
    """Orthogonalise toutes les matrices écrivant au residual stream contre `direction` (unitaire).

    `alpha` : force d'ablation gravée dans les poids (1.0 = complète ; <1.0 = graduée, miroir
    permanent du hook inference-time `make_ablation_hook(..., alpha)`).
    """
    H = direction.shape[-1]
    applied = skipped = 0
    for target in adapter.residual_writers():
        W = target.module.weight.data
        output_is_rows = (target.kind != WriteKind.EMBEDDING) and (not target.is_conv1d)
        out_dim_size = W.shape[0] if output_is_rows else W.shape[1]
        if out_dim_size != H:
            # dimension de sortie ≠ taille de la direction -> matrice non concernée, on saute.
            # Sur une archi standard cela ne devrait jamais arriver : si un writer attendu est
            # sauté, du refus résiduel peut subsister. On le signale plutôt que de l'avaler.
            skipped += 1
            log.warning("Writer résiduel sauté (dim sortie %d ≠ %d) : %s",
                        out_dim_size, H, getattr(target, "name", target.kind))
            continue
        _orthogonalize_tensor(W, direction, output_is_rows, norm_preserve, alpha=alpha)
        applied += 1
    log.info("Orthogonalisation : %d writers traités, %d sautés (α=%.2f).", applied, skipped, alpha)
