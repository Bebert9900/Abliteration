"""Dérive d'un atlas entre deux checkpoints (suivi de fine-tuning)."""
import math

import torch

from meridian.atlas import Atlas, atlas_drift, subspace_drift


def test_subspace_drift_zero_for_identical_subspace():
    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert subspace_drift(a, a.clone()) < 1e-6


def test_subspace_drift_max_for_orthogonal_subspaces():
    a = torch.tensor([[1.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 1.0, 0.0]])
    assert abs(subspace_drift(a, b) - 1.0) < 1e-6


def test_subspace_drift_invariant_to_basis_rotation_within_subspace():
    # Même plan (xy), bases différentes -> dérive nulle.
    a = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    inv = 1.0 / math.sqrt(2.0)
    b = torch.tensor([[inv, inv, 0.0], [inv, -inv, 0.0]])
    assert subspace_drift(a, b) < 1e-6


def _atlas(subject_dirs, latent_dirs, gaps, names=("a", "b")):
    return Atlas(
        subject_names=list(names),
        subject_dirs=subject_dirs,
        gap_norms=gaps,
        latent_dirs=latent_dirs,
        explained_variance=torch.ones(latent_dirs.shape[0], latent_dirs.shape[1]) / latent_dirs.shape[1],
        meta={"model": "toy"},
    )


def test_atlas_drift_identity_is_zero():
    subj = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])     # (S=2, L+1=1, H=2)
    lat = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])        # (L+1=1, k=2, H=2)
    gaps = torch.tensor([[2.0], [1.0]])
    a = _atlas(subj, lat, gaps)
    d = atlas_drift(a, a)
    assert d["subjects"]["a"]["summary"] < 1e-6
    assert d["latent_subspace_drift"]["summary"] < 1e-6
    assert abs(d["gap_norm_delta"]["a"]) < 1e-6


def test_atlas_drift_subject_rotation():
    ref_subj = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    ang = math.radians(60.0)
    rot_a = torch.tensor([[[math.cos(ang), math.sin(ang)]], [[0.0, 1.0]]])
    lat = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    gaps = torch.tensor([[2.0], [1.0]])
    ref = _atlas(ref_subj, lat, gaps)
    curr = _atlas(rot_a, lat, gaps)
    d = atlas_drift(curr, ref)
    # 1 - |cos(60°)| = 1 - 0.5 = 0.5 pour le sujet 'a' ; sujet 'b' inchangé.
    assert abs(d["subjects"]["a"]["summary"] - 0.5) < 1e-5
    assert d["subjects"]["b"]["summary"] < 1e-6


def test_summary_excludes_embedding_and_edge_layers():
    # 5 couches : seule la couche 0 (embeddings) diffère -> résumé (couches milieu) ≈ 0,
    # mais la dérive de la couche 0 reste visible dans per_layer (transparence).
    base = torch.tensor([[1.0, 0.0]]).repeat(5, 1)               # (5, 2), [1,0] partout
    ref_subj = base.unsqueeze(0).repeat(2, 1, 1)                 # (S=2, 5, 2)
    curr = ref_subj.clone()
    curr[0, 0] = torch.tensor([0.0, 1.0])                        # sujet 'a', couche 0 tournée 90°
    lat = torch.eye(2).unsqueeze(0).repeat(5, 1, 1)              # (5, k=2, 2) identiques
    gaps = torch.ones(2, 5)
    ref = _atlas(ref_subj, lat, gaps)
    cur = _atlas(curr, lat, gaps)
    d = atlas_drift(cur, ref)
    assert d["subjects"]["a"]["summary"] < 1e-6                  # couche 0 exclue du résumé
    assert abs(d["subjects"]["a"]["per_layer"][0] - 1.0) < 1e-6  # mais visible par couche


def test_subject_drift_is_signed_so_a_flip_is_maximal():
    # La direction d'un sujet (μ_s − μ_reste) est ORIENTÉE : un retournement à 180° est un
    # vrai changement -> drift signé = 1 − cos(180°) = 2 (|cos| l'aurait masqué à 0).
    base = torch.tensor([[1.0, 0.0]]).repeat(5, 1)
    ref_subj = base.unsqueeze(0).repeat(2, 1, 1)       # (S=2, 5, 2)
    curr = ref_subj.clone()
    curr[0] = -curr[0]                                  # sujet 'a' inversé à toutes les couches
    lat = torch.eye(2).unsqueeze(0).repeat(5, 1, 1)
    gaps = torch.ones(2, 5)
    d = atlas_drift(_atlas(curr, lat, gaps), _atlas(ref_subj, lat, gaps))
    assert abs(d["subjects"]["a"]["summary"] - 2.0) < 1e-6
    assert d["subjects"]["b"]["summary"] < 1e-6


def test_atlas_drift_only_common_subjects():
    subj = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
    lat = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
    gaps = torch.tensor([[2.0], [1.0]])
    ref = _atlas(subj, lat, gaps, names=("a", "b"))
    curr = _atlas(subj, lat, gaps, names=("a", "c"))
    d = atlas_drift(curr, ref)
    assert set(d["subjects"]) == {"a"}             # intersection des noms
