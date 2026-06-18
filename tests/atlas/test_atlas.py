"""Objet Atlas : pont sujet↔latent, identification, sérialisation."""
import torch

from meridian.atlas import Atlas, load_atlas, save_atlas


def _atlas():
    return Atlas(
        subject_names=["a", "b"],
        subject_dirs=torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),     # (S=2, L+1=1, H=2)
        gap_norms=torch.tensor([[2.0], [1.5]]),                      # (S, L+1)
        latent_dirs=torch.tensor([[[0.0, 1.0], [1.0, 0.0]]]),        # (L+1=1, k=2, H=2)
        explained_variance=torch.tensor([[0.7, 0.3]]),               # (L+1, k)
        meta={"model": "toy", "k": 2, "method": "svd", "center": "rest"},
    )


def test_match_subjects_to_latents():
    m = _atlas().match_subjects_to_latents(layer=0)
    # a=[1,0] le plus proche du latent 1 ; b=[0,1] le plus proche du latent 0.
    assert m["subject_to_latent"]["a"]["latent_index"] == 1
    assert abs(m["subject_to_latent"]["a"]["cosine"] - 1.0) < 1e-5
    assert m["subject_to_latent"]["b"]["latent_index"] == 0


def test_identify_returns_nearest_subjects_by_abs_cosine():
    matches = _atlas().identify(torch.tensor([0.96, 0.28]), k=2, layer=0)
    assert matches[0]["name"] == "a"          # plus proche de [1,0]
    assert matches[0]["cosine"] >= matches[1]["cosine"]


def test_identify_ignores_sign():
    # direction opposée à 'b' -> reste 'b' (sign arbitraire des directions).
    matches = _atlas().identify(torch.tensor([0.0, -1.0]), k=1, layer=0)
    assert matches[0]["name"] == "b" and abs(matches[0]["cosine"] - 1.0) < 1e-5


def test_save_load_roundtrip(tmp_path):
    a = _atlas()
    path = tmp_path / "atlas.safetensors"
    save_atlas(a, str(path))
    b = load_atlas(str(path))
    assert b.subject_names == a.subject_names
    assert b.meta == a.meta
    assert torch.allclose(b.subject_dirs, a.subject_dirs)
    assert torch.allclose(b.latent_dirs, a.latent_dirs)
    assert torch.allclose(b.explained_variance, a.explained_variance)
    assert torch.allclose(b.gap_norms, a.gap_norms)
