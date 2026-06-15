"""Tests du cache disque : déterminisme de la clé, hit/miss, écriture atomique, opt-out."""
import torch

from abliteration.cache import cached_tensor, make_key


def test_make_key_is_deterministic_and_sensitive():
    k1 = make_key("means", "model-a", "harmful", (1, 2, 3))
    k2 = make_key("means", "model-a", "harmful", (1, 2, 3))
    k3 = make_key("means", "model-b", "harmful", (1, 2, 3))   # modèle différent
    assert k1 == k2
    assert k1 != k3


def test_cache_miss_computes_then_hit_returns_cached(tmp_path):
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return torch.tensor([1.0, 2.0, 3.0])

    a = cached_tensor("k1", compute, cache_dir=tmp_path)
    b = cached_tensor("k1", compute, cache_dir=tmp_path)   # doit lire le cache, pas recalculer
    assert calls["n"] == 1
    assert torch.allclose(a, b)


def test_disabled_cache_always_recomputes(tmp_path):
    calls = {"n": 0}

    def compute():
        calls["n"] += 1
        return torch.zeros(2)

    cached_tensor("k", compute, enabled=False, cache_dir=tmp_path)
    cached_tensor("k", compute, enabled=False, cache_dir=tmp_path)
    assert calls["n"] == 2


def test_cache_handles_dict_of_tensors(tmp_path):
    val = {"a": torch.ones(3), "b": torch.zeros(2)}
    cached_tensor("d", lambda: val, cache_dir=tmp_path)
    got = cached_tensor("d", lambda: {}, cache_dir=tmp_path)
    assert torch.allclose(got["a"], torch.ones(3))
