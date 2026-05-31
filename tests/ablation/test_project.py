"""Tests de project_out (orthogonalité numérique) et des variantes de direction."""
import torch

from src.directions import Directions
from src.ablation import Variant, ablation_direction, project_out


def test_project_out_removes_component_along_against():
    d = torch.tensor([1.0, 1.0])
    out = project_out(d, against=[torch.tensor([1.0, 0.0])])
    # composante x retirée -> direction unitaire selon y
    assert torch.allclose(out, torch.tensor([0.0, 1.0]), atol=1e-6)


def test_project_out_result_is_orthogonal_to_each_against_and_unit():
    torch.manual_seed(0)
    d = torch.randn(16)
    against = [torch.randn(16), torch.randn(16)]
    out = project_out(d, against)
    assert torch.allclose(out.norm(), torch.tensor(1.0), atol=1e-5)
    for v in against:
        assert abs(float(out @ (v / v.norm()))) < 1e-5


def test_project_out_empty_against_just_normalizes():
    out = project_out(torch.tensor([3.0, 0.0]), against=[])
    assert torch.allclose(out, torch.tensor([1.0, 0.0]), atol=1e-6)


def _dirs():
    return Directions(
        refusal=torch.tensor([[1.0, 1.0, 0.0]]),
        harmless=torch.tensor([[0.0, 0.0, 1.0]]),
        negation=torch.tensor([[1.0, 0.0, 0.0]]),
        agentic=torch.tensor([[0.0, 1.0, 0.0]]),
    )


def test_conventional_direction_is_raw_refusal_normalized():
    out = ablation_direction(_dirs(), layer=0, variant=Variant.CONVENTIONAL)
    assert torch.allclose(out, torch.tensor([1.0, 1.0, 0.0]) / (2 ** 0.5), atol=1e-6)


def test_preserving_direction_is_orthogonal_to_preserved():
    out = ablation_direction(
        _dirs(), layer=0, variant=Variant.PRESERVING, preserve=["negation", "agentic"]
    )
    # orthogonale à négation [1,0,0] ET agentique [0,1,0] -> selon z
    assert abs(float(out @ torch.tensor([1.0, 0.0, 0.0]))) < 1e-6
    assert abs(float(out @ torch.tensor([0.0, 1.0, 0.0]))) < 1e-6
