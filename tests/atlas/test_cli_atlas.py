"""CLI atlas-build / atlas-identify / atlas-monitor : parsing, schéma, handler hors-ligne."""
import contextlib
import io
import json

import pytest
import torch

from meridian.atlas import Atlas, save_atlas
from meridian.cli import build_parser, main


def _run_main(argv):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = main(argv)
    return rc, buf.getvalue()


def test_atlas_build_parser():
    ns = build_parser().parse_args(
        ["atlas-build", "m", "--dataset", "ds.jsonl", "--k", "16", "--out", "a.safetensors"])
    assert ns.command == "atlas-build" and ns.dataset == "ds.jsonl" and ns.k == 16
    assert ns.out == "a.safetensors"


def test_atlas_monitor_parser():
    ns = build_parser().parse_args(
        ["atlas-monitor", "--checkpoints", "c0,c1,c2", "--dataset", "ds.jsonl"])
    assert ns.command == "atlas-monitor" and ns.checkpoints == "c0,c1,c2"


def test_atlas_dataset_is_required():
    with pytest.raises(SystemExit):
        build_parser().parse_args(["atlas-build", "m", "--out", "a.safetensors"])


def test_json_flag_on_atlas_commands():
    parser = build_parser()
    for argv in (["atlas-build", "m", "--dataset", "d", "--out", "o"],
                 ["atlas-identify", "--atlas", "a", "--subject", "x"],
                 ["atlas-monitor", "--checkpoints", "c", "--dataset", "d"]):
        ns = parser.parse_args(argv)
        assert hasattr(ns, "json") and ns.json is False


def test_schema_lists_atlas_commands_with_output():
    _, out = _run_main(["schema", "--json"])
    cmds = json.loads(out)["data"]["commands"]
    for c in ("atlas-build", "atlas-identify", "atlas-monitor"):
        assert c in cmds
    assert "matches" in cmds["atlas-identify"]["output"]
    assert "series" in cmds["atlas-monitor"]["output"]


def test_atlas_identify_handler_offline(tmp_path):
    atlas = Atlas(
        subject_names=["a", "b"],
        subject_dirs=torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
        gap_norms=torch.tensor([[2.0], [1.0]]),
        latent_dirs=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        explained_variance=torch.tensor([[0.6, 0.4]]),
        meta={"model": "toy"},
    )
    path = tmp_path / "atlas.safetensors"
    save_atlas(atlas, str(path))

    rc, out = _run_main(["atlas-identify", "--atlas", str(path), "--subject", "a", "--json"])
    env = json.loads(out)
    assert rc == 0 and env["status"] == "ok" and env["command"] == "atlas-identify"
    assert env["data"]["matches"][0]["name"] == "a"


def test_atlas_identify_requires_one_query_source(tmp_path):
    atlas = Atlas(["a", "b"], torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]]),
                  torch.tensor([[2.0], [1.0]]), torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
                  torch.tensor([[0.6, 0.4]]), {"model": "toy"})
    path = tmp_path / "atlas.safetensors"
    save_atlas(atlas, str(path))
    rc, out = _run_main(["atlas-identify", "--atlas", str(path), "--json"])
    env = json.loads(out)
    assert rc == 1 and env["status"] == "error"   # ni --subject ni --direction
