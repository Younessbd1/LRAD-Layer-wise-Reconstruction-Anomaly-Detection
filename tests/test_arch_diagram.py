"""Tests for the DeepEnsemble architecture SVG generator."""

from __future__ import annotations

import xml.dom.minidom as minidom

import pytest

from lrad.arch_diagram import (
    classifier_n_params,
    render_ensemble_svg,
    resolve_member_configs,
)
from lrad.model import build_model


def _cfg(size: int = 3) -> dict:
    return {
        "experiment": {"name": "test", "seed": 42},
        "dataset": {"image_size": 64},
        "model": {
            "in_channels": 3,
            "channels": [8, 16, 32, 32, 32],
            "kernel_size": 3,
            "n_attrs": 6,
            "n_gender": 2,
        },
        "ensemble": {
            "size": size,
            "member_variants": [
                {"channels": [8, 16, 32, 32, 32], "kernel_size": 3},
                {"channels": [4, 8, 16, 32, 64], "kernel_size": 5},
            ],
        },
    }


def test_resolve_member_configs_cycles_variants():
    cfgs = resolve_member_configs(_cfg(size=5))
    assert len(cfgs) == 5
    # variants cycle: members 0, 2, 4 get variant 0; members 1, 3 variant 1
    assert cfgs[0]["model"]["channels"] == [8, 16, 32, 32, 32]
    assert cfgs[1]["model"]["kernel_size"] == 5
    assert cfgs[2]["model"]["channels"] == cfgs[0]["model"]["channels"]
    assert cfgs[3]["model"]["channels"] == cfgs[1]["model"]["channels"]
    # deep copies — mutating one member must not leak into another
    cfgs[0]["model"]["channels"][0] = 999
    assert cfgs[2]["model"]["channels"][0] == 8


def test_resolve_member_configs_rejects_block_count_mismatch():
    cfg = _cfg()
    cfg["ensemble"]["member_variants"] = [{"channels": [8, 16], "kernel_size": 3}]
    with pytest.raises(ValueError):
        resolve_member_configs(cfg)


def test_classifier_n_params_matches_torch():
    for mcfg in (
        {"channels": [8, 16, 32, 32, 32], "kernel_size": 3},
        {"channels": [4, 8, 16, 32, 64], "kernel_size": 5,
         "cutpaste_head": True},
    ):
        model = build_model({"model": mcfg, "dataset": {"image_size": 32}})
        assert classifier_n_params(mcfg) == sum(
            p.numel() for p in model.parameters()
        )


def test_render_ensemble_svg_writes_valid_xml(tmp_path):
    cfgs = resolve_member_configs(_cfg(size=4))
    out = tmp_path / "arch.svg"
    path = render_ensemble_svg(
        cfgs, out, per_model=[{"seed": 42 + i} for i in range(4)],
    )
    assert path == out and out.exists()
    doc = minidom.parse(str(out))
    lanes = [
        g for g in doc.getElementsByTagName("g")
        if g.getAttribute("id").startswith("model_")
    ]
    assert len(lanes) == 4


def test_render_ensemble_svg_rejects_empty(tmp_path):
    with pytest.raises(ValueError):
        render_ensemble_svg([], tmp_path / "x.svg")
