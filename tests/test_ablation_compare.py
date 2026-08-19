"""scripts/compare_ablation.py — arm discovery, deltas, tables, figures.

Runs the comparison end to end on tiny synthetic run dirs (JSONs only —
no models, no data), so the whole file is fast. The synthetic metrics are
chosen so every expected delta is a round, assertable number.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import compare_ablation as ca  # noqa: E402

BLOCKS = 5


def _write_arm(
    root: Path,
    dirname: str,
    *,
    base: float,
    cutpaste: bool,
    fused: bool = True,
) -> Path:
    """A minimal but structurally faithful arm run dir."""
    ens = root / dirname / "ensemble"
    ens.mkdir(parents=True)
    per_block = [round(base - 0.05 + 0.01 * k, 6) for k in range(BLOCKS)]
    summary = {
        "experiment": f"abl_{dirname}",
        "ensemble_size": 3,
        "seeds": [42, 43, 44],
        "agg": "p95",
        "per_model": [
            {"seed": 42 + i, "gender_acc": 0.9 + 0.01 * i,
             "auroc_recon": base - 0.02 * i, "n_params": 1000 + i}
            for i in range(3)
        ],
        "decomposition_auroc": {
            "aggregated": {"risk": base, "bias": base + 0.01,
                           "variance": base + 0.02},
            "per_block": {t: per_block for t in ("risk", "bias", "variance")},
        },
        "anomaly_auroc": {"aggregated": base + 0.01,
                          "per_block": per_block},
        "uncertainty_auroc": {
            "epistemic": {"gender": base, "attrs": base,
                          "combined": base + 0.03},
        },
    }
    with open(ens / "summary.json", "w") as f:
        json.dump(summary, f)
    if fused:
        auroc = {
            "locfre_b3": {"auroc": base + 0.1},
            "ens_energy_gender": {"auroc": base + 0.05},
            "unc_epistemic_combined": {"auroc": base + 0.03},
            "fused_rank": {"auroc": base + 0.15},
            "fused_supervised": {"auroc": base + 0.2},
        }
        if cutpaste:
            auroc["cutpaste_prob"] = {"auroc": base + 0.08}
        with open(ens / "fused_auroc.json", "w") as f:
            json.dump({"auroc": auroc}, f)
        with open(ens / "localized_auroc.json", "w") as f:
            json.dump({"auroc": {
                "zscore_risk_aggregated": {"auroc": base + 0.06},
                "zscore_risk_per_block_0": {"auroc": base},
            }}, f)
    return root / dirname


@pytest.fixture()
def ablation_root(tmp_path: Path) -> Path:
    root = tmp_path / "ablation"
    _write_arm(root, "baseline_20260819_100000_1", base=0.60, cutpaste=False)
    _write_arm(root, "arch_20260819_100000_2", base=0.63, cutpaste=False)
    _write_arm(root, "cutpaste_20260819_100000_3", base=0.62, cutpaste=True)
    _write_arm(root, "arch_cutpaste_20260819_100000_4", base=0.65,
               cutpaste=True)
    return root


def test_discover_prefers_newest_and_ignores_foreign_dirs(tmp_path: Path):
    root = tmp_path / "ablation"
    old = _write_arm(root, "arch_20260101_000000_1", base=0.6, cutpaste=False)
    new = _write_arm(root, "arch_20260819_000000_2", base=0.7, cutpaste=False)
    # A dir that merely starts with the arm name must not match "arch"...
    _write_arm(root, "arch_cutpaste_20260819_000000_3", base=0.8,
               cutpaste=True)
    # ...nor an unfinished run (no summary.json).
    (root / "arch_20270101_000000_9").mkdir()
    assert ca.discover_arm(root, "arch") == new
    assert old.exists()  # untouched, just not selected


def test_discover_missing_arm_returns_none(tmp_path: Path):
    (tmp_path / "ablation").mkdir()
    assert ca.discover_arm(tmp_path / "ablation", "cutpaste") is None


def test_load_arm_tolerates_missing_fused(tmp_path: Path):
    d = _write_arm(tmp_path, "baseline_20260819_100000_1", base=0.6,
                   cutpaste=False, fused=False)
    arm = ca.load_arm("baseline", d)
    assert arm["has_fused"] is False
    assert "fused_supervised" not in arm["metrics"]
    assert arm["metrics"]["bias"] == pytest.approx(0.61)
    assert len(arm["per_block"]["bias"]) == BLOCKS
    assert len(arm["members"]["auroc_recon"]) == 3


def test_case_deltas_skip_metrics_absent_on_either_side():
    base = {"metrics": {"fused_supervised": 0.80, "bias": 0.61}}
    arm = {"metrics": {"fused_supervised": 0.85, "bias": 0.66,
                       "cutpaste_prob": 0.7}}
    deltas = ca.case_deltas(arm, base)
    assert deltas["fused (supervised)"] == pytest.approx(0.05)
    assert deltas["decomp bias (p95)"] == pytest.approx(0.05)
    # cutpaste_prob has no baseline counterpart -> no delta row.
    assert all("cutpaste" not in k for k in deltas)


def test_main_end_to_end(ablation_root: Path, monkeypatch):
    out = ablation_root / "comparison"
    monkeypatch.setattr(
        sys, "argv",
        ["compare_ablation.py", "--root", str(ablation_root)],
    )
    assert ca.main() == 0

    results = json.loads((out / "ablation_results.json").read_text())
    assert [a["arm"] for a in results["arms"]] == list(ca.ARMS)
    deltas = {c["arm"]: c["deltas"] for c in results["case_studies"]}
    assert deltas["arch"]["fused (supervised)"] == pytest.approx(0.03)
    assert deltas["cutpaste"]["fused (supervised)"] == pytest.approx(0.02)
    assert deltas["arch_cutpaste"]["fused (supervised)"] == pytest.approx(0.05)

    table = (out / "ablation_table.md").read_text()
    assert "| fused (supervised) | 0.8000 | 0.8300 | 0.8200 | 0.8500 |" \
        in table
    # cutpaste_prob exists only on the two pretext arms.
    assert "| cutpaste P(altered) | — | — | 0.7000 | 0.7300 |" in table

    csv = (out / "ablation_table.csv").read_text()
    assert "fused_supervised,baseline,0.800000" in csv

    for name in ("auroc_bars", "case_studies", "per_block", "members"):
        assert (out / "plots" / f"{name}.png").stat().st_size > 0


def test_main_without_baseline_writes_nothing(tmp_path: Path, monkeypatch):
    root = tmp_path / "ablation"
    _write_arm(root, "arch_20260819_100000_2", base=0.63, cutpaste=False)
    monkeypatch.setattr(
        sys, "argv", ["compare_ablation.py", "--root", str(root)],
    )
    assert ca.main() == 2
    assert not (root / "comparison").exists()


def test_main_with_explicit_override_and_partial_arms(
    ablation_root: Path, tmp_path: Path, monkeypatch,
):
    """--arm NAME=DIR wins over discovery; a missing arm degrades cleanly."""
    other = _write_arm(tmp_path, "elsewhere", base=0.70, cutpaste=False)
    import shutil
    for d in ablation_root.glob("cutpaste_*"):
        shutil.rmtree(d)
    monkeypatch.setattr(
        sys, "argv",
        ["compare_ablation.py", "--root", str(ablation_root),
         "--arm", f"arch={other}", "--no-plots"],
    )
    assert ca.main() == 0
    results = json.loads(
        (ablation_root / "comparison" / "ablation_results.json").read_text(),
    )
    by_arm = {a["arm"]: a for a in results["arms"]}
    assert by_arm["arch"]["run_dir"] == str(other)
    assert "cutpaste" not in by_arm
    deltas = {c["arm"]: c["deltas"] for c in results["case_studies"]}
    assert deltas["cutpaste"] == {}
    assert deltas["arch"]["fused (supervised)"] == pytest.approx(0.10)
