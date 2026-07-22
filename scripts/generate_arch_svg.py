#!/usr/bin/env python3
"""Generate the DeepEnsemble member-architecture SVG from a config.

Resolves ``ensemble.member_variants`` exactly like ``run_ensemble.py``
(one architecture per member, cycled if the ensemble outgrows the list)
and renders one structured lane per member: conv blocks (height ∝ channel
width, colour = kernel size), MaxPool markers, per-block decoder taps,
GAP and the classifier heads, with channel/kernel/param annotations.

Usage:
    python scripts/generate_arch_svg.py --config configs/celeba_ood_cutpaste128.yaml
    python scripts/generate_arch_svg.py --config configs/celeba_ood.yaml \\
        --out docs/diagrams/ensemble_architectures.svg --size 10
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from lrad.arch_diagram import (  # noqa: E402
    render_ensemble_svg,
    resolve_member_configs,
)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render the DeepEnsemble architecture diagram (SVG).",
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=None,
                    help="Output SVG path. Default: docs/diagrams/"
                         "ensemble_architectures_<experiment>.svg")
    ap.add_argument("--size", type=int, default=None,
                    help="Number of members. Default: ensemble.size.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    member_cfgs = resolve_member_configs(cfg, args.size)
    base_seed = int(cfg.get("ensemble", {}).get(
        "base_seed", cfg.get("experiment", {}).get("seed", 42),
    ))
    per_model = [{"seed": base_seed + i} for i in range(len(member_cfgs))]

    name = cfg.get("experiment", {}).get("name", "ensemble")
    out = args.out or (
        _ROOT / "docs" / "diagrams" / f"ensemble_architectures_{name}.svg"
    )
    path = render_ensemble_svg(
        member_cfgs, out, per_model=per_model,
        title=f"DeepEnsemble — member architectures  ({name}, "
              f"{len(member_cfgs)} models)",
    )
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
