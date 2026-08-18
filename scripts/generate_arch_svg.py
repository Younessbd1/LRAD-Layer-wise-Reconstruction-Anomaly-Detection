#!/usr/bin/env python3
"""Generate the LRAD architecture diagrams (SVG) from a config.

Three figures, all derived from the config so they cannot drift from what
the code builds:

  * ``pipeline_classifier.svg`` — one member's forward pass, input tensor
    to task heads, with the per-block shapes and the loss weights.
  * ``pipeline_decoder.svg``    — the per-block ``BlockDecoder`` stack,
    one lane per conv block, with the analytic parameter cost.
  * ``ensemble_architectures_<experiment>.svg`` — one lane per ensemble
    member, resolving ``ensemble.member_variants`` exactly like
    ``run_ensemble.py`` (cycled if the ensemble outgrows the list).

Usage:
    python scripts/generate_arch_svg.py --config configs/celeba_ood_128.yaml
    python scripts/generate_arch_svg.py --config configs/celeba_ood.yaml \\
        --out-dir docs/diagrams --size 10
    python scripts/generate_arch_svg.py --config configs/celeba_ood.yaml \\
        --only ensemble
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
    render_classifier_svg,
    render_decoder_svg,
    render_ensemble_svg,
    resolve_member_configs,
)

FIGURES = ("classifier", "decoder", "ensemble")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Render the LRAD architecture diagrams (SVG).",
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory. Default: docs/diagrams/")
    ap.add_argument("--size", type=int, default=None,
                    help="Number of members. Default: ensemble.size.")
    ap.add_argument("--member", type=int, default=0,
                    help="Which member the classifier figure describes "
                         "(0-based). Default: 0.")
    ap.add_argument("--only", choices=FIGURES, default=None,
                    help="Render a single figure instead of all three.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    member_cfgs = resolve_member_configs(cfg, args.size)
    if not 0 <= args.member < len(member_cfgs):
        ap.error(f"--member must be in [0, {len(member_cfgs) - 1}]")
    base_seed = int(cfg.get("ensemble", {}).get(
        "base_seed", cfg.get("experiment", {}).get("seed", 42),
    ))
    per_model = [{"seed": base_seed + i} for i in range(len(member_cfgs))]

    name = cfg.get("experiment", {}).get("name", "ensemble")
    out_dir = args.out_dir or (_ROOT / "docs" / "diagrams")
    wanted = FIGURES if args.only is None else (args.only,)

    if "classifier" in wanted:
        path = render_classifier_svg(
            member_cfgs[args.member],
            out_dir / "pipeline_classifier.svg",
            member_index=args.member,
            n_members=len(member_cfgs),
        )
        print(f"wrote {path}")
    if "decoder" in wanted:
        path = render_decoder_svg(
            member_cfgs[args.member], out_dir / "pipeline_decoder.svg",
        )
        print(f"wrote {path}")
    if "ensemble" in wanted:
        title = (f"DeepEnsemble — member architectures  ({name}, "
                 f"{len(member_cfgs)} models)")
        path = render_ensemble_svg(
            member_cfgs,
            out_dir / f"ensemble_architectures_{name}.svg",
            per_model=per_model,
            title=title,
        )
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
