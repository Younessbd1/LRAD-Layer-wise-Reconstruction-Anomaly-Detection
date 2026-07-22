"""Structured SVG diagram of every DeepEnsemble member architecture.

One horizontal lane per ensemble member, drawn from the *resolved* member
configs (``ensemble.member_variants`` applied on top of ``model``): the
input tile, the five conv blocks (rectangle height ∝ channel width, colour
= kernel size), the ×½ MaxPool markers on the arrows, the per-block
decoder taps, then GAP and the classifier heads. Channel widths, kernel
size, spatial sizes and the parameter count are all printed, so the figure
doubles as an architecture table.

Pure Python — no torch, no matplotlib — so the diagram can be produced on
any machine (``scripts/generate_arch_svg.py``) and is deterministic. The
parameter count is computed analytically from the config (conv without
bias + BatchNorm + linear heads), matching ``model.count_parameters`` on
the classifier.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Sequence

# Kernel size keeps the same fixed colours as the matplotlib figures
# (lrad.plots): 3×3 = blue, 5×5 = green.
KERNEL_COLORS = {3: "#0072B2", 5: "#009E73"}
_KERNEL_DEFAULT = "#888888"


def resolve_member_configs(cfg: dict, size: int | None = None) -> list[dict]:
    """One resolved config per ensemble member.

    ``ensemble.member_variants`` is a list of model overrides
    (``{channels, kernel_size}``); member ``i`` gets variant ``i`` (cycled
    when the ensemble outgrows the list). Every variant must keep the same
    number of conv blocks so the per-block decomposition stays aligned
    across members. Without variants every member shares ``cfg['model']``
    (the historical behaviour). ``size`` defaults to ``ensemble.size``.
    """
    ecfg = cfg.get("ensemble", {})
    size = int(size if size is not None else ecfg.get("size", 5))
    variants = ecfg.get("member_variants") or []
    n_blocks = len(cfg.get("model", {}).get("channels",
                                            (32, 64, 128, 256, 256)))
    member_cfgs: list[dict] = []
    for i in range(size):
        mc = copy.deepcopy(cfg)
        if variants:
            var = variants[i % len(variants)]
            channels = list(var.get("channels",
                                    mc["model"].get("channels")))
            if len(channels) != n_blocks:
                raise ValueError(
                    f"member_variants[{i % len(variants)}] has "
                    f"{len(channels)} blocks, expected {n_blocks} — the "
                    "per-block decomposition needs the same block count "
                    "in every member."
                )
            mc["model"]["channels"] = channels
            mc["model"]["kernel_size"] = int(var.get("kernel_size", 3))
        member_cfgs.append(mc)
    return member_cfgs


def classifier_n_params(mcfg: dict) -> int:
    """Analytic parameter count of a ``FacialCNN`` built from ``mcfg``.

    Conv blocks are ``Conv(k×k, bias=False) + BN`` (2 params/channel);
    heads are plain ``Linear``. Matches
    ``sum(p.numel() for p in model.parameters())`` for the classifier.
    """
    channels = list(mcfg.get("channels", (32, 64, 128, 256, 256)))
    k = int(mcfg.get("kernel_size", 3))
    in_ch = int(mcfg.get("in_channels", 3))
    n_attrs = int(mcfg.get("n_attrs", 6))
    n_gender = int(mcfg.get("n_gender", 2))
    total = 0
    prev = in_ch
    for c in channels:
        total += k * k * prev * c + 2 * c
        prev = c
    total += prev * n_gender + n_gender
    total += prev * n_attrs + n_attrs
    if mcfg.get("cutpaste_head"):
        total += prev * 2 + 2
    return total


def _fmt_params(n: int) -> str:
    return f"{n / 1e6:.2f} M" if n >= 1e6 else f"{n / 1e3:.0f} k"


def _spatial_sizes(image_size: int, n_blocks: int) -> list[int]:
    """Output spatial size of each conv block (pool after all but the last)."""
    sizes, s = [], image_size
    for i in range(n_blocks):
        if i < n_blocks - 1:
            s //= 2
        sizes.append(s)
    return sizes


def render_ensemble_svg(
    member_cfgs: Sequence[dict],
    out_path: str | Path,
    *,
    per_model: Sequence[dict] | None = None,
    title: str = "DeepEnsemble — member architectures",
) -> Path:
    """Write the ensemble architecture diagram to ``out_path``.

    ``member_cfgs`` are the resolved per-member configs (one full config
    dict each, as built by ``resolve_member_configs``). ``per_model``
    optionally supplies the matching summary records (``seed``,
    ``n_params``) — missing values fall back to the analytic count and the
    member index. Returns the written path.
    """
    lanes = len(member_cfgs)
    if lanes == 0:
        raise ValueError("need at least one member config")

    # --- Layout constants (px) -------------------------------------------
    label_w = 236          # left column: member name + summary lines
    bw, bgap = 92, 52      # block rectangle width and arrow gap
    input_w = 54
    head_w = 118
    lane_h = 158
    top = 74
    ch_scale = 0.16        # rect height = 18 + channels * ch_scale
    base_off = 108         # lane baseline (block bottoms) below lane top

    cfg0 = member_cfgs[0]
    n_blocks = len(cfg0.get("model", {}).get("channels", [32, 64, 128, 256, 256]))
    width = (label_w + input_w + (bgap + bw) * n_blocks + bgap + 46
             + head_w + 30)
    height = top + lanes * lane_h + 96

    e: list[str] = []
    e.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
        f'height="{height}" viewBox="0 0 {width} {height}" '
        f'font-family="Georgia, \'Times New Roman\', serif">'
    )
    e.append(
        '<style>'
        '.t{fill:#222;}'
        '.muted{fill:#666;}'
        '.small{font-size:10px;}'
        '.tiny{font-size:8.5px;}'
        '.lane{stroke:#ddd;stroke-width:1;}'
        '.arrow{stroke:#555;stroke-width:1.2;fill:none;}'
        '.blk{fill-opacity:0.14;stroke-width:1.4;}'
        '.head{fill:#f4f4f4;stroke:#777;stroke-width:1;}'
        '.dec{stroke:#999;stroke-width:0.9;stroke-dasharray:3 2;fill:none;}'
        '</style>'
    )
    e.append(
        '<defs><marker id="ah" viewBox="0 0 8 8" refX="7" refY="4" '
        'markerWidth="7" markerHeight="7" orient="auto">'
        '<path d="M0,0 L8,4 L0,8 z" fill="#555"/></marker></defs>'
    )
    e.append(f'<rect width="{width}" height="{height}" fill="white"/>')
    e.append(
        f'<text x="{width / 2:.0f}" y="30" text-anchor="middle" '
        f'font-size="19" class="t">{title}</text>'
    )
    e.append(
        f'<text x="{width / 2:.0f}" y="50" text-anchor="middle" '
        f'font-size="11" class="muted">Each block: Conv(k×k, pad k/2, '
        f'no bias) + BatchNorm + ReLU; ×½ = MaxPool 2×2 (all blocks but '
        f'the last). Rectangle height ∝ channel width; colour = kernel '
        f'size.</text>'
    )

    image_size = int(cfg0.get("dataset", {}).get("image_size", 64))

    for i, cfg in enumerate(member_cfgs):
        mcfg = cfg.get("model", {})
        channels = list(mcfg.get("channels", []))
        k = int(mcfg.get("kernel_size", 3))
        colour = KERNEL_COLORS.get(k, _KERNEL_DEFAULT)
        rec = (per_model[i] if per_model and i < len(per_model) else {})
        n_params = int(rec.get("n_params") or classifier_n_params(mcfg))
        seed = rec.get("seed")
        spatial = _spatial_sizes(image_size, len(channels))

        y0 = top + i * lane_h
        base = y0 + base_off
        e.append(f'<g id="model_{i}">')
        if i > 0:
            e.append(
                f'<line x1="16" y1="{y0}" x2="{width - 16}" y2="{y0}" '
                f'class="lane"/>'
            )

        # --- left label column ---
        lx = 24
        e.append(
            f'<text x="{lx}" y="{y0 + 34}" font-size="14" class="t" '
            f'font-weight="bold">Model {i + 1}</text>'
        )
        seed_txt = f'seed {seed} · ' if seed is not None else ""
        e.append(
            f'<text x="{lx}" y="{y0 + 52}" class="small muted">'
            f'{seed_txt}kernel {k}×{k} · {_fmt_params(n_params)} params'
            f'</text>'
        )
        e.append(
            f'<text x="{lx}" y="{y0 + 68}" class="small muted">channels '
            f'{"-".join(str(c) for c in channels)}</text>'
        )

        # --- input tile ---
        x = label_w
        ih = 18 + 3  # 3 input channels, same scale floor as blocks
        e.append(
            f'<rect x="{x}" y="{base - ih}" width="{input_w}" '
            f'height="{ih}" fill="#fafafa" stroke="#999" stroke-width="1"/>'
        )
        e.append(
            f'<text x="{x + input_w / 2:.0f}" y="{base + 14}" '
            f'text-anchor="middle" class="tiny muted">input '
            f'3×{image_size}²</text>'
        )
        x += input_w

        # --- conv blocks ---
        for b, c in enumerate(channels):
            pool = b < len(channels) - 1
            e.append(
                f'<line x1="{x}" y1="{base - 10}" x2="{x + bgap - 6}" '
                f'y2="{base - 10}" class="arrow" marker-end="url(#ah)"/>'
            )
            if pool:
                e.append(
                    f'<text x="{x + bgap / 2:.0f}" y="{base - 16}" '
                    f'text-anchor="middle" class="tiny muted">×½</text>'
                )
            bx = x + bgap
            bh = 18 + c * ch_scale
            e.append(
                f'<rect x="{bx}" y="{base - bh:.1f}" width="{bw}" '
                f'height="{bh:.1f}" rx="3" class="blk" fill="{colour}" '
                f'stroke="{colour}"/>'
            )
            e.append(
                f'<text x="{bx + bw / 2:.0f}" y="{base - bh - 6:.1f}" '
                f'text-anchor="middle" class="small t">L{b} · '
                f'Conv{k}×{k}</text>'
            )
            e.append(
                f'<text x="{bx + bw / 2:.0f}" y="{base - bh / 2 + 3.5:.1f}" '
                f'text-anchor="middle" class="small t" '
                f'font-weight="bold">{c}</text>'
            )
            e.append(
                f'<text x="{bx + bw / 2:.0f}" y="{base + 14}" '
                f'text-anchor="middle" class="tiny muted">'
                f'{spatial[b]}²</text>'
            )
            # per-block decoder tap
            e.append(
                f'<line x1="{bx + bw / 2:.0f}" y1="{base + 18}" '
                f'x2="{bx + bw / 2:.0f}" y2="{base + 30}" class="dec"/>'
            )
            e.append(
                f'<text x="{bx + bw / 2:.0f}" y="{base + 41}" '
                f'text-anchor="middle" class="tiny muted">dec L{b}</text>'
            )
            x = bx + bw

        # --- GAP + heads ---
        e.append(
            f'<line x1="{x}" y1="{base - 10}" x2="{x + bgap - 6}" '
            f'y2="{base - 10}" class="arrow" marker-end="url(#ah)"/>'
        )
        gx = x + bgap + 18
        e.append(
            f'<circle cx="{gx}" cy="{base - 10}" r="15" fill="#fafafa" '
            f'stroke="#777" stroke-width="1"/>'
        )
        e.append(
            f'<text x="{gx}" y="{base - 7}" text-anchor="middle" '
            f'class="tiny t">GAP</text>'
        )
        heads = [("gender", 2), ("attrs", int(mcfg.get("n_attrs", 6)))]
        if mcfg.get("cutpaste_head"):
            heads.append(("cutpaste", 2))
        hx = gx + 24
        hh, hgap = 17, 5
        hy0 = base - 10 - (len(heads) * hh + (len(heads) - 1) * hgap) / 2
        for j, (name, n_out) in enumerate(heads):
            hy = hy0 + j * (hh + hgap)
            e.append(
                f'<line x1="{gx + 15}" y1="{base - 10}" x2="{hx}" '
                f'y2="{hy + hh / 2:.1f}" class="arrow"/>'
            )
            e.append(
                f'<rect x="{hx}" y="{hy:.1f}" width="{head_w}" '
                f'height="{hh}" rx="3" class="head"/>'
            )
            e.append(
                f'<text x="{hx + head_w / 2:.0f}" y="{hy + hh - 5:.1f}" '
                f'text-anchor="middle" class="tiny t">{name} → '
                f'{n_out}</text>'
            )
        e.append('</g>')

    # --- legend / footnote ---
    ly = top + lanes * lane_h + 30
    lx = 24
    for k, colour in sorted(KERNEL_COLORS.items()):
        e.append(
            f'<rect x="{lx}" y="{ly - 11}" width="14" height="14" rx="3" '
            f'class="blk" fill="{colour}" stroke="{colour}"/>'
        )
        e.append(
            f'<text x="{lx + 20}" y="{ly}" class="small t">kernel '
            f'{k}×{k}</text>'
        )
        lx += 110
    e.append(
        f'<text x="24" y="{ly + 24}" class="small muted">dec Lk: frozen-'
        f'trunk BlockDecoder (ConvTranspose 4×4 stride-2 stages + BN + '
        f'ReLU) reconstructing the {image_size}² input from block k — the '
        f'reconstructions drive the ensemble bias/variance '
        f'decomposition.</text>'
    )
    e.append('</svg>')

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(e), encoding="utf-8")
    return out_path
