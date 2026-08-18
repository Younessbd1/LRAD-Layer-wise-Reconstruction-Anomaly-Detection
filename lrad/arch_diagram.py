"""Structured SVG diagrams of the LRAD pipeline.

Three figures, all rendered from a *config* dict so they can never drift
from what the code actually builds:

``render_classifier_svg``
    One ensemble member's forward pass — input tensor, the conv blocks
    with their per-block output shapes, GAP, and the task/SSL heads with
    their losses.
``render_decoder_svg``
    The per-block ``BlockDecoder`` stack: one lane per conv block, the
    ConvTranspose stages that grow the activation back to image
    resolution, the 1×1 + Sigmoid output head, and the parameter cost.
``render_ensemble_svg``
    One horizontal lane per ensemble member, drawn from the *resolved*
    member configs (``ensemble.member_variants`` applied on top of
    ``model``): the input tile, the conv blocks (rectangle height ∝
    channel width, colour = kernel size), the ×½ MaxPool markers, the
    per-block decoder taps, then GAP and the classifier heads.

Pure Python — no torch, no matplotlib — so the diagrams can be produced
on any machine (``scripts/generate_arch_svg.py``) and are deterministic.
Parameter counts are computed analytically from the config (conv without
bias + BatchNorm + linear heads), matching ``model.count_parameters`` on
the classifier and ``decoder.build_decoders`` on the decoders.
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Sequence

# Kernel size keeps the same fixed colours as the matplotlib figures
# (lrad.plots): 3×3 = blue, 5×5 = green. Do not repoint these — plots.py
# imports them for the architecture scatter so the two agree.
KERNEL_COLORS = {3: "#0072B2", 5: "#009E73"}
_KERNEL_DEFAULT = "#888888"

# Diagram palette. The conv/tensor/head hues carry meaning and are shared
# across the three figures: blue = tensor, teal = conv, purple = task
# head, peach = self-supervised / output.
_INK = "#11161f"          # bold titles
_BODY = "#2d333d"          # body copy
_MUTED = "#686f7b"         # secondary copy
_FAINT = "#94a3b8"         # tertiary copy
_RULE = "#e1e5ea"          # card borders and dividers
_CARD = "#ffffff"

_BLUE_BG, _BLUE_FG = "#d9eafc", "#004e90"
_TEAL_BG, _TEAL_FG = "#cbedea", "#004c49"
_PURPLE_BG, _PURPLE_FG = "#ebe3fc", "#5b21b6"
_PEACH_BG, _PEACH_FG = "#ffe4da", "#a53a12"

# Ensemble block gradients (light top-left → dark bottom-right).
_KERNEL_GRAD = {
    3: ("#7fb8c7", "#3b88a0"),
    5: ("#84bca5", "#4c9777"),
}
_GRAD_DEFAULT = ("#c2c8d0", "#8a929c")

# Real family names only, no CSS pseudo-families: browsers understand
# "-apple-system", but headless renderers (cairosvg/rsvg → Pango) fail to
# match it and drop the whole stack to their default, which on some Linux
# boxes is Bitstream Vera — a font with none of the arrows or maths signs.
_FONT = ("'Segoe UI', Roboto, 'Helvetica Neue', Helvetica, Arial, "
         "'Liberation Sans', 'Noto Sans', 'DejaVu Sans', sans-serif")
_MONO = ("ui-monospace, SFMono-Regular, Menlo, Consolas, "
         "'DejaVu Sans Mono', 'Liberation Mono', monospace")
# Pango picks ONE font per run — it has no per-glyph fallback the way a
# browser does — so the few rare glyphs (∈ ℝ ∝) get their own stack that
# leads with families known to carry them.
_MATHF = ("'DejaVu Sans', 'Noto Sans Math', 'Segoe UI Symbol', "
          "'Nimbus Sans', sans-serif")

# U+00B2 is Latin-1 and safe everywhere; the other superscript/subscript
# digits are not, so anything beyond "²" goes through a <tspan> instead
# (see _text_parts).
_SQ = "²"


# --------------------------------------------------------------------------
# config resolution / analytic parameter counts
# --------------------------------------------------------------------------
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
    return total


def decoder_channel_path(
    in_channels: int, n_up: int, min_channels: int = 16,
) -> list[int]:
    """Channel width after each ×2 stage of a ``BlockDecoder``.

    Mirrors ``decoder.BlockDecoder``: halve, floored at ``min_channels``.
    """
    path, ch = [], int(in_channels)
    for _ in range(n_up):
        ch = max(min_channels, ch // 2)
        path.append(ch)
    return path


def decoder_n_params(
    in_channels: int,
    in_size: int,
    out_size: int,
    out_channels: int = 3,
    min_channels: int = 16,
) -> int:
    """Analytic parameter count of one ``BlockDecoder``.

    ``ConvTranspose2d(4×4, bias=False) + BN`` per ×2 stage, then a 1×1
    conv with bias. Matches ``sum(p.numel() for p in decoder.parameters())``.
    """
    n_up = int(round(math.log2(out_size // in_size)))
    total, ch = 0, int(in_channels)
    for new_ch in decoder_channel_path(in_channels, n_up, min_channels):
        total += 4 * 4 * ch * new_ch + 2 * new_ch
        ch = new_ch
    if n_up == 0:
        ref = max(min_channels, in_channels)
        total += 3 * 3 * in_channels * ref + 2 * ref
        ch = ref
    total += ch * out_channels + out_channels
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


# --------------------------------------------------------------------------
# tiny SVG toolkit
# --------------------------------------------------------------------------
def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;"))


# Glyphs outside Latin-1 that plenty of UI sans faces simply do not carry.
# fontconfig never reports a miss — it silently substitutes — so a headless
# renderer would draw tofu for these. Each one is wrapped in its own tspan
# pinned to _MATHF, which leaves the surrounding design font untouched.
_RARE = frozenset("→∈ℝ∝≈±∞")


def _rich(s: str) -> str:
    """Escape ``s``, isolating rare glyphs into their own font run."""
    text = str(s)
    if not any(c in _RARE for c in text):
        return _esc(text)
    out, buf = [], []
    for c in text:
        if c in _RARE:
            if buf:
                out.append(_esc("".join(buf)))
                buf = []
            out.append(f'<tspan font-family="{_MATHF}">{_esc(c)}</tspan>')
        else:
            buf.append(c)
    if buf:
        out.append(_esc("".join(buf)))
    return "".join(out)


def _tw(text: str, size: float, bold: bool = False,
        tracking: float = 0.0) -> float:
    """Rough advance width of ``text`` — enough to size pills and tags.

    Maths glyphs (∈ ℝ →) come from a fallback face and run noticeably
    wider than the Latin average, so they are charged extra; without that
    the ``h ∈ ℝ`` tag is cut short and its superscript spills out.
    """
    factor = 0.575 if bold else 0.525
    wide = sum(1 for c in text if ord(c) > 0x7F)
    return len(text) * (size * factor + tracking) + wide * size * 0.22


def _text(x: float, y: float, s: str, *, size: float = 13,
          fill: str = _BODY, bold: bool = False, anchor: str = "start",
          mono: bool = False, tracking: float = 0.0,
          opacity: float | None = None, math: bool = False) -> str:
    bits = [f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size:g}"',
            f'fill="{fill}"']
    if bold:
        bits.append('font-weight="700"')
    if anchor != "start":
        bits.append(f'text-anchor="{anchor}"')
    if mono:
        bits.append(f'font-family="{_MONO}"')
    elif math:
        bits.append(f'font-family="{_MATHF}"')
    if tracking:
        bits.append(f'letter-spacing="{tracking:g}"')
    if opacity is not None:
        bits.append(f'opacity="{opacity:g}"')
    return " ".join(bits) + f">{_rich(s)}</text>"


def _parts_width(parts: Sequence[tuple[str, int]], size: float,
                 bold: bool = False) -> float:
    """Advance width of a ``_text_parts`` run."""
    return sum(_tw(txt, size if lvl == 0 else size * 0.72, bold=bold)
               for txt, lvl in parts)


def _text_parts(x: float, y: float, parts: Sequence[tuple[str, int]], *,
                size: float = 13, fill: str = _BODY, bold: bool = False,
                anchor: str = "start", math: bool = False) -> str:
    """Text with super/subscript runs.

    ``parts`` is a sequence of ``(text, level)`` with level ``0`` for
    baseline, ``+1`` superscript and ``-1`` subscript. Rendered with
    <tspan dy=…> rather than Unicode superscript digits, which most
    sans fonts only cover for ² ³ ¹.
    """
    bits = [f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size:g}"',
            f'fill="{fill}"']
    if bold:
        bits.append('font-weight="700"')
    if anchor != "start":
        bits.append(f'text-anchor="{anchor}"')
    if math:
        bits.append(f'font-family="{_MATHF}"')
    out = [" ".join(bits) + ">"]
    shift = 0.0
    for txt, lvl in parts:
        want = (-size * 0.36 if lvl > 0 else
                size * 0.18 if lvl < 0 else 0.0)
        dy = want - shift
        shift = want
        if lvl == 0 and dy == 0:
            out.append(_rich(txt))
        else:
            out.append(f'<tspan dy="{dy:.2f}" '
                       f'font-size="{size * 0.72:.1f}">{_rich(txt)}</tspan>'
                       if lvl != 0 else
                       f'<tspan dy="{dy:.2f}">{_rich(txt)}</tspan>')
    out.append("</text>")
    return "".join(out)


def _num(v: float) -> str:
    """Format a loss weight: keep one decimal on whole numbers (2 → 2.0)."""
    return f"{v:.1f}" if float(v) == int(v) else f"{v:g}"


def _card(x: float, y: float, w: float, h: float, r: float = 16) -> str:
    return (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" '
            f'height="{h:.1f}" rx="{r:g}" fill="{_CARD}" stroke="{_RULE}" '
            f'stroke-width="1.2" filter="url(#soft)"/>')


def _pill(x: float, y: float, label: str, bg: str, fg: str, *,
          size: float = 9.5, pad: float = 11, h: float = 20) -> tuple[str, float]:
    """Uppercase badge. Returns (svg, width)."""
    w = _tw(label, size, bold=True, tracking=0.9) + 2 * pad
    svg = (f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:g}" '
           f'rx="{h / 2:g}" fill="{bg}"/>'
           + _text(x + pad, y + h / 2 + size * 0.36, label, size=size,
                   fill=fg, bold=True, tracking=0.9))
    return svg, w


def _tag(cx_right: float, cy: float, label: str, bg: str, fg: str, *,
         size: float = 13, pad: float = 13, h: float = 30,
         sup: str | None = None, math: bool = False) -> str:
    """Right-aligned rounded tensor-shape tag, optional superscript tail."""
    parts: list[tuple[str, int]] = [(label, 0)]
    if sup:
        parts.append((sup, 1))
    w = _parts_width(parts, size, bold=True) + 2 * pad
    x = cx_right - w
    # Left-anchored: with text-anchor="middle" some renderers re-anchor
    # every <tspan> to the same point, so a superscript tail lands on top
    # of the glyph it should follow.
    return (f'<rect x="{x:.1f}" y="{cy - h / 2:.1f}" width="{w:.1f}" '
            f'height="{h:g}" rx="8" fill="{bg}"/>'
            + _text_parts(x + pad, cy + size * 0.36, parts, size=size,
                          fill=fg, bold=True, math=math))


def _vline_arrow(x: float, y0: float, y1: float, *, colour: str = _INK,
                 dashed: bool = False) -> str:
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    return (f'<line x1="{x:.1f}" y1="{y0:.1f}" x2="{x:.1f}" '
            f'y2="{y1:.1f}" stroke="{colour}" stroke-width="2.2"'
            f'{dash} marker-end="url(#tip)"/>')


def _hline_arrow(y: float, x0: float, x1: float, *, colour: str = _INK,
                 dashed: bool = False, width: float = 2.2) -> str:
    dash = ' stroke-dasharray="6 5"' if dashed else ""
    marker = "tipgrey" if colour != _INK else "tip"
    return (f'<line x1="{x0:.1f}" y1="{y:.1f}" x2="{x1:.1f}" '
            f'y2="{y:.1f}" stroke="{colour}" stroke-width="{width:g}"'
            f'{dash} marker-end="url(#{marker})"/>')


def _header(width: float, height: float, title: str) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width:.0f}" '
        f'height="{height:.0f}" viewBox="0 0 {width:.0f} {height:.0f}" '
        f'font-family="{_FONT}">',
        f'<title>{_esc(title)}</title>',
        '<defs>'
        '<marker id="tip" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        f'<path d="M0,1 L9,5 L0,9 z" fill="{_INK}"/></marker>'
        '<marker id="tipgrey" viewBox="0 0 10 10" refX="8" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M0,1 L9,5 L0,9 z" fill="#8b93a0"/></marker>'
        '<filter id="soft" x="-8%" y="-20%" width="116%" height="150%">'
        '<feDropShadow dx="0" dy="1.5" stdDeviation="2.2" '
        'flood-color="#5b6675" flood-opacity="0.13"/></filter>'
        '</defs>',
        f'<rect width="{width:.0f}" height="{height:.0f}" fill="#ffffff"/>',
    ]


def _write(elems: list[str], out_path: str | Path) -> Path:
    elems.append("</svg>")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(elems), encoding="utf-8")
    return out_path


# --------------------------------------------------------------------------
# 1. classifier pipeline
# --------------------------------------------------------------------------
def render_classifier_svg(
    cfg: dict,
    out_path: str | Path,
    *,
    member_index: int = 0,
    n_members: int | None = None,
) -> Path:
    """Write the single-member classifier pipeline diagram to ``out_path``.

    Everything — spatial sizes, channel widths, loss weights, the head
    list and the parameter count — is read from ``cfg``, so the figure
    tracks the config instead of restating it.
    """
    mcfg = cfg.get("model", {})
    dcfg = cfg.get("dataset", {})
    tcfg = cfg.get("training", {})
    channels = list(mcfg.get("channels", (32, 64, 128, 256, 256)))
    k = int(mcfg.get("kernel_size", 3))
    img = int(mcfg.get("input_size", dcfg.get("image_size", 64)))
    n_attrs = int(mcfg.get("n_attrs", 6))
    n_gender = int(mcfg.get("n_gender", 2))
    feat = channels[-1]
    spatial = _spatial_sizes(img, len(channels))
    attr_w = float(tcfg.get("attr_loss_weight", 1.0))
    n_members = int(n_members if n_members is not None
                    else cfg.get("ensemble", {}).get("size", 1))

    # --- rows -------------------------------------------------------------
    rows: list[dict] = [{
        "badge": ("INPUT", _BLUE_BG, _BLUE_FG),
        "title": "Image x",
        "lines": [f"CelebA aligned  ·  Resize {img}{_SQ}  ·  ",
                  "ToTensor ∈ [0,1]"],
        "tag": (f"3×{img}×{img}", _BLUE_BG, _BLUE_FG),
    }]
    for i, c in enumerate(channels):
        pooled = i < len(channels) - 1
        tail = "MaxPool 2×2" if pooled else "(no pooling)"
        rows.append({
            "badge": ("CONV BLOCK", _TEAL_BG, _TEAL_FG),
            "title": f"Block {i + 1}",
            "lines": [f"Conv {k}×{k} → {c}  ·  BatchNorm2d +",
                      f"ReLU  ·  {tail}"],
            "tag": (f"{c}×{spatial[i]}×{spatial[i]}", _TEAL_BG, _TEAL_FG),
        })
    rows.append({
        "badge": ("POOLED", _BLUE_BG, _BLUE_FG),
        "title": "GAP",
        "lines": ["AdaptiveAvgPool2d(1×1)"],
        "tag": ("h ∈ ℝ", _BLUE_BG, _BLUE_FG),
        "tag_sup": str(feat),
        "tag_math": True,
    })

    heads: list[dict] = [
        {"badge": ("TASK HEAD", _PURPLE_BG, _PURPLE_FG),
         "title": "head_gender",
         "lines": [f"Linear {feat}→{n_gender}",
                   "softmax → p_g  ·  CE(p_g,", "Male)"],
         # MSP was dropped from the codebase (see lrad/evaluate.py) — the
         # gender head feeds the entropy and energy scores only.
         "accent": "→ entropy, energy", "accent_fill": _PURPLE_FG},
        {"badge": ("TASK HEAD", _PURPLE_BG, _PURPLE_FG),
         "title": "head_attrs",
         "lines": [f"Linear {feat}→{n_attrs}",
                   "sigmoid → p_a  ·  BCE,", f"weight {_num(attr_w)}"],
         "accent": f"{n_attrs} attributes, 4 periocular",
         "accent_fill": _PURPLE_FG},
    ]
    # --- layout -----------------------------------------------------------
    cw, ch_, gap = 800.0, 100.0, 34.0
    cx = 70.0
    top = 46.0
    width = cw + 2 * cx
    head_w = (cw - 2 * 24) / 3
    head_h = 232.0
    branch = 54.0
    head_top = top + len(rows) * (ch_ + gap) + branch
    height = head_top + head_h + 92

    e = _header(width, height, "LRAD — classifier pipeline")
    mid = cx + cw / 2

    for i, row in enumerate(rows):
        y = top + i * (ch_ + gap)
        e.append(_card(cx, y, cw, ch_))
        label, bg, fg = row["badge"]
        pill, _ = _pill(cx + 30, y + 18, label, bg, fg)
        e.append(pill)
        e.append(_text(cx + 30, y + 74, row["title"], size=21, fill=_INK,
                       bold=True))
        e.append(f'<line x1="{cx + 262:.1f}" y1="{y + 20:.1f}" '
                 f'x2="{cx + 262:.1f}" y2="{y + ch_ - 20:.1f}" '
                 f'stroke="{_RULE}" stroke-width="1.4"/>')
        ly = y + (ch_ / 2) - (len(row["lines"]) - 1) * 11 + 5
        for j, line in enumerate(row["lines"]):
            e.append(_text(cx + 292, ly + j * 22, line, size=14.5,
                           fill=_BODY))
        tag, tbg, tfg = row["tag"]
        e.append(_tag(cx + cw - 28, y + ch_ / 2, tag, tbg, tfg,
                      sup=row.get("tag_sup"),
                      math=bool(row.get("tag_math"))))
        if i < len(rows) - 1:
            e.append(_vline_arrow(mid, y + ch_ + 7, y + ch_ + gap - 3))

    # branch from GAP down to the head row
    gap_bottom = top + len(rows) * (ch_ + gap) - gap
    bar_y = gap_bottom + 30
    head_xs = [cx + i * (head_w + 24) for i in range(len(heads))]
    centres = [hx + head_w / 2 for hx in head_xs]
    e.append(f'<line x1="{mid:.1f}" y1="{gap_bottom + 7:.1f}" '
             f'x2="{mid:.1f}" y2="{bar_y:.1f}" stroke="{_INK}" '
             f'stroke-width="2.2"/>')
    e.append(f'<line x1="{min(centres):.1f}" y1="{bar_y:.1f}" '
             f'x2="{max(centres):.1f}" y2="{bar_y:.1f}" stroke="{_INK}" '
             f'stroke-width="2.2"/>')
    for c_x in centres:
        e.append(_vline_arrow(c_x, bar_y, head_top - 4))

    for hx, head in zip(head_xs, heads):
        e.append(_card(hx, head_top, head_w, head_h))
        label, bg, fg = head["badge"]
        pill, _ = _pill(hx + 26, head_top + 20, label, bg, fg)
        e.append(pill)
        e.append(_text(hx + 26, head_top + 76, head["title"], size=19,
                       fill=_INK, bold=True))
        for j, line in enumerate(head["lines"]):
            e.append(_text(hx + 26, head_top + 108 + j * 25, line,
                           size=14, fill=_BODY))
        e.append(_text(hx + 26, head_top + 200, head["accent"], size=13.5,
                       fill=head["accent_fill"]))

    # --- legend / provenance ---------------------------------------------
    ly = head_top + head_h + 46
    legend = [("tensor", _BLUE_BG, _BLUE_FG), ("conv block", _TEAL_BG,
                                               _TEAL_FG),
              ("task head", _PURPLE_BG, _PURPLE_FG)]
    lx = cx
    for name, bg, fg in legend:
        e.append(f'<rect x="{lx:.1f}" y="{ly - 10:.1f}" width="13" '
                 f'height="13" rx="4" fill="{bg}" stroke="{fg}" '
                 f'stroke-width="1.1"/>')
        e.append(_text(lx + 20, ly, name, size=12, fill=_MUTED))
        lx += 24 + _tw(name, 12) + 22
    e.append(_text(cx + cw, ly, f"FacialCNN — member {member_index + 1} of "
                                f"{n_members}  ·  "
                                f"{classifier_n_params(mcfg):,} params",
                   size=12, fill=_FAINT, anchor="end", mono=True))
    e.append(_text(cx + cw, ly + 20,
                   "lrad/model.py · lrad/train.py",
                   size=12, fill=_FAINT, anchor="end", mono=True))
    return _write(e, out_path)


# --------------------------------------------------------------------------
# 2. decoder pipeline
# --------------------------------------------------------------------------
def render_decoder_svg(
    cfg: dict,
    out_path: str | Path,
    *,
    min_channels: int = 16,
) -> Path:
    """Write the per-block decoder-stack diagram to ``out_path``."""
    mcfg = cfg.get("model", {})
    dcfg = cfg.get("dataset", {})
    channels = list(mcfg.get("channels", (32, 64, 128, 256, 256)))
    img = int(mcfg.get("input_size", dcfg.get("image_size", 64)))
    spatial = _spatial_sizes(img, len(channels))

    lanes = []
    for i, (c, s) in enumerate(zip(channels, spatial)):
        n_up = int(round(math.log2(img // s)))
        path = decoder_channel_path(c, n_up, min_channels)
        stages, cur, size = [], c, s
        for out_ch in path:
            size *= 2
            stages.append({"label": f"ConvT {cur}→{out_ch}",
                           "shape": f"{out_ch} × {size} × {size}"})
            cur = out_ch
        lanes.append({
            "index": i, "in_ch": c, "in_size": s, "n_up": n_up,
            "stages": stages,
            "params": decoder_n_params(c, s, img,
                                       min_channels=min_channels),
        })
    max_stages = max(len(ln["stages"]) for ln in lanes)
    total = sum(ln["params"] for ln in lanes)

    # --- layout -----------------------------------------------------------
    in_w, st_w, hd_w, out_w = 220.0, 196.0, 210.0, 190.0
    col_gap, lane_h, lane_gap = 34.0, 150.0, 22.0
    left, top = 56.0, 118.0
    st_x0 = left + in_w + col_gap
    hd_x = st_x0 + max_stages * (st_w + col_gap) + 26
    out_x = hd_x + hd_w + col_gap
    width = out_x + out_w + left
    height = top + len(lanes) * (lane_h + lane_gap) + 96

    e = _header(width, height, "LRAD — per-block decoder stack")
    e.append(_text(left, 44, "INPUT ACTIVATION", size=12, fill=_MUTED,
                   bold=True, tracking=1.0))
    e.append(_text(st_x0, 34, "Decoder stages  ·  ConvTranspose2d(4×4, "
                              "stride 2, pad 1) → BN → ReLU — channels "
                              "halve each stage,", size=12.5, fill=_MUTED))
    e.append(_text(st_x0, 52, f"floor {min_channels}", size=12.5,
                   fill=_MUTED))
    e.append(_text(hd_x, 44, "HEAD", size=12, fill=_MUTED, bold=True,
                   tracking=1.0))
    e.append(_text(out_x, 44, "OUTPUT", size=12, fill=_MUTED, bold=True,
                   tracking=1.0))
    e.append(f'<line x1="{left:.1f}" y1="{top - 26:.1f}" '
             f'x2="{width - left:.1f}" y2="{top - 26:.1f}" '
             f'stroke="#d4d8de" stroke-width="1.4"/>')

    for row, ln in enumerate(lanes):
        y = top + row * (lane_h + lane_gap)
        cy = y + lane_h / 2
        i = ln["index"]
        deepest = i == len(lanes) - 1

        # input activation card
        e.append(_card(left, y, in_w, lane_h))
        pill, _ = _pill(left + 20, y + 18, "INPUT", _BLUE_BG, _BLUE_FG)
        e.append(pill)
        e.append(_text_parts(left + 20, y + 68,
                             [("a", 0), (str(i), -1),
                              (f" — block {i + 1}", 0)],
                             size=15, fill=_INK, bold=True))
        yy = y + 90
        if deepest:
            e.append(_text(left + 20, yy, "(deepest)", size=12.5,
                           fill=_FAINT))
            yy += 20
        e.append(_text(left + 20, yy, f"{ln['in_ch']} × {ln['in_size']} × "
                                      f"{ln['in_size']}", size=12.5,
                       fill=_MUTED))
        if row == 0:
            e.append(_text_parts(
                left + 20, yy + 20,
                [("n_up = log", 0), ("2", -1),
                 (f"({img}/{ln['in_size']}) = {ln['n_up']}", 0)],
                size=12.5, fill=_MUTED))
        else:
            e.append(_text(left + 20, yy + 20, f"n_up = {ln['n_up']}",
                           size=12.5, fill=_MUTED))

        # conv stages
        for j, st in enumerate(ln["stages"]):
            sx = st_x0 + j * (st_w + col_gap)
            e.append(_card(sx, y, st_w, lane_h))
            pill, _ = _pill(sx + 18, y + 18, "CONV", _TEAL_BG, _TEAL_FG)
            e.append(pill)
            e.append(_text(sx + 18, y + 68, st["label"], size=14.5,
                           fill=_INK, bold=True))
            e.append(_text(sx + 18, y + 92, st["shape"], size=12.5,
                           fill=_MUTED))
            e.append(_text(sx + 18, y + 114, "BN + ReLU", size=12.5,
                           fill=_MUTED))
            prev_right = (left + in_w if j == 0
                          else sx - col_gap)
            e.append(_hline_arrow(cy, prev_right + 6, sx - 6))

        # dashed run to the head across the stages this lane does not use
        last_right = (st_x0 + (len(ln["stages"]) - 1) * (st_w + col_gap)
                      + st_w)
        if len(ln["stages"]) < max_stages:
            e.append(_hline_arrow(cy, last_right + 8, hd_x - 8,
                                  colour="#8b93a0", dashed=True,
                                  width=1.8))
        else:
            e.append(_hline_arrow(cy, last_right + 6, hd_x - 6))

        # output head
        e.append(_card(hd_x, y, hd_w, lane_h))
        pill, _ = _pill(hd_x + 20, y + 18, "HEAD", _PURPLE_BG, _PURPLE_FG)
        e.append(pill)
        e.append(_text(hd_x + 20, y + 68, "Conv 1×1 → 3", size=14.5,
                       fill=_INK, bold=True))
        e.append(_text(hd_x + 20, y + 92, "Sigmoid", size=12.5, fill=_MUTED))
        if row == 0:
            e.append(_text(hd_x + 20, y + 114, "output ∈ [0,1]", size=12.5,
                           fill=_MUTED))
        e.append(_hline_arrow(cy, hd_x + hd_w + 6, out_x - 6))

        # reconstruction
        e.append(_card(out_x, y, out_w, lane_h))
        pill, _ = _pill(out_x + 18, y + 18, "OUTPUT", _PEACH_BG, _PEACH_FG)
        e.append(pill)
        # "f-hat" with the circumflex drawn rather than set as U+0302:
        # combining marks over Latin letters render inconsistently, and
        # several common sans faces have no glyph for them at all.
        e.append(_text_parts(out_x + 18, y + 70,
                             [("f", 0), (str(i), -1), ("(x)", 0)],
                             size=16, fill=_INK, bold=True))
        hx = out_x + 18
        e.append(f'<path d="M{hx + 1.4:.1f},{y + 58.5:.1f} '
                 f'L{hx + 4.6:.1f},{y + 54.6:.1f} '
                 f'L{hx + 7.8:.1f},{y + 58.5:.1f}" fill="none" '
                 f'stroke="{_INK}" stroke-width="1.6" '
                 f'stroke-linecap="round" stroke-linejoin="round"/>')
        e.append(_text(out_x + 18, y + 94, f"3×{img}×{img}", size=12.5,
                       fill=_MUTED))
        e.append(_text(out_x + 18, y + 116, f"{ln['params']:,} par.",
                       size=12.5, fill=_FAINT))

    # --- legend / provenance ---------------------------------------------
    ly = top + len(lanes) * (lane_h + lane_gap) + 34
    e.append(f'<line x1="{left:.1f}" y1="{ly - 4:.1f}" '
             f'x2="{left + 30:.1f}" y2="{ly - 4:.1f}" stroke="{_INK}" '
             f'stroke-width="2.2"/>')
    e.append(_text(left + 40, ly, "sequential step", size=12, fill=_MUTED))
    dx = left + 40 + _tw("sequential step", 12) + 28
    e.append(f'<line x1="{dx:.1f}" y1="{ly - 4:.1f}" x2="{dx + 30:.1f}" '
             f'y2="{ly - 4:.1f}" stroke="#8b93a0" stroke-width="1.8" '
             f'stroke-dasharray="6 5"/>')
    e.append(_text(dx + 40, ly, "skipped stages (channel floor reached)",
                   size=12, fill=_MUTED))
    e.append(_text(width - left, ly, f"{total:,} params  ·  "
                                     f"lrad/decoder.py · "
                                     f"lrad/train.py:train_decoders",
                   size=12, fill=_FAINT, anchor="end", mono=True))
    return _write(e, out_path)


# --------------------------------------------------------------------------
# 3. ensemble member architectures
# --------------------------------------------------------------------------
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

    cfg0 = member_cfgs[0]
    channels0 = cfg0.get("model", {}).get("channels", [32, 64, 128, 256, 256])
    n_blocks = len(channels0)
    image_size = int(cfg0.get("model", {}).get(
        "input_size", cfg0.get("dataset", {}).get("image_size", 64)))
    spatial = _spatial_sizes(image_size, n_blocks)
    max_ch = max(max(c.get("model", {}).get("channels", [1]))
                 for c in member_cfgs)

    # --- layout -----------------------------------------------------------
    label_w = 300.0
    bw, bgap = 104.0, 56.0
    input_w = 62.0
    head_w = 148.0
    lane_h = 122.0
    left, top = 26.0, 128.0
    blk_x0 = left + label_w + input_w + bgap
    gap_x = blk_x0 + n_blocks * bw + (n_blocks - 1) * bgap + bgap + 22
    head_x = gap_x + 44
    width = head_x + head_w + left
    height = top + lanes * lane_h + 110
    base_off = 74.0                     # block baseline below lane top
    ch_scale = 56.0 / max(max_ch, 1)    # tallest block ≈ 56 px

    e = _header(width, height, title)
    e.append(_text(left, 34, f"Each block: Conv(k×k, pad k/2, no bias) + "
                             f"BatchNorm + ReLU; ×½ = MaxPool 2×2 (all "
                             f"blocks but the last). Rectangle height ∝ "
                             f"channel width; colour = kernel size.",
                   size=12.5, fill=_MUTED))
    e.append('<defs>')
    for kk, (c0, c1) in _KERNEL_GRAD.items():
        e.append(f'<linearGradient id="k{kk}" x1="0" y1="0" x2="1" y2="1">'
                 f'<stop offset="0" stop-color="{c0}"/>'
                 f'<stop offset="1" stop-color="{c1}"/></linearGradient>')
    e.append('</defs>')

    # column header
    hy = top - 26
    e.append(_text(left, hy, "MODEL", size=11.5, fill=_MUTED, bold=True,
                   tracking=1.0))
    for b in range(n_blocks):
        cx = blk_x0 + b * (bw + bgap) + bw / 2
        e.append(_text(cx, hy, f"L{b} · {spatial[b]}"
                               f"{_SQ}", size=11.5,
                       fill=_MUTED, bold=True, anchor="middle",
                       tracking=0.6))
    e.append(_text(head_x + head_w, hy, "PREDICTION HEADS", size=11.5,
                   fill=_MUTED, bold=True, anchor="end", tracking=1.0))
    e.append(f'<line x1="{left:.1f}" y1="{hy + 12:.1f}" '
             f'x2="{width - left:.1f}" y2="{hy + 12:.1f}" '
             f'stroke="#d4d8de" stroke-width="1.4"/>')

    for i, cfg in enumerate(member_cfgs):
        mcfg = cfg.get("model", {})
        channels = list(mcfg.get("channels", []))
        k = int(mcfg.get("kernel_size", 3))
        grad = f"url(#k{k})" if k in _KERNEL_GRAD else _GRAD_DEFAULT[0]
        swatch = _KERNEL_GRAD.get(k, _GRAD_DEFAULT)[1]
        rec = (per_model[i] if per_model and i < len(per_model) else {})
        n_params = int(rec.get("n_params") or classifier_n_params(mcfg))
        seed = rec.get("seed")

        y0 = top + i * lane_h
        base = y0 + base_off
        e.append(f'<g id="model_{i}">')
        if i > 0:
            e.append(f'<line x1="{left:.1f}" y1="{y0:.1f}" '
                     f'x2="{width - left:.1f}" y2="{y0:.1f}" '
                     f'stroke="#f0f2f5" stroke-width="1"/>')

        # left label column
        e.append(f'<rect x="{left:.1f}" y="{y0 + 18:.1f}" width="11" '
                 f'height="11" rx="3" fill="{swatch}"/>')
        e.append(_text(left + 20, y0 + 28, f"Model {i + 1}", size=15,
                       fill=_INK, bold=True))
        seed_txt = f"seed {seed} · " if seed is not None else ""
        e.append(_text(left, y0 + 50,
                       f"{seed_txt}kernel {k}×{k} · "
                       f"{_fmt_params(n_params)} params",
                       size=12, fill=_MUTED))
        e.append(_text(left, y0 + 68,
                       "channels " + "-".join(str(c) for c in channels),
                       size=12, fill=_FAINT))

        # input tile
        ix = left + label_w
        e.append(_text(ix + input_w / 2, base - 30,
                       f"3×{image_size}{_SQ}", size=10.5,
                       fill=_FAINT, anchor="middle"))
        e.append(f'<rect x="{ix:.1f}" y="{base - 20:.1f}" '
                 f'width="{input_w:.1f}" height="20" rx="5" fill="#ffffff" '
                 f'stroke="#c3cad3" stroke-width="1.2"/>')

        # conv blocks
        for b, c in enumerate(channels):
            bx = blk_x0 + b * (bw + bgap)
            prev_right = ix + input_w if b == 0 else bx - bgap
            e.append(_hline_arrow(base - 20, prev_right + 6, bx - 6,
                                  colour="#8b93a0", width=1.5))
            if b < len(channels) - 1:
                e.append(_text(bx - bgap / 2, base - 28, "×½", size=10.5,
                               fill=_FAINT, anchor="middle"))
            bh = 20 + c * ch_scale
            e.append(f'<rect x="{bx:.1f}" y="{base - bh:.1f}" '
                     f'width="{bw:.1f}" height="{bh:.1f}" rx="7" '
                     f'fill="{grad}"/>')
            e.append(_text(bx + bw / 2, base - bh / 2 + 5, str(c), size=14,
                           fill="#ffffff", bold=True, anchor="middle"))
            # per-block decoder tap — the footnote below explains it, so it
            # has to actually be drawn.
            e.append(f'<line x1="{bx + bw / 2:.1f}" y1="{base + 6:.1f}" '
                     f'x2="{bx + bw / 2:.1f}" y2="{base + 18:.1f}" '
                     f'stroke="#b6bec8" stroke-width="1" '
                     f'stroke-dasharray="3 2.5"/>')
            e.append(_text(bx + bw / 2, base + 30, f"dec L{b}", size=9.5,
                           fill=_FAINT, anchor="middle"))

        # GAP + heads
        last_right = blk_x0 + (n_blocks - 1) * (bw + bgap) + bw
        e.append(_hline_arrow(base - 20, last_right + 6, gap_x - 20,
                              colour="#8b93a0", width=1.5))
        e.append(f'<circle cx="{gap_x:.1f}" cy="{base - 20:.1f}" r="16" '
                 f'fill="#ffffff" stroke="#a7b0bd" stroke-width="1.2"/>')
        e.append(_text(gap_x, base - 16, "GAP", size=10, fill=_MUTED,
                       anchor="middle"))

        n_attrs = int(mcfg.get("n_attrs", 6))
        heads = [("gender", int(mcfg.get("n_gender", 2))),
                 ("attrs", n_attrs)]
        hh, hgap = 21.0, 6.0
        hy0 = (base - 20) - (len(heads) * hh + (len(heads) - 1) * hgap) / 2
        for j, (name, n_out) in enumerate(heads):
            hyj = hy0 + j * (hh + hgap)
            e.append(f'<line x1="{gap_x + 16:.1f}" y1="{base - 20:.1f}" '
                     f'x2="{head_x:.1f}" y2="{hyj + hh / 2:.1f}" '
                     f'stroke="#c3cad3" stroke-width="1"/>')
            e.append(f'<rect x="{head_x:.1f}" y="{hyj:.1f}" '
                     f'width="{head_w:.1f}" height="{hh:g}" rx="6" '
                     f'fill="#ffffff" stroke="#cbd5e1" stroke-width="1"/>')
            # Centred by offset rather than text-anchor: the label holds
            # an arrow, which _rich puts in its own <tspan>, and some
            # renderers re-anchor every tspan to the same x.
            hlabel = f"{name} → {n_out}"
            e.append(_text(head_x + (head_w - _tw(hlabel, 11)) / 2,
                           hyj + hh - 6.5, hlabel, size=11, fill=_MUTED))
        e.append('</g>')

    # --- legend / footnote ------------------------------------------------
    ly = top + lanes * lane_h + 40
    lx = left
    for kk in sorted(_KERNEL_GRAD):
        e.append(f'<rect x="{lx:.1f}" y="{ly - 11:.1f}" width="13" '
                 f'height="13" rx="4" fill="url(#k{kk})"/>')
        e.append(_text(lx + 20, ly, f"kernel {kk}×{kk}", size=12,
                       fill=_MUTED))
        lx += 20 + _tw(f"kernel {kk}×{kk}", 12) + 34
    e.append(f'<rect x="{lx:.1f}" y="{ly - 8:.1f}" width="13" height="7" '
             f'rx="1" fill="none" stroke="#b6bec8" stroke-width="1" '
             f'stroke-dasharray="3 2.5"/>')
    e.append(_text(lx + 20, ly, "decoder tap", size=12, fill=_MUTED))
    e.append(_text(left, ly + 26,
                   f"dec Lk: frozen-trunk BlockDecoder (ConvTranspose 4×4 "
                   f"stride-2 stages + BN + ReLU) reconstructing the "
                   f"{image_size}{_SQ} input from block k "
                   f"— the reconstructions drive the ensemble bias/"
                   f"variance decomposition.", size=12, fill=_FAINT))
    return _write(e, out_path)
