#!/usr/bin/env python3
"""Inter-model variability sigma(e) as a function of training epochs.

For a deep ensemble of ``M`` models with per-epoch checkpoints (see
``training.save_every_epoch`` in the config), this script traces the curve

    sigma(e) = sqrt( mean_{test image I} mean_{pixel p} Var_m[ err_m(I, p) ] )

where ``err_m(I, p)`` is model ``m``'s squared reconstruction error at
pixel ``p`` (RGB-summed, consistent with ``lrad/ensemble.py``).
``mean_I mean_p Var_m`` is exactly the mean of the **Variance** term of the
``Risk = Bias + Variance`` decomposition — the per-pixel spread of the ``M``
reconstructions around their ensemble mean — and ``sigma`` is its square
root. It is computed at the deepest reconstruction depth (last conv block)
by default; pass ``--all-blocks`` for one curve per block.

PROTOCOL — simple (decoder-epochs) variant
------------------------------------------
The decoders are trained on the FINAL **frozen** classifier, so the only
per-epoch checkpoints that are mutually consistent within a single run are
the decoder ones: a classifier checkpoint at epoch ``e`` was never paired
with decoders trained on it. This script therefore implements the *simple*
variant on purpose:

  * classifier: fixed at its full schedule (``model.pt``),
  * decoders:   per-epoch checkpoint ``decoders_ep{e}.pt``,

so ``sigma(e)`` is a function of the DECODER training epochs only, read
from the per-epoch checkpoints of one single run. The fully-coupled
variant (classifier at epoch ``e`` + decoders re-trained ``e`` epochs on
that classifier) would require one complete decoder run per classifier
epoch and is deliberately left out.

Everything is computed in streaming over the test loader — only running
sums are kept, never the full error maps (O(1) memory, like
``collect_decomposition_scores``).

Outputs (under ``--output-dir``, default ``<run-dir>/epoch_variability``):
  * ``sigma_vs_epochs.csv``  — rows ``epoch,split,block,sigma``
  * ``sigma_vs_epochs.json`` — ``{split: {block: {epoch: sigma}}}``
  * ``sigma_vs_epochs.png``  — the sigma(e) curve(s)

Usage:
    python scripts/epoch_variability_study.py \\
        --config configs/celeba_ood.yaml \\
        --run-dir outputs/celeba_ood/ensemble_run \\
        [--epochs 1 2 5 10 20 30] [--all-blocks] [--include-ood]
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_ROOT), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from run_celeba import apply_overrides, load_config  # noqa: E402

from lrad.dataset import get_celeba_loaders  # noqa: E402
from lrad.decoder import build_decoders  # noqa: E402
from lrad.ensemble import block_reconstructions  # noqa: E402
from lrad.model import build_model  # noqa: E402
from lrad.utils import get_device, setup_logging  # noqa: E402

logger = logging.getLogger("celeba_ood")


def _discover_members(run_dir: Path) -> list[Path]:
    """Sorted ``model_<i>/`` member directories of an ensemble run."""
    members = sorted(
        (d for d in run_dir.glob("model_*") if d.is_dir()),
        key=lambda d: int(d.name.split("_")[-1]),
    )
    if not members:
        raise FileNotFoundError(f"no model_<i>/ directories under {run_dir}")
    return members


def _discover_epochs(members: list[Path]) -> list[int]:
    """Epochs whose decoder checkpoint exists for EVERY ensemble member."""
    pat = re.compile(r"decoders_ep(\d+)\.pt$")
    common: set[int] | None = None
    for m in members:
        eps = {
            int(g.group(1))
            for f in (m / "weights").glob("decoders_ep*.pt")
            if (g := pat.search(f.name))
        }
        common = eps if common is None else (common & eps)
    epochs = sorted(common or set())
    if not epochs:
        raise FileNotFoundError(
            "no per-epoch decoder checkpoints (decoders_ep<e>.pt) found in "
            "every member — re-run training with "
            "training.save_every_epoch: true"
        )
    return epochs


@torch.no_grad()
def compute_sigma(
    models: list,
    decoders_list: list,
    loader,
    device: torch.device,
    blocks: list[int],
) -> dict[int, float]:
    """sigma per block, streaming over ``loader`` (O(1) memory).

    Accumulates, per block, the sum over images and pixels of the Variance
    term ``mean_m (f_hat^m - f_bar)^2`` (RGB-summed), then returns
    ``sqrt(sum / n_pixels)`` — i.e. sigma = sqrt of the mean per-pixel
    inter-model variance of the reconstruction error decomposition.
    """
    sums = {k: 0.0 for k in blocks}
    n_px = 0
    for batch in loader:
        img = batch[0].to(device, non_blocking=True)
        recons_per_model = [
            block_reconstructions(models[m], decoders_list[m], img)
            for m in range(len(models))
        ]
        for k in blocks:
            stack = torch.stack(
                [r[k] for r in recons_per_model], dim=0,
            )  # (M, B, 3, H, W)
            mean = stack.mean(dim=0)
            var = (
                ((stack - mean.unsqueeze(0)) ** 2).sum(dim=2).mean(dim=0)
            )  # (B, H, W)
            sums[k] += float(var.sum().item())
        n_px += img.size(0) * img.shape[-2] * img.shape[-1]
    return {k: math.sqrt(sums[k] / max(n_px, 1)) for k in blocks}


def _plot_sigma(
    epochs: list[int],
    sigma: dict[str, dict[int, dict[int, float]]],
    save_path: Path,
) -> None:
    """sigma(e) curves — one line per (split, block)."""
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)
    styles = {"in": ("-o", "#377eb8"), "ood": ("--s", "#e41a1c")}
    for split, per_block in sigma.items():
        fmt, color = styles.get(split, ("-", None))
        blocks = sorted(per_block.keys())
        many = len(blocks) > 1
        for k in blocks:
            ys = [per_block[k][e] for e in epochs]
            label = f"{split} (block {k})" if many else split
            ax.plot(epochs, ys, fmt, color=None if many else color,
                    label=label)
    ax.set_xlabel("Decoder training epoch")
    ax.set_ylabel(r"$\sigma$ = sqrt(mean inter-model variance)")
    ax.set_title("Inter-model variability vs training epochs")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Trace the inter-model variability sigma(e) across "
                    "per-epoch decoder checkpoints of an ensemble run.",
    )
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--run-dir", type=Path, required=True,
                    help="Ensemble output root holding model_<i>/ dirs.")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Default: <run-dir>/epoch_variability.")
    ap.add_argument("--epochs", type=int, nargs="*", default=None,
                    help="Epoch grid. Default: every epoch whose decoder "
                         "checkpoint exists for all members.")
    ap.add_argument("--all-blocks", action="store_true",
                    help="One sigma curve per conv block "
                         "(default: deepest block only).")
    ap.add_argument("--include-ood", action="store_true",
                    help="Also trace sigma(e) on the OOD test set.")
    ap.add_argument("--override", nargs="*", default=[],
                    help="Dotted overrides, e.g. dataset.batch_size=128.")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg = apply_overrides(cfg, args.override)

    output_dir = Path(args.output_dir or (args.run_dir / "epoch_variability"))
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(str(output_dir), run_tag="epoch_variability")
    device = get_device()

    members = _discover_members(args.run_dir)
    epochs = sorted(args.epochs) if args.epochs else _discover_epochs(members)
    logger.info(f"Members : {len(members)}  ({members[0].parent})")
    logger.info(f"Epochs  : {epochs}")
    logger.info(f"Device  : {device}")

    loaders = get_celeba_loaders(cfg)
    splits = {"in": loaders["test_in"]}
    if args.include_ood:
        splits["ood"] = loaders["test_ood"]

    # Classifiers are FIXED at their full schedule (simple variant — see
    # module docstring); only the decoder weights change with the epoch.
    image_size = cfg.get("dataset", {}).get("image_size", 64)
    models, decoders_list = [], []
    for m_dir in members:
        model = build_model(cfg).to(device).eval()
        model.load_state_dict(
            torch.load(m_dir / "weights" / "model.pt", map_location=device),
        )
        models.append(model)
        decoders_list.append(
            build_decoders(model, image_size=image_size).to(device).eval(),
        )

    n_blocks = len(decoders_list[0])
    blocks = list(range(n_blocks)) if args.all_blocks else [n_blocks - 1]
    logger.info(f"Blocks  : {blocks}  (n_blocks={n_blocks})")

    sigma: dict[str, dict[int, dict[int, float]]] = {
        s: {k: {} for k in blocks} for s in splits
    }
    for e in epochs:
        for m_dir, decoders in zip(members, decoders_list):
            ckpt = m_dir / "weights" / f"decoders_ep{e}.pt"
            decoders.load_state_dict(torch.load(ckpt, map_location=device))
            decoders.eval()
        for split, loader in splits.items():
            res = compute_sigma(models, decoders_list, loader, device, blocks)
            for k, v in res.items():
                sigma[split][k][e] = v
            logger.info(
                f"epoch {e:>3}  [{split}]  " +
                "  ".join(f"sigma(block {k})={res[k]:.6f}" for k in blocks)
            )

    # --- Write CSV / JSON / plot -----------------------------------------
    csv_path = output_dir / "sigma_vs_epochs.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "split", "block", "sigma"])
        for split in sigma:
            for k in blocks:
                for e in epochs:
                    w.writerow([e, split, k, f"{sigma[split][k][e]:.8f}"])

    json_path = output_dir / "sigma_vs_epochs.json"
    with open(json_path, "w") as f:
        json.dump(
            {
                "protocol": "simple variant — classifier fixed at its full "
                            "schedule; sigma(e) varies the DECODER epochs "
                            "only (per-epoch checkpoints of a single run)",
                "epochs": epochs,
                "blocks": blocks,
                "n_models": len(members),
                "sigma": {
                    s: {str(k): {str(e): sigma[s][k][e] for e in epochs}
                        for k in blocks}
                    for s in sigma
                },
            },
            f, indent=2,
        )

    plot_path = output_dir / "sigma_vs_epochs.png"
    _plot_sigma(epochs, sigma, plot_path)
    logger.info(f"Wrote {csv_path}, {json_path}, {plot_path}")
    logger.info("All done.")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        if logger.handlers:
            logger.exception("epoch_variability_study crashed")
        else:
            traceback.print_exc()
        sys.exit(1)
