#!/bin/bash
# ---------------------------------------------------------------------------
# OAR submission script for Grid'5000 — MVTec AD per-category LRAD ensembles
# (configs/mvtec.yaml), for the PatchCore comparison.
#
# COST. MVTec categories are tiny (60-391 training images against ~180k
# CelebA faces), which makes an EPOCH cheap but is exactly why the schedule
# is set in optimizer STEPS instead (configs/mvtec.yaml, training.steps /
# training.decoders.steps) — a fixed epoch count would train toothbrush
# 6.5x less than hazelnut and leave both far short of convergence.
#
# The budget is 4,000 classifier + 8,000 decoder steps per member. At 224 px
# on an A40 that is roughly 30-45 min per member, so:
#
#     4-category pilot x 4 members  ~=  8-12 h    (fits this walltime)
#     all 15 categories x 4 members ~=  30-45 h   (needs 48 h — see below)
#
# These are estimates, not measurements. RUN THE PILOT FIRST: the log prints
# a per-member wall time, which turns the numbers above into real ones before
# committing to the full sweep. results.md is rewritten after every category,
# so a job killed at the walltime still leaves a complete table for whatever
# finished — nothing is lost by starting and re-submitting the remainder.
#
# For all 15 categories, raise the walltime to 48:00:00 below, or split the
# sweep across two jobs with CATEGORIES=... (see below).
#
# BEFORE submitting: fetch the dataset once on the frontend (it needs
# network access the compute nodes may not have):
#   python scripts/download_mvtec.py --root ~/lrad/data
#
# Submit (from the Nancy frontend, in ~/lrad):
#   ./scripts/oar_run_mvtec.sh              # pilot categories from the config
#   CATEGORIES=all ./scripts/oar_run_mvtec.sh
#   CATEGORIES="bottle cable capsule carpet grid hazelnut leather metal_nut" \
#       ./scripts/oar_run_mvtec.sh          # first half of the full sweep
# Track:   oarstat -u $USER ; tail -f outputs/mvtec/_oar/oar.<jobid>.stdout
# Cancel:  oardel <jobid>
# ---------------------------------------------------------------------------

#OAR -n mvtec-lrad-gruss
#OAR -q production
#OAR -p cluster='gruss'
#OAR -l gpu=1,walltime=12:00:00
#OAR -O outputs/mvtec/_oar/oar.%jobid%.stdout
#OAR -E outputs/mvtec/_oar/oar.%jobid%.stderr

set -euo pipefail

if [[ -z "${OAR_JOB_ID:-}" ]]; then
    cd "$HOME/lrad"
    mkdir -p outputs/mvtec/_oar
    echo "Submitting to the production queue via oarsub -S $0"
    exec oarsub -S "$0"
fi

cd "$HOME/lrad"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/mvtec/run_${RUN_TS}_${OAR_JOB_ID}"
mkdir -p "$OUTPUT_DIR"/logs
echo "OUTPUT_DIR=$OUTPUT_DIR"

PYTHON="$HOME/.conda/envs/lrad/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: lrad Python not found at $PYTHON" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

echo "=== OAR job ${OAR_JOB_ID} on $(hostname) [gruss] at $(date) ==="
nvidia-smi || echo "nvidia-smi not available"
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

DATA_ROOT="$HOME/lrad/data"

# Fail early and loudly if the dataset was never fetched — otherwise every
# category raises the same FileNotFoundError one after another and the real
# cause is buried in the middle of the log.
if ! "$PYTHON" -u scripts/download_mvtec.py --root "$DATA_ROOT" --verify-only; then
    echo "ERROR: MVTec AD is missing or incomplete under $DATA_ROOT." >&2
    echo "Run this on the FRONTEND first:" >&2
    echo "  python scripts/download_mvtec.py --root $DATA_ROOT" >&2
    exit 1
fi

CATEGORIES="${CATEGORIES:-}"
CAT_ARGS=()
[[ -n "$CATEGORIES" ]] && CAT_ARGS=(--categories $CATEGORIES)

set +e
"$PYTHON" -u scripts/run_mvtec.py \
    --config configs/mvtec.yaml \
    --output-dir "$OUTPUT_DIR" \
    "${CAT_ARGS[@]}" \
    --override dataset.root="$DATA_ROOT" dataset.num_workers=4
RC=$?
set -e

if [[ $RC -eq 0 ]]; then
    echo "--- results table ---"
    cat "$OUTPUT_DIR/results.md" || true
fi

echo "=== job ${OAR_JOB_ID} finished at $(date) with exit code ${RC} ==="
exit "${RC}"
