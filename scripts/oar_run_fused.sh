#!/bin/bash
# ---------------------------------------------------------------------------
# OAR submission script for Grid'5000 — FUSED OOD scoring (locfre +
# epistemic + energy, see lrad.fusion) plus the localized pixel scoring
# (lrad.localized) on an ALREADY-TRAINED ensemble. Inference only: both
# evaluations stream the full test splits (18 941 in / 13 193 OOD) through
# the 10 members once each — well under an hour total on a 2080 Ti; the
# 4 h walltime is pure safety margin and also fits the 24 h-max graffiti
# nodes 1–3, so it schedules fast.
#
# Reads the trained members from the ensemble run directory passed as the
# first argument (default: the 20260707 run) and writes
#   <run>/ensemble/fused_auroc.json      (headline: fused AUROC)
#   <run>/ensemble/localized_auroc.json  (pixel z-score+patch-max vs p95)
#
# Submit (from the Nancy frontend, in ~/lrad):
#   ./scripts/oar_run_fused.sh [outputs/celeba_ood/ensemble_...]
#
# Track:
#   oarstat -u $USER
#   tail -f outputs/celeba_ood/_oar/oar.<jobid>.stdout
#
# Cancel:
#   oardel <jobid>
# ---------------------------------------------------------------------------

#OAR -n celeba-ood-fused-graffiti
#OAR -q production
#OAR -p cluster='graffiti'
#OAR -l gpu=1,walltime=4:00:00
#OAR -O outputs/celeba_ood/_oar/oar.%jobid%.stdout
#OAR -E outputs/celeba_ood/_oar/oar.%jobid%.stderr

set -euo pipefail

ENSEMBLE_DIR="${1:-outputs/celeba_ood/ensemble_20260707_152254_6754917}"

# If invoked directly (not through oarsub), bootstrap and submit, forwarding
# the ensemble dir as an argument (env vars do not survive into OAR jobs).
if [[ -z "${OAR_JOB_ID:-}" ]]; then
    cd "$HOME/lrad"
    if [[ ! -d "$ENSEMBLE_DIR" ]]; then
        echo "ERROR: ensemble dir not found: $ENSEMBLE_DIR" >&2
        exit 1
    fi
    mkdir -p outputs/celeba_ood/_oar
    echo "Submitting to the production queue via oarsub -S \"$0 $ENSEMBLE_DIR\""
    exec oarsub -S "$0 $ENSEMBLE_DIR"
fi

cd "$HOME/lrad"

# --- Environment ---------------------------------------------------------
PYTHON="$HOME/.conda/envs/lrad/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: lrad Python not found at $PYTHON" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

# --- Diagnostics ---------------------------------------------------------
echo "=== OAR job ${OAR_JOB_ID} on $(hostname) [graffiti] at $(date) ==="
echo "PWD=$(pwd)"
echo "PYTHON=$PYTHON"
echo "ENSEMBLE_DIR=$ENSEMBLE_DIR"
nvidia-smi || echo "nvidia-smi not available"
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

# --- Run -----------------------------------------------------------------
# Full test splits (no --max-eval-batches); reference stats from the
# default 40-batch cap (~10k train images).
RC=0

echo "--- fused scoring (lrad.fusion, supervised calibration) ---"
# --supervised: logistic fusion calibrated on train negatives + a dedicated
# OOD half (split seed 42), evaluated on test_in vs the OTHER OOD half.
# The unsupervised rank fusion is reported alongside from the same pass.
set +e
"$PYTHON" -u scripts/run_fused.py \
    --output-dir "$ENSEMBLE_DIR" \
    --supervised --blocks 1 2 3 \
    --num-workers 4
RC=$?
set -e

echo "--- localized pixel scoring (lrad.localized) ---"
set +e
"$PYTHON" -u scripts/run_localized.py \
    --output-dir "$ENSEMBLE_DIR" \
    --num-workers 4
RC2=$?
set -e
[[ $RC -eq 0 ]] && RC=$RC2

echo "=== job ${OAR_JOB_ID} finished at $(date) with exit code ${RC} ==="
exit "${RC}"
