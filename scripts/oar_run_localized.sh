#!/bin/bash
# ---------------------------------------------------------------------------
# OAR submission script for Grid'5000 — localized (z-score + patch-max) OOD
# scoring on an ALREADY-TRAINED ensemble (see scripts/run_localized.py).
# Same graffiti pinning as oar_run_ensemble.sh, but this job is
# inference-only: 10 members × (~10k reference + 18 941 test_in + 13 193
# test_ood) images at 64x64 — well under an hour on a 2080 Ti. The 4 h
# walltime is pure safety margin, and a short job also fits the 24 h-max
# graffiti nodes 1–3, so it schedules faster than the 48 h training job.
#
# Reads the trained members from the ensemble run directory passed as the
# first argument (default: the 20260707 run) and writes
# <run>/ensemble/localized_auroc.json next to the existing summaries.
#
# Submit (from the Nancy frontend, in ~/lrad):
#   ./scripts/oar_run_localized.sh [outputs/celeba_ood/ensemble_...]
#
# Track:
#   oarstat -u $USER
#   tail -f outputs/celeba_ood/_oar/oar.<jobid>.stdout
#
# Cancel:
#   oardel <jobid>
# ---------------------------------------------------------------------------

#OAR -n celeba-ood-localized-graffiti
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
# default 40-batch cap (~10k train images, plenty for per-pixel mean/std).
set +e
"$PYTHON" -u scripts/run_localized.py \
    --output-dir "$ENSEMBLE_DIR" \
    --num-workers 4
RC=$?
set -e

echo "=== job ${OAR_JOB_ID} finished at $(date) with exit code ${RC} ==="
exit "${RC}"
