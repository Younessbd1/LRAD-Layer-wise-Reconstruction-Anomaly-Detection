#!/bin/bash
# ---------------------------------------------------------------------------
# OAR submission script for Grid'5000 — LRAD on CelebA (Wearing_Hat).
#
# Submit:
#   ./scripts/oar_run_celeba_hat.sh         # creates logs dir then oarsub -S
# or manually:
#   mkdir -p ~/lrad/outputs/celeba_hat/run/logs
#   oarsub -S ./scripts/oar_run_celeba_hat.sh
#
# Track:
#   oarstat -u $USER
#   tail -f outputs/celeba_hat/run/logs/oar.<jobid>.stdout
#   tail -f outputs/celeba_hat/run/logs/lrad.log
#
# Cancel:
#   oardel <jobid>
# ---------------------------------------------------------------------------

#OAR -n lrad-celeba-wearing-hat
#OAR -l host=1/gpu=1,walltime=06:00:00
#OAR -O outputs/celeba_hat/run/logs/oar.%jobid%.stdout
#OAR -E outputs/celeba_hat/run/logs/oar.%jobid%.stderr
# Pin to GPU clusters at Nancy (uncomment / adjust to your site):
##OAR -p "cluster='grele' OR cluster='grappe' OR cluster='gruss'"

set -euo pipefail

OUTPUT_DIR="outputs/celeba_hat/run"

# If the user invoked the script directly (not via oarsub), do the submit
# dance for them: create the log dir first (OAR opens stdout/stderr files
# *before* the script body runs, so the directory must exist), then submit.
if [[ -z "${OAR_JOB_ID:-}" ]]; then
    cd "$HOME/lrad"
    mkdir -p "$OUTPUT_DIR"/logs "$OUTPUT_DIR"/plots "$OUTPUT_DIR"/weights
    echo "Submitting via oarsub -S $0"
    exec oarsub -S "$0"
fi

cd "$HOME/lrad"
# Defensive: re-create the dirs in case the job was submitted before the
# guard above existed (older job versions).
mkdir -p "$OUTPUT_DIR"/logs "$OUTPUT_DIR"/plots "$OUTPUT_DIR"/weights

# --- Environment ---------------------------------------------------------
PYTHON="$HOME/.conda/envs/lrad/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: lrad Python not found at $PYTHON" >&2
    exit 1
fi

# Force unbuffered stdio so `tail -f` shows progress live.
export PYTHONUNBUFFERED=1
# Avoid silently swallowing CUDA OOM stack traces.
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
# A safe default for OAR nodes (12 cores typical); cap MKL/OMP threads so
# DataLoader workers don't oversubscribe the CPU.
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

# --- Diagnostics ---------------------------------------------------------
echo "=== OAR job ${OAR_JOB_ID} on $(hostname) at $(date) ==="
echo "PWD=$(pwd)"
echo "PYTHON=$PYTHON"
nvidia-smi || echo "nvidia-smi not available"
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

# --- Data ----------------------------------------------------------------
# CelebA (~1.4 GB) lives at $HOME/lrad/data/celeba/. Download it ONCE on the
# frontend (torchvision's Google Drive download often fails on compute nodes
# due to quota): see the README block at the bottom of oar_run_celeba.sh.
DATA_ROOT="$HOME/lrad/data"

# --- Run -----------------------------------------------------------------
# `-u` = unbuffered Python; redundant with PYTHONUNBUFFERED but cheap.
# Capture exit code so we can echo a final status before exit.
set +e
"$PYTHON" -u scripts/run_celeba.py \
    --config configs/celeba_wearing_hat.yaml \
    --override dataset.root="$DATA_ROOT" dataset.download=false dataset.num_workers=4
RC=$?
set -e

echo "=== job ${OAR_JOB_ID} finished at $(date) with exit code ${RC} ==="
exit "${RC}"
