#!/bin/bash
# ---------------------------------------------------------------------------
# OAR submission script for Grid'5000 — CelebA OOD experiment.
#
# Submit:
#   ./scripts/oar_run_celeba.sh
# or manually:
#   mkdir -p ~/lrad/outputs/celeba_ood/run/logs
#   oarsub -S ./scripts/oar_run_celeba.sh
#
# Track:
#   oarstat -u $USER
#   tail -f outputs/celeba_ood/run/logs/oar.<jobid>.stdout
#   tail -f outputs/celeba_ood/run/logs/celeba_ood_*.log
#
# Cancel:
#   oardel <jobid>
# ---------------------------------------------------------------------------

#OAR -n celeba-ood
#OAR -l host=1/gpu=1,walltime=04:00:00
#OAR -O outputs/celeba_ood/run/logs/oar.%jobid%.stdout
#OAR -E outputs/celeba_ood/run/logs/oar.%jobid%.stderr

set -euo pipefail

OUTPUT_DIR="outputs/celeba_ood/run"

# If invoked directly (not through oarsub), bootstrap the log dir and submit.
if [[ -z "${OAR_JOB_ID:-}" ]]; then
    cd "$HOME/lrad"
    mkdir -p "$OUTPUT_DIR"/logs "$OUTPUT_DIR"/plots "$OUTPUT_DIR"/weights
    echo "Submitting via oarsub -S $0"
    exec oarsub -S "$0"
fi

cd "$HOME/lrad"
mkdir -p "$OUTPUT_DIR"/logs "$OUTPUT_DIR"/plots "$OUTPUT_DIR"/weights

# --- Environment ---------------------------------------------------------
PYTHON="$HOME/.conda/envs/lrad/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: lrad Python not found at $PYTHON" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

# --- Diagnostics ---------------------------------------------------------
echo "=== OAR job ${OAR_JOB_ID} on $(hostname) at $(date) ==="
echo "PWD=$(pwd)"
echo "PYTHON=$PYTHON"
nvidia-smi || echo "nvidia-smi not available"
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

# --- Run -----------------------------------------------------------------
DATA_ROOT="$HOME/lrad/data"

set +e
"$PYTHON" -u scripts/run_celeba.py \
    --config configs/celeba_ood.yaml \
    --override dataset.root="$DATA_ROOT" dataset.download=false dataset.num_workers=4
RC=$?
set -e

echo "=== job ${OAR_JOB_ID} finished at $(date) with exit code ${RC} ==="
exit "${RC}"
