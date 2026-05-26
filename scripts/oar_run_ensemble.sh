#!/bin/bash
# ---------------------------------------------------------------------------
# OAR submission script for Grid'5000 — CelebA OOD *deep-ensemble* run,
# pinned to the **gratouille** cluster (Nancy — A100 GPUs).
#
# This trains `ensemble.size` models sequentially in a single job (each is
# a full classifier + decoders), then runs the bias/variance decomposition.
# A single CelebA run is ~1h, so a 5-model ensemble is ~5-6h; the 10h
# reservation below is deliberately generous but never under-size it — a
# reservation is cut at its end even mid-epoch and training does not
# checkpoint.
#
# We run as an ADVANCE RESERVATION (oarsub -r), NOT besteffort: a
# reservation books the GPU for a fixed future window and is guaranteed —
# it is never preempted by other jobs.
#
# Submit (advance reservation — give a future start time):
#   ./scripts/oar_run_ensemble.sh '2026-05-23 20:00:00'
# or manually:
#   mkdir -p ~/lrad/outputs/celeba_ood/_oar
#   oarsub -r '2026-05-23 20:00:00' -S ./scripts/oar_run_ensemble.sh
#
# Submit immediately (normal queued job, no reservation):
#   ./scripts/oar_run_ensemble.sh
#
# Track:
#   oarstat -u $USER
#   tail -f outputs/celeba_ood/_oar/oar.<jobid>.stdout
#   tail -f outputs/celeba_ood/ensemble_*_<jobid>/logs/celeba_ood_ensemble_*.log
#
# Cancel:
#   oardel <jobid>
# ---------------------------------------------------------------------------

#OAR -n celeba-ood-ensemble-gratouille
#OAR -p cluster='gratouille'
#OAR -l gpu=1,walltime=10:00:00
#OAR -O outputs/celeba_ood/_oar/oar.%jobid%.stdout
#OAR -E outputs/celeba_ood/_oar/oar.%jobid%.stderr

set -euo pipefail

# If invoked directly (not through oarsub), bootstrap and submit.
# Optional arg 1 = advance-reservation start time, e.g. '2026-05-23 20:00:00'.
if [[ -z "${OAR_JOB_ID:-}" ]]; then
    cd "$HOME/lrad"
    mkdir -p outputs/celeba_ood/_oar
    if [[ -n "${1:-}" ]]; then
        echo "Submitting advance reservation for '$1' via oarsub -r"
        exec oarsub -r "$1" -S "$0"
    fi
    echo "Submitting (immediate, normal queue) via oarsub -S $0"
    exec oarsub -S "$0"
fi

cd "$HOME/lrad"

# Each OAR ensemble run gets its own folder under outputs/celeba_ood/.
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/celeba_ood/ensemble_${RUN_TS}_${OAR_JOB_ID}"
mkdir -p "$OUTPUT_DIR"/logs
echo "OUTPUT_DIR=$OUTPUT_DIR"

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
echo "=== OAR job ${OAR_JOB_ID} on $(hostname) [gratouille] at $(date) ==="
echo "PWD=$(pwd)"
echo "PYTHON=$PYTHON"
nvidia-smi || echo "nvidia-smi not available"
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

# --- Run -----------------------------------------------------------------
DATA_ROOT="$HOME/lrad/data"

set +e
"$PYTHON" -u scripts/run_ensemble.py \
    --config configs/celeba_ood.yaml \
    --output-dir "$OUTPUT_DIR" \
    --override dataset.root="$DATA_ROOT" dataset.download=false dataset.num_workers=4
RC=$?
set -e

echo "=== job ${OAR_JOB_ID} finished at $(date) with exit code ${RC} ==="
exit "${RC}"
