#!/bin/bash
# ---------------------------------------------------------------------------
# OAR submission script for Grid'5000 — CelebA OOD deep-ensemble run with the
# ConvTranspose2d decoder variant (training.decoders.upsample=transpose).
#
# Same job as oar_run_ensemble.sh in every respect EXCEPT the decoder
# upsampling stage: here each per-block decoder grows its activation back to
# image size with learnable ConvTranspose2d(4x4, stride 2) instead of the
# default bilinear resize-then-conv. Same trunk, same schedule, same plots —
# so the per-instance figures, bias/variance maps and AUROCs come out directly
# comparable to the bilinear run. Outputs go to their own
# ensemble_transpose_* folder so the two variants never overwrite each other.
#
# Pinned to the gratouille cluster (Nancy — A100). An OAR reservation is cut
# at its walltime even mid-epoch and the ensemble run does not checkpoint, so
# the walltime is sized generously — never under-size.
#
# Submit immediately (normal queued job):
#   ./scripts/oar_run_ensemble_transpose.sh
#
# Submit as an advance reservation (guaranteed future window, never preempted):
#   ./scripts/oar_run_ensemble_transpose.sh '2026-06-28 20:00:00'
#
# Track:
#   oarstat -u $USER
#   tail -f outputs/celeba_ood/_oar/oar.<jobid>.stdout
#   tail -f outputs/celeba_ood/ensemble_transpose_*_<jobid>/logs/celeba_ood_ensemble_*.log
#
# Cancel:
#   oardel <jobid>
# ---------------------------------------------------------------------------

#OAR -n celeba-ood-ensemble-transpose-gratouille
#OAR -p cluster='gratouille'
#OAR -l gpu=1,walltime=24:00:00
#OAR -O outputs/celeba_ood/_oar/oar.%jobid%.stdout
#OAR -E outputs/celeba_ood/_oar/oar.%jobid%.stderr

set -euo pipefail

# If invoked directly (not through oarsub), bootstrap and submit.
# Optional arg 1 = advance-reservation start time, e.g. '2026-06-28 20:00:00'.
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

# Variant-specific output folder so transpose and bilinear runs stay separate.
RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/celeba_ood/ensemble_transpose_${RUN_TS}_${OAR_JOB_ID}"
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
# The only functional difference from oar_run_ensemble.sh is the
# training.decoders.upsample=transpose override on the last line.
DATA_ROOT="$HOME/lrad/data"

set +e
"$PYTHON" -u scripts/run_ensemble.py \
    --config configs/celeba_ood.yaml \
    --output-dir "$OUTPUT_DIR" \
    --override dataset.root="$DATA_ROOT" dataset.download=false dataset.num_workers=4 \
               training.decoders.upsample=transpose
RC=$?
set -e

echo "=== job ${OAR_JOB_ID} finished at $(date) with exit code ${RC} ==="
exit "${RC}"
