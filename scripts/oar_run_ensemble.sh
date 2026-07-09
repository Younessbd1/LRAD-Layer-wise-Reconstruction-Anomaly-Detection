#!/bin/bash
# ---------------------------------------------------------------------------
# OAR submission script for Grid'5000 — CelebA OOD *deep-ensemble* run,
# pinned to the **graffiti** cluster (Nancy, production/Abaca queue —
# 12 nodes x 4 Nvidia RTX 2080 Ti, 11 GiB VRAM each).
#
# Why graffiti: it is the largest GPU pool at Nancy (44 GPUs), so queue
# time is usually minutes to a couple of hours — much faster than the
# single-node A100/L40S clusters (grat, gres) or the lone V100 node
# (gratouille, which no longer has A100s). The run needs < 10 GiB of
# VRAM (0.9–2.7M-parameter members, 64x64 images, batch 256), so the
# 2080 Ti's 11 GiB is enough. Fallbacks with the same script: change the
# cluster line to gruss (2 x A40 45 GiB, ~3x faster per GPU, 4 nodes) or
# grue (4 x Tesla T4 15 GiB, 5 nodes).
#
# This trains `ensemble.size` models sequentially in a single job (each is
# a full classifier + per-block ConvTranspose2d decoders), then runs the
# bias/variance decomposition + every plot — including the per-instance
# folders and top_ood_glasses.png — in that one run (there is no separate
# eval/plot job any more). The current config does 10 models, each with its
# OWN 5-block architecture (ensemble.member_variants: channel widths +
# kernel size) and a 20 classifier + 25 decoder epoch schedule per model —
# an architecturally diverse deep ensemble (no regularizer).
#
# Walltime: 48 h. A 2080 Ti is ~2–3x slower than the A100 this used to run
# on (~1.2 h/model there), so budget ~3 h/model → ~30 h for 10 models.
# Requesting 48 h makes OAR place the job on graffiti nodes 4–13 (the
# 24 h-max nodes 1–3 are excluded automatically). Never under-size: an OAR
# job is cut at its walltime even mid-epoch and this pipeline does not
# checkpoint the ensemble run.
#
# Production queue rules (Abaca): the job MUST carry `-q production`
# (set in the directives below) and advance reservations (`oarsub -r`)
# are NOT allowed — submission only; the job starts when a slot frees up.
#
# Submit (from the Nancy frontend, in ~/lrad):
#   ./scripts/oar_run_ensemble.sh
# or manually:
#   mkdir -p ~/lrad/outputs/celeba_ood/_oar
#   oarsub -S ./scripts/oar_run_ensemble.sh
#
# Track:
#   oarstat -u $USER
#   tail -f outputs/celeba_ood/_oar/oar.<jobid>.stdout
#   tail -f outputs/celeba_ood/ensemble_*_<jobid>/logs/celeba_ood_ensemble_*.log
#
# Cancel:
#   oardel <jobid>
# ---------------------------------------------------------------------------

#OAR -n celeba-ood-ensemble-graffiti
#OAR -q production
#OAR -p cluster='graffiti'
#OAR -l gpu=1,walltime=48:00:00
#OAR -O outputs/celeba_ood/_oar/oar.%jobid%.stdout
#OAR -E outputs/celeba_ood/_oar/oar.%jobid%.stderr

set -euo pipefail

# If invoked directly (not through oarsub), bootstrap and submit.
# (No advance-reservation path: `oarsub -r` is refused by the production
# queue — submission only.)
if [[ -z "${OAR_JOB_ID:-}" ]]; then
    cd "$HOME/lrad"
    mkdir -p outputs/celeba_ood/_oar
    echo "Submitting to the production queue via oarsub -S $0"
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
echo "=== OAR job ${OAR_JOB_ID} on $(hostname) [graffiti] at $(date) ==="
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
