#!/bin/bash
# ---------------------------------------------------------------------------
# OAR submission script for Grid'5000 — CutPaste + 128 px ensemble run
# (configs/celeba_ood_cutpaste128.yaml: 10 members, cutpaste pretext head,
# image_size 128). Pinned to **gruss** (Nancy production — 4 nodes x
# 2 A40 45 GiB): an A40 is ~3x a 2080 Ti, and 128 px costs ~4x per image,
# so budget ~3-4 h/member → ~30-40 h for 10 members; walltime 48 h.
# graffiti would NOT fit this run inside 48 h — do not retarget it there
# without also shrinking the ensemble.
#
# BEFORE submitting: run scripts/oar_run_gridsearch.sh and paste its best
# config into training.cutpaste of configs/celeba_ood_cutpaste128.yaml
# (the file ships with the grid-search default values).
#
# Trains the ensemble, then chains the same post-evaluations as
# oar_run_ensemble.sh: fused scoring with supervised calibration
# (fused_auroc.json + plots/fused_auroc.png) and localized pixel scoring
# (localized_auroc.json). The fused recipe automatically includes the new
# cutpaste_prob signal because the members carry the pretext head.
#
# Submit (from the Nancy frontend, in ~/lrad):
#   ./scripts/oar_run_cutpaste128.sh
# Track:   oarstat -u $USER ; tail -f outputs/celeba_ood/_oar/oar.<jobid>.stdout
# Cancel:  oardel <jobid>
# ---------------------------------------------------------------------------

#OAR -n celeba-ood-cutpaste128-gruss
#OAR -q production
#OAR -p cluster='gruss'
#OAR -l gpu=1,walltime=48:00:00
#OAR -O outputs/celeba_ood/_oar/oar.%jobid%.stdout
#OAR -E outputs/celeba_ood/_oar/oar.%jobid%.stderr

set -euo pipefail

if [[ -z "${OAR_JOB_ID:-}" ]]; then
    cd "$HOME/lrad"
    mkdir -p outputs/celeba_ood/_oar
    echo "Submitting to the production queue via oarsub -S $0"
    exec oarsub -S "$0"
fi

cd "$HOME/lrad"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/celeba_ood/cp128_${RUN_TS}_${OAR_JOB_ID}"
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

set +e
"$PYTHON" -u scripts/run_ensemble.py \
    --config configs/celeba_ood_cutpaste128.yaml \
    --output-dir "$OUTPUT_DIR" \
    --override dataset.root="$DATA_ROOT" dataset.download=false dataset.num_workers=4
RC=$?
set -e

if [[ $RC -eq 0 ]]; then
    echo "--- fused scoring (lrad.fusion, supervised calibration) ---"
    set +e
    "$PYTHON" -u scripts/run_fused.py \
        --output-dir "$OUTPUT_DIR" \
        --supervised --blocks 1 2 3 \
        --num-workers 4
    RC_FUSED=$?
    echo "--- localized pixel scoring (lrad.localized) ---"
    "$PYTHON" -u scripts/run_localized.py \
        --output-dir "$OUTPUT_DIR" \
        --num-workers 4
    RC_LOC=$?
    set -e
    [[ $RC_FUSED -ne 0 ]] && RC=$RC_FUSED
    [[ $RC -eq 0 && $RC_LOC -ne 0 ]] && RC=$RC_LOC
else
    echo "training failed (rc=$RC) — skipping the post-training evaluations"
fi

echo "=== job ${OAR_JOB_ID} finished at $(date) with exit code ${RC} ==="
exit "${RC}"
