#!/bin/bash
# ---------------------------------------------------------------------------
# OAR submission script for Grid'5000 — CutPaste hyperparameter GRID SEARCH
# (scripts/run_gridsearch.py), MULTI-METRIC: each of the 30 pruned configs
# (scar_prob x area_range x prob x loss_weight) trains a short classifier
# (6 epochs) AND its per-block decoders (8 epochs) at 64 px, then is scored
# on the whole stack — cutpaste head, reconstruction bias p95, locfre_b3,
# energy, and their rank fusion (the selection metric). A per-epoch
# cutpaste-AUROC curve is recorded to pick the epoch budget.
# ~18-20 min per config on a 2080 Ti → ~9-10 h; walltime 24 h is margin
# (still fits every graffiti node).
#
# Writes outputs/celeba_ood/gridsearch_<jobid>/:
#   gridsearch_results.json   every config + all metrics, sorted by fused
#   gridsearch_auroc.png      grouped bars, every metric per config
#   gridsearch_epochs.png     pretext AUROC vs classifier epochs
# Paste the logged best config into configs/celeba_ood_128.yaml
# (training.cutpaste) before submitting oar_run_128.sh.
#
# Submit (from the Nancy frontend, in ~/lrad):
#   ./scripts/oar_run_gridsearch.sh
# Track:   oarstat -u $USER ; tail -f outputs/celeba_ood/_oar/oar.<jobid>.stdout
# Cancel:  oardel <jobid>
# ---------------------------------------------------------------------------

#OAR -n celeba-ood-cutpaste-grid
#OAR -q production
#OAR -p cluster='graffiti'
#OAR -l gpu=1,walltime=24:00:00
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

PYTHON="$HOME/.conda/envs/lrad/bin/python3"
if [[ ! -x "$PYTHON" ]]; then
    echo "ERROR: lrad Python not found at $PYTHON" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-2}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-2}

echo "=== OAR job ${OAR_JOB_ID} on $(hostname) at $(date) ==="
nvidia-smi || echo "nvidia-smi not available"
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

OUTPUT_DIR="outputs/celeba_ood/gridsearch_${OAR_JOB_ID}"
mkdir -p "$OUTPUT_DIR"

set +e
"$PYTHON" -u scripts/run_gridsearch.py \
    --output-dir "$OUTPUT_DIR" \
    --epochs 6 \
    --num-workers 4
RC=$?
set -e

echo "=== job ${OAR_JOB_ID} finished at $(date) with exit code ${RC} ==="
exit "${RC}"
