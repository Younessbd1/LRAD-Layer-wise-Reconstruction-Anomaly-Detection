#!/bin/bash
# ---------------------------------------------------------------------------
# OAR submission script for Grid'5000 — CelebA OOD ablation study,
# ONE JOB PER ARM so the arms train in parallel on different nodes.
#
# The study isolates the two ingredients of the ensemble recipe against a
# common control (configs/ablation_baseline.yaml documents the design):
#
#   arm            ensemble members            pretext    config
#   baseline       10 x same architecture      none       ablation_baseline.yaml
#   arch           10 different architectures  none       ablation_arch.yaml
#   cutpaste       10 x same architecture      CutPaste   ablation_cutpaste.yaml
#   arch_cutpaste  10 different architectures  CutPaste   ablation_arch_cutpaste.yaml
#
# Case studies: (1) arch vs baseline, (2) cutpaste vs baseline,
# (3) arch_cutpaste vs baseline — scripts/compare_ablation.py builds the
# tables + figures once the arms are done (each job tries it on exit, so
# the LAST job to finish writes the comparison automatically).
#
# By default only the first THREE arms are submitted: the arch_cutpaste
# recipe was already run as outputs/celeba_ood/LASTOF_RESULTS (128 px,
# 10 diverse members, CutPaste winner, seeds 42..51) and the comparison
# script falls back to it. Submit the fourth arm explicitly only to
# re-verify that run under the current code:
#   ARMS=arch_cutpaste ./scripts/oar_run_ablation.sh
#
# CLUSTER CHOICE — any of gres / grat / gruss, whichever frees a GPU
# first. All three carry A40-class-or-faster cards, and a 128 px member is
# ~3-4 h on an A40 (~10-20 h on a grue T4), so 10 members fit the 48 h
# walltime on any of them:
#
#   gres   7 nodes x 2 L40S 48 GiB  (fastest, largest modern pool)
#   grat   1 node  x 8 A100 40 GiB
#   gruss  4 nodes x 2 A40  45 GiB  (only its >=48 h-walltime nodes match)
#
# Pinning gruss alone (the first submission, jobs 6866466-68) left the
# three jobs Waiting: its 8 A40s were fully busy and the 48 h request
# excludes its 24 h-max node. The IN (...) disjunction below triples the
# eligible pool; gres and grat allow 168 h on every node, so the 48 h
# request never shrinks them. Arms are fully independent — two arms
# landing on one node's two GPUs is harmless. Do NOT add grue (T4) or
# graffiti (2080 Ti) without shrinking the ensemble: 10 members overrun
# 48 h there and OAR kills the job mid-epoch with nothing checkpointed.
#
# Each job trains its arm's ensemble (run_ensemble.py), then chains the
# same post-evaluations as oar_run_128.sh: fused scoring with supervised
# calibration (fused_auroc.json + plots/fused_auroc.png) and localized
# pixel scoring (localized_auroc.json). Every arm run dir has the same
# structure as LASTOF_RESULTS: model_0..9/ + ensemble/ (plots, summary,
# fused + localized JSONs) + logs/.
#
# Submit (from the Nancy frontend, in ~/lrad):
#   ./scripts/oar_run_ablation.sh                 # baseline + arch + cutpaste
#   ARMS="baseline cutpaste" ./scripts/oar_run_ablation.sh
#   ARMS=all ./scripts/oar_run_ablation.sh        # all four arms
# Track:   oarstat -u $USER ; tail -f outputs/celeba_ood/_oar/oar.<jobid>.stdout
# Cancel:  oardel <jobid>
# Compare (after the arms finish, or rerun any time):
#   python scripts/compare_ablation.py
# ---------------------------------------------------------------------------

#OAR -n celeba-ood-ablation
#OAR -q production
#OAR -p cluster IN ('gres', 'grat', 'gruss')
#OAR -l gpu=1,walltime=48:00:00
#OAR -O outputs/celeba_ood/_oar/oar.%jobid%.stdout
#OAR -E outputs/celeba_ood/_oar/oar.%jobid%.stderr

set -euo pipefail

KNOWN_ARMS="baseline arch cutpaste arch_cutpaste"

# If invoked directly (not through oarsub), submit one job per arm.
# OAR runs the job in a fresh shell on the node — it does NOT export the
# submitting shell's environment — so the arm is baked into the submitted
# command line, where it comes back as $1.
if [[ -z "${OAR_JOB_ID:-}" ]]; then
    cd "$HOME/lrad"
    mkdir -p outputs/celeba_ood/_oar
    ARMS="${ARMS:-baseline arch cutpaste}"
    [[ "$ARMS" == "all" ]] && ARMS="$KNOWN_ARMS"
    for ARM in $ARMS; do
        if [[ " $KNOWN_ARMS " != *" $ARM "* ]]; then
            echo "ERROR: unknown arm '$ARM' (known: $KNOWN_ARMS)" >&2
            exit 1
        fi
        if [[ ! -f "configs/ablation_${ARM}.yaml" ]]; then
            echo "ERROR: configs/ablation_${ARM}.yaml not found" >&2
            exit 1
        fi
        echo "Submitting arm '$ARM' via oarsub -S $0 $ARM"
        oarsub -S "$0 $ARM"
    done
    exit 0
fi

cd "$HOME/lrad"

# On the node the arm arrives as the positional arg baked in at submission.
ARM="${1:?missing arm argument — submit through ./scripts/oar_run_ablation.sh}"
if [[ " $KNOWN_ARMS " != *" $ARM "* ]]; then
    echo "ERROR: unknown arm '$ARM' (known: $KNOWN_ARMS)" >&2
    exit 1
fi

RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/celeba_ood/ablation/${ARM}_${RUN_TS}_${OAR_JOB_ID}"
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

echo "=== OAR job ${OAR_JOB_ID} (arm: $ARM) on $(hostname) at $(date) ==="
nvidia-smi || echo "nvidia-smi not available"
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

DATA_ROOT="$HOME/lrad/data"

set +e
"$PYTHON" -u scripts/run_ensemble.py \
    --config "configs/ablation_${ARM}.yaml" \
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
fi

# Opportunistic comparison: cheap (reads JSONs only) and idempotent. The
# arms finish at different times; whichever job ends last finds every arm
# present and leaves the final tables + figures in
# outputs/celeba_ood/ablation/comparison/. Never fails the job.
if [[ $RC -eq 0 ]]; then
    echo "--- ablation comparison (best effort) ---"
    "$PYTHON" -u scripts/compare_ablation.py || \
        echo "comparison skipped (some arms not finished yet)"
else
    echo "arm '$ARM' failed (rc=$RC) — skipping post-evaluations/comparison"
fi

echo "=== job ${OAR_JOB_ID} (arm: $ARM) finished at $(date) with exit code ${RC} ==="
exit "${RC}"
