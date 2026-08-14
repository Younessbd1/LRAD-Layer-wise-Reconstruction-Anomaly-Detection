#!/bin/bash
# ---------------------------------------------------------------------------
# OAR submission script for Grid'5000 — MVTec AD per-category LRAD ensembles
# (configs/mvtec.yaml), for the PatchCore comparison.
#
# COST. MVTec categories are tiny (60-391 training images against ~180k
# CelebA faces), which makes an EPOCH cheap but is exactly why the schedule
# is set in optimizer STEPS instead (configs/mvtec.yaml, training.steps /
# training.decoders.steps) — a fixed epoch count would train toothbrush
# 6.5x less than hazelnut and leave both far short of convergence.
#
# The budget is 4,000 classifier + 8,000 decoder steps per member, unchanged
# from the earlier 4-member version of the config. The ensemble is now the
# 10-member set from docs/diagrams/ensemble_diversity_cubes.pdf, but it runs
# at 128 px on a 5-block trunk instead of 224 px on a 6-block one, and 128^2
# is ~0.33x the pixels — so 10 members cost about what 3.3 of the old ones
# did and the totals below barely move. On an A40, ~10-15 min per member:
#
#     5-category pilot x 10 members  ~=  9-13 h  on an A40
#     all 15 categories x 10 members ~=  25-38 h on an A40
#
# CLUSTER CHOICE: grue (Tesla T4, 15 GB), not gruss (A40, 46 GB).
#
# gruss is the faster card but we are queue p2 there, and its 8 A40s are held
# by p1 users who always preempt us — a job sat Waiting for over a day with a
# predicted start another day out, and shortening the walltime did not move
# that prediction because the cluster is booked solid, not merely fragmented.
# grue has free GPUs now, so the job starts immediately.
#
# The T4 is the right fallback for two specific reasons:
#
#   1. VRAM is a non-issue. The largest ensemble member is 2.7 M params
#      (M6/M10 in configs/mvtec.yaml) at 128 px, batch 32, fp32, and members
#      train one at a time — peak well under 4 GB. The T4's 15 GB is ample;
#      even the 11 GB cards would have fit.
#   2. The installed torch (2.5.1+cu121) is compiled for sm_50/60/70/75/80/
#      86/90. The T4 is sm_75, explicitly present. The GTX 1080 Ti on grele
#      is sm_61, which is NOT in that list — it was the faster free card on
#      paper and was rejected for this reason. Re-check the arch list before
#      moving to grele:
#        strings .../torch/lib/libtorch_cuda.so | grep -oE 'sm_[0-9]+' | sort -u
#
# Training is plain fp32 — no autocast, bf16, tf32 or torch.compile anywhere
# in lrad/ — so the A40's tensor cores were never in use and the honest T4-vs-
# A40 gap is the fp32 one (~8.1 vs ~37.4 TFLOPS peak), not the tensor-core
# one. These models are small enough to be far from saturating either card,
# so the real slowdown should land well under that 4.6x ceiling.
#
# MEASURED ON A T4 (job 6848964, bottle, 2026-08-13), not estimated:
#
#     classifier  572 epochs           = 1186 s
#     decoders   1143 epochs @ 2.16 s  = 2469 s
#     -> 61 min per member, 10.2 h per 10-member category
#
# So the 5-category pilot is ~51 h SEQUENTIALLY, which overran even a 48 h
# walltime. Run it in PARALLEL instead — categories are fully independent
# (a fresh ensemble per category, no shared state), so one job per category
# finishes the whole pilot in ~10 h of wall-clock rather than ~51 h:
#
#     for c in bottle carpet screw transistor hazelnut; do
#         CATEGORIES=$c ./scripts/oar_run_mvtec.sh
#     done
#
# Walltime is 24 h: 2.4x headroom over the measured 10.2 h, and the >=24 h
# tier had 11 free GPUs against only 3 in the >=48 h tier, so every job
# starts immediately. Do not raise it to 48 h to run categories
# sequentially — that trades a 10 h wall-clock for a 51 h one AND competes
# for a much scarcer tier.
#
# Each job writes its own run dir, so merge the per-category results with:
#     python scripts/merge_mvtec_results.py outputs/mvtec/run_*_<jobids>

#
# These are estimates, not measurements. RUN THE PILOT FIRST: the log prints
# a per-member wall time, which turns the numbers above into real ones before
# committing to the full sweep. results.md is rewritten after every category,
# so a job killed at the walltime still leaves a complete table for whatever
# finished — nothing is lost by starting and re-submitting the remainder.
#
# For all 15 categories, raise the walltime to 48:00:00 below, or split the
# sweep across two jobs with CATEGORIES=... (see below).
#
# BEFORE submitting: fetch the dataset once on the frontend (it needs
# network access the compute nodes may not have):
#   python scripts/download_mvtec.py --root ~/lrad/data
#
# Submit (from the Nancy frontend, in ~/lrad):
#   ./scripts/oar_run_mvtec.sh              # pilot categories from the config
#   CATEGORIES=all ./scripts/oar_run_mvtec.sh
#   CATEGORIES="bottle cable capsule carpet grid hazelnut leather metal_nut" \
#       ./scripts/oar_run_mvtec.sh          # first half of the full sweep
# Track:   oarstat -u $USER ; tail -f outputs/mvtec/_oar/oar.<jobid>.stdout
# Cancel:  oardel <jobid>
# ---------------------------------------------------------------------------

#OAR -n mvtec-lrad-grue
#OAR -q production
#OAR -p cluster='grue'
#OAR -l gpu=1,walltime=24:00:00
#OAR -O outputs/mvtec/_oar/oar.%jobid%.stdout
#OAR -E outputs/mvtec/_oar/oar.%jobid%.stderr

set -euo pipefail

if [[ -z "${OAR_JOB_ID:-}" ]]; then
    cd "$HOME/lrad"
    mkdir -p outputs/mvtec/_oar
    # OAR runs the job in a fresh shell on the node — it does NOT export the
    # submitting shell's environment — so CATEGORIES=... has to be baked into
    # the submitted command line, where it comes back as positional args.
    SUBMIT="$0"
    [[ -n "${CATEGORIES:-}" ]] && SUBMIT="$0 $CATEGORIES"
    echo "Submitting to the production queue via oarsub -S $SUBMIT"
    exec oarsub -S "$SUBMIT"
fi

cd "$HOME/lrad"

RUN_TS="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR="outputs/mvtec/run_${RUN_TS}_${OAR_JOB_ID}"
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

echo "=== OAR job ${OAR_JOB_ID} on $(hostname) at $(date) ==="
nvidia-smi || echo "nvidia-smi not available"
"$PYTHON" -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

DATA_ROOT="$HOME/lrad/data"

# Fail early and loudly if the dataset was never fetched — otherwise every
# category raises the same FileNotFoundError one after another and the real
# cause is buried in the middle of the log.
if ! "$PYTHON" -u scripts/download_mvtec.py --root "$DATA_ROOT" --verify-only; then
    echo "ERROR: MVTec AD is missing or incomplete under $DATA_ROOT." >&2
    echo "Run this on the FRONTEND first:" >&2
    echo "  python scripts/download_mvtec.py --root $DATA_ROOT" >&2
    exit 1
fi

# On the node the categories arrive as positional args (see the submission
# block above); CATEGORIES=... still works when the script is run directly.
CATEGORIES="${CATEGORIES:-$*}"
CAT_ARGS=()
[[ -n "$CATEGORIES" ]] && CAT_ARGS=(--categories $CATEGORIES)

set +e
"$PYTHON" -u scripts/run_mvtec.py \
    --config configs/mvtec.yaml \
    --output-dir "$OUTPUT_DIR" \
    "${CAT_ARGS[@]}" \
    --override dataset.root="$DATA_ROOT" dataset.num_workers=4
RC=$?
set -e

if [[ $RC -eq 0 ]]; then
    echo "--- results table ---"
    cat "$OUTPUT_DIR/results.md" || true
fi

echo "=== job ${OAR_JOB_ID} finished at $(date) with exit code ${RC} ==="
exit "${RC}"
