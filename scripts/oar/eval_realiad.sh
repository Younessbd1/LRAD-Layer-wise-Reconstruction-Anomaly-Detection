#!/usr/bin/env bash
#OAR -n lrad_eval
#OAR -O /home/ybahaddo/lrad/logs/oar/eval_%jobid%.out
#OAR -E /home/ybahaddo/lrad/logs/oar/eval_%jobid%.err
#OAR -l /nodes=1,walltime=01:00:00
#OAR -p gpu_count>=1
# ---------------------------------------------------------------------------
# Re-evaluate a previously trained run (skip Phase 1 + 2).
# Loads weights from <output_dir>/weights/ and regenerates all metrics/plots.
#
# Submit from the repo root:
#   oarsub -S scripts/oar/eval_realiad.sh
#
# To override the output directory:
#   LRAD_OUTPUT_DIR=outputs/realiad/my_run \
#       oarsub -S scripts/oar/eval_realiad.sh
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$OAR_WORKING_DIRECTORY"
mkdir -p logs/oar

source scripts/oar/env_setup.sh

LRAD_CONFIG="${LRAD_CONFIG:-configs/realiad.yaml}"
ARGS=(--config "$LRAD_CONFIG" --eval-only)
if [ -n "${LRAD_OUTPUT_DIR:-}" ]; then
    ARGS+=(--output-dir "$LRAD_OUTPUT_DIR")
fi

python scripts/run_realiad.py "${ARGS[@]}" "$@"
