#!/usr/bin/env bash
#OAR -n lrad_ablation
#OAR -O /home/ybahaddo/lrad/logs/oar/ablation_%jobid%.out
#OAR -E /home/ybahaddo/lrad/logs/oar/ablation_%jobid%.err
#OAR -l /nodes=1,walltime=24:00:00
#OAR -p gpu_count>=1
# ---------------------------------------------------------------------------
# Depth-ablation sweep: resnet10 → resnet18 → resnet34 on one GPU, serially.
# Walltime is generous (24h) because resnet34 on a large category set is slow.
#
# Submit from the repo root:
#   oarsub -S scripts/oar/ablation_depth.sh
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$OAR_WORKING_DIRECTORY"
mkdir -p logs/oar

source scripts/oar/env_setup.sh

LRAD_CONFIG="${LRAD_CONFIG:-configs/realiad.yaml}"

python scripts/ablation_depth.py \
    --config  "$LRAD_CONFIG" \
    --presets resnet10 resnet18 resnet34 \
    --out-dir outputs/realiad/ablation_depth \
    "$@"
