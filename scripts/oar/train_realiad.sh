#!/usr/bin/env bash
#OAR -n lrad_train
#OAR -O /home/ybahaddo/lrad/logs/oar/train_%jobid%.out
#OAR -E /home/ybahaddo/lrad/logs/oar/train_%jobid%.err
#OAR -l /nodes=1,walltime=12:00:00
#OAR -p gpu_count>=1
# ---------------------------------------------------------------------------
# Train + evaluate LRAD on Real-IAD — single GPU, default config.
#
# grimoire = Grid'5000 Nancy A100 80 GB cluster (highest available GPU).
# Remove the -p line if grimoire is saturated and a V100/P100 is acceptable.
#
# Submit from the repo root:
#   oarsub -S scripts/oar/train_realiad.sh
#
# Override config at submit time via the LRAD_CONFIG env variable:
#   LRAD_CONFIG=configs/realiad.yaml oarsub -S scripts/oar/train_realiad.sh
#
# Pass run_realiad.py overrides after the script path:
#   oarsub -S "scripts/oar/train_realiad.sh
#     --override model.preset=resnet34 dataset.categories=null"
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$OAR_WORKING_DIRECTORY"
mkdir -p logs/oar

source scripts/oar/env_setup.sh

LRAD_CONFIG="${LRAD_CONFIG:-configs/realiad.yaml}"

echo "[job] node    : $(hostname)   oar_id=$OAR_JOB_ID"
echo "[job] config  : $LRAD_CONFIG"
echo "[job] started : $(date -Is)"
echo "[job] GPU(s)  :"
nvidia-smi --query-gpu=index,name,memory.total,driver_version \
           --format=csv,noheader || true

python scripts/run_realiad.py --config "$LRAD_CONFIG" "$@"

echo "[job] finished: $(date -Is)"
