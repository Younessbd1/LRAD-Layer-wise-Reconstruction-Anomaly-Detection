#!/usr/bin/env bash
#OAR -n lrad_prep
#OAR -O /home/ybahaddo/lrad/logs/oar/prepare_%jobid%.out
#OAR -E /home/ybahaddo/lrad/logs/oar/prepare_%jobid%.err
#OAR -l /nodes=1,walltime=00:30:00
# ---------------------------------------------------------------------------
# One-shot pre-processing job — CPU only, no GPU reservation.
#
# Reads the Real-IAD zip archives from $LRAD_DATA_ROOT and writes:
#   outputs/realiad/manifest/manifest.csv    full sample index
#   outputs/realiad/manifest/metadata.json
#   outputs/realiad/manifest/warnings.txt
#   outputs/realiad/verify/*                 integrity report
#   outputs/realiad/analysis/*               dataset figures
#
# Submit from the repo root:
#   oarsub -S scripts/oar/prepare_realiad.sh
# ---------------------------------------------------------------------------
set -euo pipefail

cd "$OAR_WORKING_DIRECTORY"
mkdir -p logs/oar

source scripts/oar/env_setup.sh

echo "[job] START prepare  $(date -Is)"

python scripts/prepare_realiad.py \
    --data-root "$LRAD_DATA_ROOT" \
    --out-dir   outputs/realiad/manifest

python scripts/verify_realiad.py \
    --manifest outputs/realiad/manifest/manifest.csv \
    --out-dir  outputs/realiad/verify \
    --sample-frac 0.1

python scripts/analyze_realiad.py \
    --manifest outputs/realiad/manifest/manifest.csv \
    --out-dir  outputs/realiad/analysis

echo "[job] DONE prepare   $(date -Is)"
