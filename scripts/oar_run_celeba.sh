#!/bin/bash
# ---------------------------------------------------------------------------
# OAR submission script for Grid'5000 — LRAD on CelebA (Eyeglasses).
#
# Submit:
#   oarsub -S ./scripts/oar_run_celeba.sh
#
# Track:
#   oarstat -u $USER
#   tail -f outputs/celeba/run/oar.<jobid>.stdout
#
# Cancel:
#   oardel <jobid>
# ---------------------------------------------------------------------------

#OAR -n lrad-celeba-eyeglasses
#OAR -l host=1/gpu=1,walltime=06:00:00
#OAR -O outputs/celeba/run/oar.%jobid%.stdout
#OAR -E outputs/celeba/run/oar.%jobid%.stderr
# Pin to GPU clusters at Nancy (uncomment / adjust to your site):
##OAR -p "cluster='grele' OR cluster='grappe' OR cluster='gruss'"

set -euo pipefail

cd "$HOME/lrad"
mkdir -p outputs/celeba/run/{logs,plots,weights}

# --- Environment ---------------------------------------------------------
# Grid'5000 compute nodes do not pre-load conda; source the user install.
if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/.conda/etc/profile.d/conda.sh" ]]; then
    source "$HOME/.conda/etc/profile.d/conda.sh"
elif [[ -f "/grid5000/spack/share/spack/setup-env.sh" ]]; then
    source /grid5000/spack/share/spack/setup-env.sh
fi
conda activate lrad

# --- Diagnostics ---------------------------------------------------------
echo "=== OAR job ${OAR_JOB_ID:-N/A} on $(hostname) at $(date) ==="
nvidia-smi || echo "nvidia-smi not available"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'devices', torch.cuda.device_count())"

# --- Data ----------------------------------------------------------------
# CelebA (~1.4 GB) lives at $HOME/lrad/data/celeba/. Download it ONCE on the
# frontend (torchvision's Google Drive download often fails on compute nodes
# due to quota): see the README block at the bottom of this file.
DATA_ROOT="$HOME/lrad/data"

# --- Run -----------------------------------------------------------------
python scripts/run_celeba.py \
    --config configs/celeba_eyeglasses.yaml \
    --override dataset.root="$DATA_ROOT" dataset.download=false dataset.num_workers=4

echo "=== job finished at $(date) ==="

# ---------------------------------------------------------------------------
# One-time CelebA download on the frontend (run before submitting):
#
#   conda activate lrad
#   mkdir -p $HOME/lrad/data
#   python -c "
#   from torchvision.datasets import CelebA
#   CelebA(root='$HOME/lrad/data', split='all', target_type='attr', download=True)
#   "
#
# If the Google Drive download fails (common), fetch the archive manually
# from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html and unpack into
# $HOME/lrad/data/celeba/  with the layout expected by torchvision:
#   data/celeba/img_align_celeba/        (202,599 .jpg files)
#   data/celeba/list_attr_celeba.txt
#   data/celeba/list_eval_partition.txt
#   data/celeba/identity_CelebA.txt
#   data/celeba/list_bbox_celeba.txt
#   data/celeba/list_landmarks_align_celeba.txt
# ---------------------------------------------------------------------------
