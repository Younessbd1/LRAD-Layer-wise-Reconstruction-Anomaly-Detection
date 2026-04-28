#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Shared environment bootstrap for every LRAD OAR job on Grid'5000 Nancy.
# Sourced at the top of each job script, never executed on its own.
# ---------------------------------------------------------------------------
set -euo pipefail

export LRAD_ROOT="${LRAD_ROOT:-$HOME/lrad}"
export LRAD_VENV="${LRAD_VENV:-$HOME/lrad-venv}"
export LRAD_DATA_ROOT="${LRAD_DATA_ROOT:-$HOME/Real-IAD/realiad_1024}"
export LRAD_MANIFEST="${LRAD_MANIFEST:-$LRAD_ROOT/outputs/realiad/manifest/manifest.csv}"

# Keep torch model cache off the quota-limited /home on Grid'5000.
export TORCH_HOME="${TORCH_HOME:-/tmp/$USER/torch_cache}"
mkdir -p "$TORCH_HOME"

# --- CUDA -------------------------------------------------------------------
# Grid'5000 Nancy exposes CUDA via environment-modules.  Try the most common
# names; the loop stops on the first match so the order matters.
if command -v module >/dev/null 2>&1; then
    module purge 2>/dev/null || true
    for cand in cuda/12.4 cuda/12.1 cuda/11.8 cuda; do
        if module avail "$cand" 2>&1 | grep -q "$cand"; then
            module load "$cand"
            break
        fi
    done
fi

# --- Python env -------------------------------------------------------------
# Prefer the conda lrad env's Python (3.10) over the system Python (3.9),
# because lrad requires Python >=3.10.
PYTHON_BIN="${LRAD_PYTHON:-${HOME}/.conda/envs/lrad/bin/python3.10}"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

# PyTorch wheels for Pascal GPUs (GTX 1080 Ti, sm_61) live on the cu118 index;
# pip's default index would resolve to a newer build that has dropped Pascal.
PYTORCH_INDEX="https://download.pytorch.org/whl/cu118"

if [ ! -d "$LRAD_VENV" ]; then
    echo "[env] creating venv at $LRAD_VENV (using $PYTHON_BIN)"
    "$PYTHON_BIN" -m venv "$LRAD_VENV"
    source "$LRAD_VENV/bin/activate"
    pip install --upgrade pip wheel
    pip install -r "$LRAD_ROOT/requirements.txt" --extra-index-url "$PYTORCH_INDEX"
    pip install -e "$LRAD_ROOT"
else
    source "$LRAD_VENV/bin/activate"
    # Reinstall if the editable install is missing (e.g. venv was created with the wrong Python).
    python -c "import lrad" 2>/dev/null || pip install -e "$LRAD_ROOT"
fi

# Belt-and-suspenders: ensure lrad is importable even if venv activation
# doesn't propagate cleanly across OAR's batch environment.
export PYTHONPATH="$LRAD_ROOT${PYTHONPATH:+:$PYTHONPATH}"

# --- Reproducibility --------------------------------------------------------
export PYTHONHASHSEED=0
export CUBLAS_WORKSPACE_CONFIG=":4096:8"   # deterministic cuBLAS

echo "[env] python  : $(python -V)"
echo "[env] pip     : $(which pip)"
echo "[env] cuda    : $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no GPU visible')"
echo "[env] root    : $LRAD_ROOT"
echo "[env] data    : $LRAD_DATA_ROOT"
