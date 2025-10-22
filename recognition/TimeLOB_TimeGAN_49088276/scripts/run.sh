#!/bin/bash

# Train TimeGAN on UQ Rangpur and run sampling + visualisation

#SBATCH --job-name=timegan-amzn
#SBATCH --partition=a100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# Conda bootstrap (batch-safe)
if [[ -z "${CONDA_EXE:-}" ]]; then
  # adjust this path to your cluster's Anaconda/Miniconda install if needed
  source "$HOME/miniconda3/etc/profile.d/conda.sh" 2>/dev/null || {
    echo "Conda not found. Please load your conda module or fix the path." >&2
    exit 1
  }
fi

# Create/update env only if absent (avoid costly rebuilds on every run)
ENV_NAME="timegan"
if ! conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  conda env create -n "${ENV_NAME}" -f environment.yml
else
  conda env update -n "${ENV_NAME}" -f environment.yml --prune
fi
conda activate "${ENV_NAME}"

# Project paths
export PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
export PYTHONPATH="$PROJECT_ROOT"

echo "[info] PROJECT_ROOT=$PROJECT_ROOT"
echo "[info] PYTHONPATH=$PYTHONPATH"

# Training
python -m src.train \
  --dataset \
    --seq-len 128 \
    --data-dir ./data \
    --orderbook-filename AMZN_2012-06-21_34200000_57600000_orderbook_10.csv \
    --splits 0.7 0.85 1.0 \
    --no-shuffle \
  --modules \
    --batch-size 128 \
    --z-dim 40 \
    --hidden-dim 64 \
    --num-layer 3 \
    --lr 1e-4 \
    --beta1 0.5 \
    --w-gamma 1.0 \
    --w-g 1.0 \
    --num-iters 25000

# Sampling (generate flat rows)
python -m src.predict \
  --dataset \
    --seq-len 128 \
    --data-dir ./data \
    --orderbook-filename AMZN_2012-06-21_34200000_57600000_orderbook_10.csv \
    --splits 0.7 0.85 1.0 \
  --modules \
    --batch-size 128 \
    --z-dim 40 \
    --hidden-dim 64 \
    --num-layer 3

# Visualisation + metrics + latent walks
python -m src.viz.visualise \
  --dataset \
    --seq-len 128 \
    --data-dir ./data \
    --orderbook-filename AMZN_2012-06-21_34200000_57600000_orderbook_10.csv \
  --modules \
    --batch-size 128 \
    --z-dim 40 \
    --hidden-dim 64 \
    --num-layer 3 \
  --viz \
    --samples 5 \
    --out-dir ./outs/viz_run1 \
    --cmap magma \
    --dpi 240 \
    --bins 128 \
    --levels 10 \
    --metrics-csv ./outs/viz_run1/metrics.csv \
    --walk --walk-steps 8 --walk-mode cross --walk-prefix latent_walk
