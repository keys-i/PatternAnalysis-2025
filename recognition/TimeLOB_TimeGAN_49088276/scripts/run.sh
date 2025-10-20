#!/bin/bash

# script to run training on UQ Rangpur

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --partition=a100
#SBATCH --job-name=timegan-turing

 conda init
 conda env create -f environment.yml
 conda activate timegan

export PROJECT_ROOT="$PWD"
export PYTHONPATH="$PWD"

pwd

python -m src.train \
  --dataset \
    --seq-len 128 \
    --data-dir ./data \
    --orderbook-filename orderbook_10.csv \
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
    --num-iter 100

python -m src.predict \
  --dataset \
    --seq-len 128 \
    --data-dir ./data \
    --orderbook-filename orderbook_10.csv \
    --splits 0.7 0.85 1.0 \
  --modules \
    --batch-size 128 \
    --z-dim 40 \
    --hidden-dim 64 \
    --num-layer 3

python -m src.helpers.visualise  \
  --dataset \
    --seq-len 128 \
    --data-dir ./data \
    --orderbook-filename orderbook_10.csv \
  --modules \
    --batch-size 128 \
    --z-dim 40 \
    --hidden-dim 64 \
    --num-layer 3