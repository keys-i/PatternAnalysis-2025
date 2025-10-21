# TimeGAN for Synthetic Limit Order Books (AMZN, LOBSTER Level-10)

**COMP3710 - Pattern Recognition and Analysis**

<table>
  <tbody>
    <tr>
      <td>Task 14</td>
      <td>Generative Model of AMZN LOBSTER Level-10 using TimeGAN</td>
    </tr>
    <tr>
      <td>Author</td>
      <td>Radhesh Goel (49088276)</td>
    </tr>
  </tbody>
</table>

## Project Overview

This project trains a TimeGAN model to generate synthetic sequences of limit order book events from the LOBSTER dataset
using AMZN level 10 depth. The motivation is to ease data scarcity and confidentiality constraints in microstructure
research, enable safer augmentation for downstream forecasting, and allow controlled experiments on price and depth
dynamics without relying on live market streams. The synthetic sequences are intended to improve robustness, support
reproducibility, and help probe edge cases that are rare in historical data.

Quality is assessed on a held out test split using objective targets:

* Distribution similarity: KL divergence at or below 0.1 for spread and midprice return distributions between generated
  and real data.
* Visual similarity: SSIM above 0.6 between heatmaps of generated and real order book depth snapshots.

The report will include the model architecture and parameter count, the training strategy with ablations, compute
details such as GPU type and VRAM, the number of epochs, total training time, and 3 to 5 paired heatmaps with a concise
error analysis.

## Model Description

TimeGAN integrates both adversarial and supervised learning objectives to model the temporal structure of financial
sequences. The architecture consists of five main components, each contributing to the generation and recovery of
realistic limit order book sequences:

1. **Encoder**: maps observed LOB windows into a lower-dimensional latent representation that captures underlying
   market dynamics.
2. **Recovery Network**: reconstructs original price and depth features from the latent space, ensuring information
   consistency between real and encoded data.
3. **Generator**: transforms random noise vectors into synthetic latent sequences that emulate the structure of encoded
   real data.
4. **Supervisor**: predicts the next step in a latent sequence, encouraging temporal coherence and realistic sequential
   transitions.
5. **Discriminator**: distinguishes between real and generated latent sequences, providing adversarial feedback to
   improve the generator’s realism.

Training follows three phases. First, pretrain Encoder and Recovery to minimize reconstruction error and anchor the
latent space to real LOB statistics. Second, train the Supervisor for next step prediction to align latent dynamics with
empirical transitions. Third, run joint adversarial training with discriminator loss plus simple moment and consistency
terms, yielding synthetic sequences that match real markets in distribution and temporal structure.

## Table of Contents

| #  | Section                                                             |
|----|---------------------------------------------------------------------|
| 1  | [Project Structure](#project-structure)                             |
| 2  | [Dependencies](#dependencies)                                       |
| 3  | [Usage](#usage)                                                     |
| 4  | [Dataset](#dataset)                                                 |
| 5  | [Data Setup](#data-setup)                                           |
| 6  | [Model Architecture](#model-architecture)                           |
| 7  | [Training Process](#training-process)                               |
| 8  | [Results](#results)                                                 |
| 9  | [Analysis of Performance Metrics](#analysis-of-performance-metrics) |
| 10 | [Style Space and Plot Discussion](#style-space-and-plot-discussion) |
| 11 | [References](#references)                                           |
| 12 | [Citation](#citation)                                               |

## Project Structure

The project consists of the following file structure:

```ansi
TimeLOB_TimeGAN_49088276/
├── README.MD                             # Project report (including configuration, setup, training methodology, performance evaluation)
├── environment.yml                       # conda environment with all dependencies.
├── scripts/
│         ├── run.sh                      # rangpur/local script for running the project
│         └── summarise_orderbook.py      # test script to get to know about the dataset
└── src/
    ├── dataset.py                        # data loader and preprocesser (includes data loading, scaling and normalising.)
    ├── helpers/
    │         ├── args.py                 # a nested options for the model and dataset so, those files are not bloated
    │         ├── constants.py            # root-anchored paths, defaults and training constants
    │         ├── richie.py               # a common interface for pretty console logging, status spinners, tables.
    │         ├── utils.py                # metrics and utilities (KL, scaling, noise, specific feature calculators)
    │         └── visualise.py            # plotting helpers for depth heatmaps, curves, and summaries (SSIM score calculators)
    ├── modules.py                        # TimeGAN model components, training loops, checkpoints, metrics hooks.
    ├── predict.py                        # Sampling script to generate synthetic LOB sequences from a checkpoint.
    └── train.py                          # CLI entrypoint that parses options and runs training.
```

## Dependencies

Training was carried out on **macOS (BSD Unix)** using an Apple M3 Pro system; the codebase is also compatible with
Linux.
Windows was not used for training.
> **Note**
> Hardware: Apple M3 Pro GPU with MLS/Metal support, o[environment.yml](environment.yml)r equivalent; at least 8 GB of
> unified memory is advisable.

| Dependency        | Suggested version | One-line use case                                                                         |
|-------------------|------------------:|-------------------------------------------------------------------------------------------|
| Python            |            3.13.9 | Runtime for training, sampling, evaluation scripts, and utilities.                        |
| torch (PyTorch)   |             2.8.0 | Core framework for TimeGAN modules, tensor ops, autograd, and device acceleration.        |
| torchvision       |            0.24.0 | Utility helpers (e.g., image save utilities) for exporting depth heatmaps when needed.    |
| numpy             |             2.3.4 | Fast array math for windowing LOB data, metrics, and numerical transforms.                |
| matplotlib        |            3.10.7 | Plots for training curves, spread/return histograms, and LOB depth heatmaps.              |
| scikit-learn      |             1.7.2 | Analysis utilities (e.g., PCA/MI) in feature studies and ablations outside core training. |
| scikit-image      |            0.25.2 | SSIM computation to compare real vs synthetic heatmaps.                                   |
| tqdm              |            4.67.1 | Progress bars for three-phase training with periodic metric updates.                      |
| contextvars       |               2.4 | Context-local state to keep logging and progress output tidy across workers.              |
| rich              |            14.2.0 | Pretty console logs, status spinners, and summary tables during data prep and training.   |
| typing-extensions |            4.15.0 | Modern typing features (Protocol, Literal) used in model and CLI code.                    |
| scipy             |            1.16.3 | Statistical routines (e.g., Spearman correlation) for analysis scripts.                   |
| pillow (PIL)      |            12.0.0 | Image IO/encoding backend for saving figures and heatmaps to PNG.                         |
| pandas            |             2.3.3 | Tabular processing for order book summaries and feature engineering notebooks.            |
| jupyterlab        |            4.4.10 | Interactive exploration of LOB data, metrics, and experiment reports.                     |
| ipykernel         |             7.1.0 | Jupyter kernel to run notebooks for analysis and visualization.                           |

## Usage

### Training

Trains TimeGAN on AMZN level-10 LOBSTER windows with a three-phase schedule:

1. **Encoder–Recovery** pretrain for reconstruction, 2) **Supervisor** pretrain for next-step consistency, 3) **Joint
   adversarial
   training** with moment matching. Periodic validation computes KL on spread and midprice returns; checkpoints are
   saved
   regularly and depth heatmaps can be rendered for SSIM checks.

```bash
# start training from scratch (nested CLI: dataset namespace then modules namespace)
python src/train.py \
  --dataset \
    --seq-len 128 \
    --data-dir ./data \
    --orderbook-filename AMZN_2012-06-21_34200000_57600000_orderbook_10.csv \
    --splits 0.7 0.85 1.0 \
  --modules \
    --batch-size 128 \
    --z-dim 40 \
    --hidden-dim 64 \
    --num-layer 3 \
    --lr 1e-4 \
    --beta1 0.5 \
    --num-iters 25000
```

| Hyperparameter         | Value             | Notes                                      |
|------------------------|-------------------|--------------------------------------------|
| batch size             | 128               | Larger batches stabilize adversarial steps |
| `seq_len`              | 128               | Window length for LOB sequences            |
| `z_dim`                | 40                | Matches raw10 feature count                |
| `hidden_dim`           | 64                | GRU hidden size across components          |
| `layers`               | 3                 | Stacked GRU depth                          |
| `optimizer`            | Adam              | β1 tuned for GAN stability                 |
| `learning rate`        | 1e-4              | Shared across E, R, G, S, D                |
| β1                     | 0.5               | Momentum term for Adam                     |
| `iterations per phase` | 25,000            | ER, Supervisor, and Joint phases each      |
| scaling                | train-only MinMax | Fit on train split, apply to val/test      |

- **Outputs**:
    - `weights/timegan_ckpt.pt` (latest checkpoint)
    - `outs/` (generated samples, KL/SSIM plots, training curves, summaries)

### Generation

The `predict.py` script samples synthetic LOB data from a trained **TimeGAN** checkpoint. It supports flat row
generation, windowed generation, optional heatmap rendering, and quick metric checks.

#### 1. Generate flat rows (match test length)

Produces exactly `len(test)` rows in original feature space and saves as NumPy.

```bash
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
```

#### 2. Generate a fixed number of rows

Specify `--rows` to override the default.

```bash
python src/predict.py \
  --dataset \
    --seq-len 128 \
    --data-dir ./data \
    --orderbook-filename AMZN_2012-06-21_34200000_57600000_orderbook_10.csv \
  --modules \
    --batch-size 128 \
    --z-dim 40 \
    --hidden-dim 64 \
    --num-layer 3
```

#### 3. Render depth heatmaps (real vs synthetic)

Creates side-by-side heatmaps for SSIM inspection.

```bash
python -m src.viz.ssim_heatmap \
  --dataset \
    --seq-len 128 \
    --data-dir ./data \
    --orderbook-filename AMZN_2012-06-21_34200000_57600000_orderbook_10.csv \
  --modules \
    --batch-size 128 \
     --z-dim 40 \
     --hidden-dim 64 \
     --num-layer 3
# saves outs/real.png and outs/synthetic_heatmap_{i}.png
```

#### 4. Quick metrics (KL and SSIM)

During generation or via post-hoc scripts you can compute:

* **KL(spread)** and **KL(midprice returns)** on a held-out slice
* **SSIM** between real and synthetic heatmaps

```bash
python src/helpers/visualise.py \
  --dataset \
    --seq-len 128 \
    --data-dir ./data \
    --orderbook-filename orderbook_10.csv \
  --modules \
    --batch-size 128 \
    --z-dim 40 \
    --hidden-dim 64 \
    --num-layer 3
```

#### 5. Export to CSV

```bash
python npy_to_csv.py \
  --in ./outs/gen_data.npy \
  --out ./outs/gen_data.csv \
  --peek 10 \
  --summary
```

### Command-line Arguments

#### Top-level (parsed by `Options`)

| Flag          | Type      | Default  | Description                                 | Example                        |
|---------------|-----------|----------|---------------------------------------------|--------------------------------|
| `--seed`      | int       | `42`     | Global random seed.                         | `--seed 1337`                  |
| `--run-name`  | str       | `"exp1"` | Label for the run; used in logs/artifacts.  | `--run-name lob_amzn_l10`      |
| `--dataset …` | namespace | —        | Tokens after this go to **DataOptions**.    | `--dataset --seq-len 128 …`    |
| `--modules …` | namespace | —        | Tokens after this go to **ModulesOptions**. | `--modules --batch-size 128 …` |

#### Data options (parsed by `DataOptions`)

| Flag                      | Type     | Default              | Description                                                               | Example                                        |
|---------------------------|----------|----------------------|---------------------------------------------------------------------------|------------------------------------------------|
| `--seq-len`               | int      | `128`                | Sliding window length for LOB sequences.                                  | `--seq-len 128`                                |
| `--data-dir`              | str      | `DATA_DIR`           | Directory containing LOBSTER files.                                       | `--data-dir ./data`                            |
| `--orderbook-filename`    | str      | `ORDERBOOK_FILENAME` | Name of `orderbook_10.csv`.                                               | `--orderbook-filename AMZN_…_orderbook_10.csv` |
| `--no-shuffle`            | flag     | off                  | Disable shuffling of windowed sequences.                                  | `--no-shuffle`                                 |
| `--keep-zero-rows`        | flag     | off                  | Do not filter rows with zeros.                                            | `--keep-zero-rows`                             |
| `--splits TRAIN VAL TEST` | 3× float | `TRAIN_TEST_SPLIT`   | Proportions summing to ~1.0 or cumulative cutoffs (e.g., `0.7 0.85 1.0`). | `--splits 0.7 0.85 1.0`                        |

#### Model/training options (parsed by `ModulesOptions`)

| Flag           | Type  | Default                   | Description                                    | Example             |
|----------------|-------|---------------------------|------------------------------------------------|---------------------|
| `--batch-size` | int   | `128`                     | Batch size for all phases.                     | `--batch-size 128`  |
| `--seq-len`    | int   | `128`                     | Mirror of data window length for convenience.  | `--seq-len 128`     |
| `--z-dim`      | int   | `40`                      | Noise/latent input dimension.                  | `--z-dim 40`        |
| `--hidden-dim` | int   | `64`                      | GRU hidden size across components.             | `--hidden-dim 64`   |
| `--num-layer`  | int   | `3`                       | Stacked GRU layers per block.                  | `--num-layer 3`     |
| `--lr`         | float | `1e-4`                    | Adam learning rate.                            | `--lr 1e-4`         |
| `--beta1`      | float | `0.5`                     | Adam β1 for GAN stability.                     | `--beta1 0.5`       |
| `--w-gamma`    | float | `1.0`                     | Weight on supervisor-related adversarial term. | `--w-gamma 1.0`     |
| `--w-g`        | float | `1.0`                     | Weight on generator losses and moments.        | `--w-g 1.0`         |
| `--num-iters`  | int   | `NUM_TRAINING_ITERATIONS` | Iterations per phase (ER, Supervisor, Joint).  | `--num-iters 25000` |

**Outputs:**

- `outs/gen_data.npy` flat synthetic rows `[T, F]` in original feature scale
- `outs/real.png`, `outs/synthetic_heatmap_{i}.png` depth heatmaps for SSIM
- Optional plots: `outs/kl_spread_curve.png`, `outs/training_curves.png` if enabled in training/eval scripts

## Dataset

We use the **LOBSTER** limit order book for **AMZN** at **level 10** depth. The primary file is
`AMZN_2012-06-21_34200000_57600000_orderbook_10.csv` containing 40 columns
`[ask_price_1, ask_size_1, …, ask_price_10, ask_size_10, bid_price_1, bid_size_1, …, bid_price_10, bid_size_10]`.
Place the file under `data/`. By default the code performs a **chronological** split into train, validation, and test to
avoid leakage across time.

Example depth visualizations are produced during evaluation as heatmaps in `outs/` for SSIM checks.

*Files expected*

* `data/AMZN_2012-06-21_34200000_57600000_orderbook_10.csv`
* Optional: additional sessions can be summarized with `scripts/summarise_orderbook.py`

---

## Data Setup

### Preprocessing for TimeGAN (see `src/dataset.py`)

Pipeline steps applied to the order book snapshots:

```text
1) Load orderbook_10.csv  →  ndarray [T, 40]
2) Optional filter: drop rows with any zero (configurable)
3) Chronological split: train / val / test (default 0.7 / 0.15 / 0.15 or cumulative 0.7 / 0.85 / 1.0)
4) Train-only MinMax scaling (fit on train, apply to val and test)
5) Sliding windows: shape [N, seq_len, 40], with optional shuffle for training
```

#### Key flags (nested CLI):

- **Dataset**: `--seq-len`, `--data-dir`, `--orderbook-filename`, `--splits`, `--keep-zero-rows`, `--no-shuffle`
- **Modules**: `--batch-size`, `--z-dim` (use 40 for raw10), `--hidden-dim`, `--num-layer`, `--lr`, `--beta1`,
  `--num-iters`

#### Typical command:

```bash
python src/train.py \
  --dataset \
    --seq-len 128 \
    --data-dir ./data \
    --orderbook-filename AMZN_2012-06-21_34200000_57600000_orderbook_10.csv \
    --splits 0.7 0.85 1.0 \
  --modules \
    --batch-size 128 \
    --z-dim 40 \
    --hidden-dim 64 \
    --num-layer 3 \
    --lr 1e-4 \
    --beta1 0.5 \
    --num-iters 25000
```

### Data Splits

- **Training**: first segment of the day by time (no shuffling during split)
- **Validation**: middle segment for periodic checks and model selection
- **Test**: final segment held out for reporting metrics
- **Method**: chronological index cutoffs, not random splitting

#### Evaluation uses:

- **Distribution similarity**: KL divergence for spread and midprice returns on the held out set
- **Visual similarity**: SSIM between depth heatmaps of real and generated books

Heatmaps and metrics are saved to `outs/` via the training hooks and `src/helpers/visualise`.

## References

## Citation

If you use this implementation in your research, please cite:

```bibtex
@misc{timegan_lobster_amzn_l10_2025,
  title   = {TimeGAN for LOBSTER: Synthetic Limit Order Book Sequences (AMZN Level-10)},
  author  = {Radhesh Goel},
  year    = {2025},
  url     = {https://github.com/keys-i/TimeLOB_TimeGAN_49088276},
  note    = {Three-phase TimeGAN training with KL/SSIM evaluation on AMZN L10}
}
```