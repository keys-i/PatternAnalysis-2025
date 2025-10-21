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

