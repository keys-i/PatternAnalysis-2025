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
