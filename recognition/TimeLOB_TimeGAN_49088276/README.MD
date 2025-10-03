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
This project trains a generative time series model to produce realistic sequences of limit order book events using the LOBSTER dataset, focusing on AMZN Level 10 data. The aim is to create high quality synthetic LOB sequences that can expand training sets for market microstructure research where balanced, fine grained data is expensive and difficult to collect. By learning the dynamics of spreads, midprice movements, and depth across ten levels, the model seeks to capture both short term fluctuations and broader order flow patterns.

Quality is assessed on a held out test split using objective targets:
- Distribution similarity: KL divergence at or below 0.1 for spread and midprice return distributions between generated and real data.
- Visual similarity: SSIM above 0.6 between heatmaps of generated and real order book depth snapshots.

The report will document the model architecture and total parameter count, and compare training strategies such as full TimeGAN, adversarial only, and supervised only variants. It will record the hardware used, including GPU model, available VRAM, number of epochs, and total training time. To aid interpretation, the report will include three to five representative heatmaps that pair generated and real order books, along with a short error analysis that explains where the synthetic sequences align with reality and where they fall short. The goal is a practical, well evidenced benchmark for synthetic LOB generation on AMZN Level 10.
