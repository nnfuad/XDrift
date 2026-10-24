# Table 1: Predictive Performance Comparison on Held-Out Test Set (N=20,971)

*Evaluated on the held-out test split (2,693 VPN, 18,278 Non-VPN flows). Thresholds calibrated strictly on held-out validation data.*

### Panel A: Threshold-Independent Ranking Performance
| Metric | FP32 Baseline | INT8-PTQ | Delta (PTQ - FP32) | INT8-QAT | Delta (QAT - FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.8806 | 0.7554 | -0.1252 | **0.9490** | +0.0684 |
| **PR-AUC** | 0.3834 | 0.3345 | -0.0489 | **0.5932** | +0.2098 |
| **Brier Score** | 0.5000 | 0.7746 | +0.2746 | **0.0656** | -0.4343 |

---

### Panel B: Model-Specific Validation-Calibrated Thresholding (Recommended)
*Thresholds chosen on validation set: FP32 ($\tau = 0.91$), PTQ ($\tau = 0.97$), QAT ($\tau = 0.66$)*

| Metric | FP32 Baseline ($\tau = 0.91$) | INT8-PTQ ($\tau = 0.97$) | Delta (PTQ - FP32) | INT8-QAT ($\tau = 0.66$) | Delta (QAT - FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Threshold ($\tau$)** | 0.910 | 0.970 | +0.060 | 0.660 | -0.250 |
| **Positive Calls (VPN)** | 5,671 | 4,271 | -1,400 | 4,465 | -1,206 |
| **Confusion Matrix $[[TN, FP], [FN, TP]]$** | $[[15290, 2988], [10, 2683]]$ | $[[15826, 2452], [874, 1819]]$ | — | $[[16462, 1816], [44, 2649]]$ | — |
| **Accuracy** | 85.70% | 84.14% | -1.56% | **91.13%** | +5.43% |
| **Balanced Accuracy** | 91.64% | 77.07% | -14.57% | **94.22%** | +2.58% |
| **Precision (VPN)** | 47.31% | 42.59% | -4.72% | **59.33%** | +12.02% |
| **Recall (VPN)** | 99.63% | 67.55% | -32.08% | **98.37%** | -1.26% |
| **F1-Score** | 0.6416 | 0.5224 | -0.1192 | **0.7402** | +0.0986 |

---

### Panel C: Reused FP32 Threshold ($\tau = 0.90$, Uncalibrated Baseline)
*Illustrates the impact of threshold incompatibility caused by logit contraction in QAT and logit upward shift in PTQ.*

| Metric | FP32 Baseline ($\tau = 0.90$) | INT8-PTQ ($\tau = 0.90$) | Delta (PTQ - FP32) | INT8-QAT ($\tau = 0.90$) | Delta (QAT - FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Positive Calls (VPN)** | 5,738 | 19,837 | +14,099 | 0 | -5,738 |
| **Accuracy** | 85.39% | 18.17% | -67.22% | 87.16%* | +1.76% |
| **Balanced Accuracy** | 91.48% | 52.93% | -38.55% | 50.00%* | -41.48% |
| **F1-Score** | 0.6367 | 0.2383 | -0.3983 | 0.0000* | -0.6367 |
| **Prediction Flip Rate** | — | 67.24% | — | 27.36% | — |

*\*Note: QAT test probabilities peak at 0.6653 due to logit contraction, causing an all-negative classifier at $\tau = 0.90$. Calibration on validation data (Panel B) resolves this artifact.*
