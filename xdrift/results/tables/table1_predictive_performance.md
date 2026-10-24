# Table 1: Predictive Performance Comparison on Held-Out Test Set (N=49,251)

*Evaluated on the held-out test split (2,693 VPN, 46,558 Non-VPN flows). Thresholds calibrated strictly on held-out validation data.*

### Panel A: Threshold-Independent Ranking Performance
| Metric | FP32 Baseline | INT8-PTQ | Delta (PTQ - FP32) | INT8-QAT | Delta (QAT - FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.9796 | 0.9785 | -0.0012 | **0.9834** | +0.0038 |
| **PR-AUC** | 0.5610 | 0.6102 | +0.0492 | **0.6478** | +0.0868 |
| **Brier Score** | 0.0585 | 0.0584 | -0.0000 | **0.0237** | -0.0347 |

---

### Panel B: Model-Specific Validation-Calibrated Thresholding (Recommended)
*Thresholds chosen on validation set: FP32 ($\tau = 0.900$), PTQ ($\tau = 0.900$), QAT ($\tau = 0.680$)*

| Metric | FP32 Baseline ($\tau = 0.90$) | INT8-PTQ ($\tau = 0.90$) | Delta (PTQ - FP32) | INT8-QAT ($\tau = 0.68$) | Delta (QAT - FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Decision Threshold ($\tau$)** | 0.900 | 0.900 | +0.000 | 0.680 | -0.220 |
| **Positive Calls (VPN)** | 4,121 | 4,172 | +51 | 4,130 | +9 |
| **Confusion Matrix $[[TN, FP], [FN, TP]]$** | $[[45076, 1482], [54, 2639]]$ | $[[45028, 1530], [51, 2642]]$ | — | $[[45076, 1482], [45, 2648]]$ | — |
| **Accuracy** | 96.88% | 96.79% | -0.09% | **96.90%** | +0.02% |
| **Balanced Accuracy** | 97.41% | 97.41% | +0.00% | **97.57%** | +0.17% |
| **Precision (VPN)** | 64.04% | 63.33% | -0.71% | **64.12%** | +0.08% |
| **Recall (VPN)** | 97.99% | 98.11% | +0.11% | **98.33%** | +0.33% |
| **F1-Score** | 0.7746 | 0.7697 | -0.0049 | **0.7762** | +0.0016 |

---

### Panel C: Reused FP32 Threshold ($\tau = 0.90$, Uncalibrated Baseline)
*Illustrates the impact of threshold incompatibility caused by logit contraction in QAT and logit upward shift in PTQ.*

| Metric | FP32 Baseline ($\tau = 0.90$) | INT8-PTQ ($\tau = 0.90$) | Delta (PTQ - FP32) | INT8-QAT ($\tau = 0.90$) | Delta (QAT - FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Positive Calls (VPN)** | 4,121 | 4,172 | +51 | 0 | -4,121 |
| **Accuracy** | 96.88% | 96.79% | -0.09% | 94.53% | -2.35% |
| **Balanced Accuracy** | 97.41% | 97.41% | +0.00% | 50.00% | -47.41% |
| **F1-Score** | 0.7746 | 0.7697 | -0.0049 | 0.0000 | -0.7746 |
| **Prediction Flip Rate** | — | 0.12% | — | 8.37% | — |
