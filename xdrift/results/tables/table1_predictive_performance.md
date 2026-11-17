# Table 1: Predictive Performance Comparison on Held-Out Test Set (N=49,251)

*Evaluated on the held-out test split (2,693 VPN, 46,558 Non-VPN flows). Decision thresholds are calibrated strictly on held-out validation data ($N=69,564$). All rates and metrics are reported to full experimental precision.*

---

### Panel A: Threshold-Independent Ranking Performance
| Metric | FP32 Baseline | INT8-PTQ | $\Delta$ (PTQ $-$ FP32) | INT8-QAT | $\Delta$ (QAT $-$ FP32) | FP32-FT (Ablation Control) | $\Delta$ (FT $-$ FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.9796 | 0.9785 | $-0.0012$ | **0.9834** | $+0.0038$ | 0.9819 | $+0.0023$ |
| **PR-AUC** | 0.5610 | 0.6102 | $+0.0492$ | **0.6478** | $+0.0868$ | 0.6128 | $+0.0518$ |
| **Brier Score** | 0.0585 | 0.0584 | $-0.0000$ | **0.0237** | $-0.0347$ | 0.0248 | $-0.0337$ |

---

### Panel B: Model-Specific Validation-Calibrated Thresholding (Recommended)
*Decision thresholds selected via validation F1 maximization: FP32 ($\tau = 0.900$), PTQ ($\tau = 0.900$), QAT ($\tau = 0.680$), FP32-FT ($\tau = 0.830$)*

| Metric | FP32 Baseline ($\tau = 0.900$) | INT8-PTQ ($\tau = 0.900$) | $\Delta$ (PTQ $-$ FP32) | INT8-QAT ($\tau = 0.680$) | $\Delta$ (QAT $-$ FP32) | FP32-FT ($\tau = 0.830$) | $\Delta$ (FT $-$ FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Threshold ($\tau$)** | 0.900 | 0.900 | $+0.000$ | 0.680 | $-0.220$ | 0.830 | $-0.070$ |
| **Positive Calls (VPN)** | 4,121 | 4,172 | $+51$ | 4,130 | $+9$ | 3,684 | $-437$ |
| **Confusion Matrix $[[TN, FP], [FN, TP]]$** | $[[45076, 1482], [54, 2639]]$ | $[[45028, 1530], [51, 2642]]$ | — | $[[45076, 1482], [45, 2648]]$ | — | $[[45481, 1077], [86, 2607]]$ | — |
| **Accuracy** | 96.88% | 96.79% | $-0.09\%$ | 96.90% | $+0.02\%$ | **97.64%** | $+0.76\%$ |
| **Balanced Accuracy** | 97.41% | 97.41% | $+0.00\%$ | **97.57%** | $+0.17\%$ | 97.25% | $-0.16\%$ |
| **Precision (VPN)** | 64.04% | 63.33% | $-0.71\%$ | 64.12% | $+0.08\%$ | **70.77%** | $+6.73\%$ |
| **Recall (VPN)** | 97.99% | 98.11% | $+0.11\%$ | **98.33%** | $+0.33\%$ | 96.81% | $-1.18\%$ |
| **F1-Score** | 0.7746 | 0.7697 | $-0.0049$ | 0.7762 | $+0.0016$ | **0.8176** | $+0.0430$ |

---

### Panel C: Reused FP32 Threshold ($\tau = 0.900$, Uncalibrated Baseline)
*Demonstrates operational vulnerability: logit contraction under QAT and distribution shift under fine-tuning cause severe positive call suppression.*

| Metric | FP32 Baseline ($\tau = 0.900$) | INT8-PTQ ($\tau = 0.900$) | $\Delta$ (PTQ $-$ FP32) | INT8-QAT ($\tau = 0.900$) | $\Delta$ (QAT $-$ FP32) | FP32-FT ($\tau = 0.900$) | $\Delta$ (FT $-$ FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Positive Calls (VPN)** | 4,121 | 4,172 | $+51$ | 0 | $-4,121$ | 1,693 | $-2,428$ |
| **Accuracy** | 96.88% | 96.79% | $-0.09\%$ | 94.53% | $-2.35\%$ | 94.45% | $-2.43\%$ |
| **Balanced Accuracy** | 97.41% | 97.41% | $+0.00\%$ | 50.00% | $-47.41\%$ | 64.42% | $-32.99\%$ |
| **Precision (VPN)** | 64.04% | 63.33% | $-0.71\%$ | 0.00% | $-64.04\%$ | 48.85% | $-15.19\%$ |
| **Recall (VPN)** | 97.99% | 98.11% | $+0.11\%$ | 0.00% | $-97.99\%$ | 30.71% | $-67.28\%$ |
| **F1-Score** | 0.7746 | 0.7697 | $-0.0049$ | **0.0000** | $-0.7746$ | 0.3771 | $-0.3975$ |
| **Prediction Flips** | — | 0.12% (58) | — | 8.37% (4,121) | — | 5.02% (2,470) | — |
