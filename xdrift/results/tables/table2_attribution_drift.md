# Table 2: Feature Attribution Drift and Explainer Repeatability (Cohort N=100)

*All metrics evaluated on identical 100-flow balanced test cohort (50 VPN, 50 Non-VPN). Attributions explain the pre-sigmoid VPN logit with 50 background samples and 150 KernelSHAP evaluations per flow. Zero-padding tokens are masked.*

---

### Panel A: Cohort Aggregate Attribution Metrics
| Condition / Comparison | Explainer Seeds | Top-5 Jaccard ($IoU_5$) | Top-10 Jaccard ($IoU_{10}$) | Total Variation ($TV$) | Signed Cosine ($S_C$) | Spearman Rank ($r_s$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Combinatorial Random Chance** | — | 0.0220 | 0.0426 | 1.0000 | 0.0000 | 0.0000 |
| **Explainer Noise: FP32 (s42 vs s43)** | 42 vs 43 | 0.9557 ± 0.1840 | 0.9652 ± 0.1563 | 0.0568 ± 0.1623 | 0.9648 ± 0.1545 | 0.9489 ± 0.1633 |
| **Explainer Noise: FP32 (s42 vs s44)** | 42 vs 44 | 0.9547 ± 0.1834 | 0.9670 ± 0.1532 | 0.0541 ± 0.1547 | 0.9696 ± 0.1417 | 0.9569 ± 0.1581 |
| **Explainer Noise Floor (3-Pair Mean)** | Multi-seed | **0.9553 ± 0.1829** | **0.9648 ± 0.1592** | **0.0553 ± 0.1588** | **0.9676 ± 0.1450** | **0.9516 ± 0.1666** |
| **FP32 vs INT8-PTQ (Cross-Model)** | 42 vs 42 | **0.9786 ± 0.0970** | **0.9887 ± 0.0650** | **0.0369 ± 0.0620** | **0.9918 ± 0.0394** | **0.9749 ± 0.0592** |
| **FP32 vs INT8-QAT (Cross-Model)** | 42 vs 42 | **0.9553 ± 0.1841** | **0.9679 ± 0.1416** | **0.2346 ± 0.1908** | **0.4830 ± 0.3449** | **0.9140 ± 0.1838** |
| **FP32 vs FP32-FT (Ablation Control)** | 42 vs 42 | **0.9669 ± 0.1375** | **0.9762 ± 0.1169** | **0.1219 ± 0.1137** | **0.9554 ± 0.0953** | **0.9687 ± 0.1092** |

*Note: For FP32-FT, median values are: $IoU_5 = 1.0000$, $IoU_{10} = 1.0000$, $TV = 0.1507$, $S_C = 0.9672$, $r_s = 1.0000$.*

---

### Panel B: Paired Flow-by-Flow Hypothesis Testing (Cross-Model vs. Explainer Noise & Ablation Control)
*Evaluates whether cross-model attribution drift exceeds intrinsic explainer variability or differs between quantization and SGD optimization on identical flows.*

| Comparison | Paired Delta Metric | Mean ± Std | Median | Non-Parametric Bootstrap 95% CI | Wilcoxon Signed-Rank Test | Statistical Conclusion |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PTQ vs. Noise** | $\Delta TV = TV_{\text{PTQ}} - TV_{\text{Noise}}$ | $-0.0185 \pm 0.1115$ | $+0.0049$ | $[-0.0422, +0.0011]$ | $W = 1506.0$ ($R^+ = 3544.0, R^- = 1506.0$)<br>Two-sided $p = 0.00046$ (Holm: 0.00092)<br>One-sided ($H_1: \Delta TV > 0$) $p = 0.00023$ | Positive rank shift in 69/100 flows ($R^+ > R^-$), minimal median delta (+0.0049) |
| **PTQ vs. Noise** | $\Delta IoU_5 = IoU_{5, \text{PTQ}} - IoU_{5, \text{Noise}}$ | $+0.0232 \pm 0.1088$ | $+0.0000$ | $[+0.0051, +0.0472]$ | $W = 0.0, p = 0.02771$ (Holm: 0.02771) | Identical Top-5 on 95/100 flows |
| **QAT vs. Noise** | $\Delta TV = TV_{\text{QAT}} - TV_{\text{Noise}}$ | $+0.1793 \pm 0.1547$ | $+0.1975$ | $[+0.1500, +0.2071]$ | $W = 129.0, p = 1.7478 \times 10^{-16}$ (Holm: $5.24 \times 10^{-16}$) | **Significant Drift ($p < 10^{-4}$)** |
| **QAT vs. PTQ** | $\Delta TV = TV_{\text{QAT}} - TV_{\text{PTQ}}$ | $+0.1978 \pm 0.1596$ | $+0.2854$ | $[+0.1660, +0.2301]$ | $W = 0.0, p = 3.8966 \times 10^{-18}$ (Holm: $1.56 \times 10^{-17}$) | QAT restructures attributions ($p < 10^{-4}$) |
| **FP32-FT vs. Noise** | $\Delta TV = TV_{\text{FT}} - TV_{\text{Noise}}$ | $+0.0666 \pm 0.1188$ | $+0.0294$ | $[+0.0423, +0.0887]$ | $W = 821.0, p = 4.66 \times 10^{-9}$ (Holm: $9.32 \times 10^{-9}$) | Moderate drift from gradient fine-tuning ($p < 10^{-4}$) |
| **QAT vs. FP32-FT** | $\Delta TV = TV_{\text{QAT}} - TV_{\text{FT}}$ | $+0.1127 \pm 0.2221$ | $+0.2016$ | $[+0.0684, +0.1555]$ | $W = 1146.0, p = 2.12 \times 10^{-6}$ (Holm: $2.12 \times 10^{-6}$) | QAT drift significantly exceeds FP32-FT ($p < 10^{-4}$) |

---

### Panel C: Class-Specific Attribution Metrics (VPN vs. Non-VPN Subgroups)
| Metric | Overall ($N=100$) | VPN Flows Only ($N=50$) | Non-VPN Flows Only ($N=50$) | Class Difference (VPN $-$ Non-VPN) |
| :--- | :---: | :---: | :---: | :---: |
| **PTQ Top-5 Jaccard ($IoU_5$)** | 0.9786 | 0.9638 | 0.9933 | $-0.0295$ |
| **PTQ Total Variation ($TV$)** | 0.0369 | 0.0388 | 0.0349 | $+0.0040$ |
| **PTQ Signed Cosine ($S_C$)** | 0.9918 | 0.9851 | 0.9984 | $-0.0133$ |
| **QAT Top-5 Jaccard ($IoU_5$)** | 0.9553 | 0.9350 | 0.9756 | $-0.0406$ |
| **QAT Total Variation ($TV$)** | 0.2346 | 0.1163 | 0.3529 | $-0.2366$ |
| **QAT Signed Cosine ($S_C$)** | 0.4830 | 0.1665 | 0.7996 | $-0.6331$ |
| **FP32-FT Top-5 Jaccard ($IoU_5$)** | 0.9669 | 0.9519 | 0.9819 | $-0.0300$ |
| **FP32-FT Total Variation ($TV$)** | 0.1219 | 0.1980 | 0.0459 | $+0.1521$ |
| **FP32-FT Signed Cosine ($S_C$)** | 0.9554 | 0.9187 | 0.9921 | $-0.0734$ |
