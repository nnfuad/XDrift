# Table 2: Feature Attribution Drift and Explainer Repeatability (Cohort N=100)

*All metrics evaluated on identical 100-flow balanced test cohort (50 VPN, 50 Non-VPN). Attributions explain the pre-sigmoid VPN logit with 50 background samples and 150 KernelSHAP evaluations per flow.*

### Panel A: Cohort Aggregate Attribution Metrics
| Condition / Comparison | Explainer Seeds | Top-5 Jaccard ($IoU_5$) | Top-10 Jaccard ($IoU_{10}$) | Total Variation ($TV$) | Signed Cosine ($S_C$) | Spearman Rank ($r_s$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Combinatorial Random Chance** | — | 0.0220 | 0.0426 | — | 0.0000 | 0.0000 |
| **Explainer Noise: FP32 (s42 vs s43)** | 42 vs 43 | 0.9012 ± 0.2614 | 0.9328 ± 0.2014 | 0.1064 ± 0.1741 | 0.9210 ± 0.1979 | 0.8950 ± 0.1872 |
| **Explainer Noise: FP32 (s42 vs s44)** | 42 vs 44 | 0.9055 ± 0.2587 | 0.9360 ± 0.1995 | 0.1070 ± 0.1746 | 0.9238 ± 0.1975 | 0.8988 ± 0.1842 |
| **Explainer Noise (3-Pair Mean)** | Multi-seed | **0.9066 ± 0.2562** | **0.9372 ± 0.1970** | **0.1050 ± 0.1738** | **0.9258 ± 0.1971** | **0.9015 ± 0.1824** |
| **FP32 vs INT8-PTQ (Cross-Model)** | 42 vs 42 | **0.9431 ± 0.1697** | **0.9638 ± 0.1270** | **0.0907 ± 0.1387** | **0.9629 ± 0.1066** | **0.9188 ± 0.1535** |
| **FP32 vs INT8-QAT (Cross-Model)** | 42 vs 42 | **0.8934 ± 0.2788** | **0.9246 ± 0.2104** | **0.3446 ± 0.1422** | **0.2672 ± 0.6974** | **0.6208 ± 0.3812** |

---

### Panel B: Paired Flow-by-Flow Hypothesis Testing (Cross-Model vs. Explainer Noise)
*Evaluates whether cross-model attribution drift exceeds intrinsic explainer variability on identical flows.*

| Comparison | Paired Delta Metric | Mean ± Std | Median | Non-Parametric Bootstrap 95% CI | Wilcoxon Signed-Rank Test | Statistical Conclusion |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PTQ vs. Noise** | $\Delta TV = TV_{\text{PTQ}} - TV_{\text{Noise}}$ | -0.0157 ± 0.1716 | 0.0000 | $[-0.0494, +0.0166]$ | $W = 1482.0, p = 0.34516$ | **Indistinguishable from Noise** |
| **PTQ vs. Noise** | $\Delta IoU_5 = IoU_{5, \text{PTQ}} - IoU_{5, \text{Noise}}$ | +0.0419 ± 0.1313 | 0.0000 | $[+0.0196, +0.0704]$ | $W = 38.0, p = 0.00330$ | Identical Top-5 on 89/100 flows |
| **QAT vs. Noise** | $\Delta TV = TV_{\text{QAT}} - TV_{\text{Noise}}$ | +0.2382 ± 0.1654 | +0.2308 | $[+0.1923, +0.2863]$ | $W = 4491.0, p = 8.80 \times 10^{-14}$ | **Massive Significant Drift ($p < 10^{-13}$)** |
| **QAT vs. PTQ** | $\Delta TV = TV_{\text{QAT}} - TV_{\text{PTQ}}$ | +0.2539 ± 0.1582 | +0.2461 | $[+0.2104, +0.2982]$ | $W = 4510.0, p = 4.38 \times 10^{-14}$ | QAT restructures attributions ($p < 10^{-13}$) |

---

### Panel C: Class-Specific Attribution Metrics (VPN vs. Non-VPN Subgroups)
| Metric | Overall ($N=100$) | VPN Flows Only ($N=50$) | Non-VPN Flows Only ($N=50$) | Class Difference (VPN - Non-VPN) |
| :--- | :---: | :---: | :---: | :---: |
| **PTQ Top-5 Jaccard ($IoU_5$)** | 0.9431 | 0.8976 | 0.9886 | -0.0910 |
| **PTQ Total Variation ($TV$)** | 0.0907 | 0.1601 | 0.0212 | +0.1389 |
| **PTQ Signed Cosine ($S_C$)** | 0.9629 | 0.9298 | 0.9960 | -0.0662 |
| **QAT Top-5 Jaccard ($IoU_5$)** | 0.8934 | 0.8037 | 0.9831 | -0.1794 |
| **QAT Total Variation ($TV$)** | 0.3446 | 0.3546 | 0.3346 | +0.0200 |
| **QAT Signed Cosine ($S_C$)** | 0.2672 | -0.0638 | 0.5982 | -0.6620 |
