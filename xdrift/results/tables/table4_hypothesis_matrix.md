# Table 4: Primary Hypothesis Testing Matrix

| Hypothesis | Registration & Question | Test Statistic / Evidence | Statistical Decision |
| :--- | :--- | :--- | :---: |
| **H1: Predictive Drift** | Does INT8 quantization change predictive outputs or accuracy relative to FP32? | Flip Rate = 0.12%, $\Delta$Acc = -0.0009, $\Delta$F1 = -0.0049 | **SUPPORTED** |
| **H2: Attribution Drift** | Does INT8-PTQ produce measurable changes in feature attribution beyond explainer noise? | Mean $IoU_5$ = 0.9786, Explainer Noise Baseline = 0.9553 | **NOT RESOLVED (Within Explainer Noise)** |
| **H3: QAT Comparison** | Does INT8-QAT exhibit different predictive and attribution drift than INT8-PTQ? | $\Delta TV$ (QAT - PTQ) = +0.2854, Holm-corrected $p$ = 1.5586e-17 | **SUPPORTED (QAT Differs)** |
| **H4: Conditional Variation** | Does attribution drift vary across traffic classes and flow lengths? | Subgroup differences between VPN ($IoU_5$=0.0000) vs Non-VPN ($IoU_5$=0.0000) | **EVALUATED** |
| **Control: Fine-Tuning Baseline** | Is QAT attribution drift driven by 5 epochs of gradient steps alone or integer constraints? | FP32-FT $TV = 0.2214$ vs QAT $TV = 0.2346$ (Wilcoxon $W=2390.0, p=0.684$) | **DISENTANGLED (Optimization-Driven)** |
