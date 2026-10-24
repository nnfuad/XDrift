# 📊 Statistical Reanalysis and Methodological Rigor Audit

**Document ID:** `reports/audit/statistical_reanalysis.md`  
**Investigation:** Rigorous statistical recalculation of regressions, hypothesis tests, effect sizes, and chance baselines across the XDrift study.  
**Auditor:** Skeptical ML Research Engineer & Experimental Auditor  
**Date:** October 2026  

---

## 1. Sequence Length Regression Analysis

The study proposes that feature attribution drift is strongly correlated with active sequence length $L$ (the number of active packet slots before zero-padding). Below is the independent econometric and statistical audit of this relationship.

### 1.1 Model 1: Univariate OLS Regression ($N=746$)
$$\text{Top-5 IoU}_{\text{feat}} = \beta_0 + \beta_1 \cdot L + \varepsilon$$

| Parameter | Estimate | Std. Error | $95\%$ Confidence Interval | $t$-Statistic | $p$-Value |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intercept ($\beta_0$)** | 0.839170 | 0.011966 | [0.8157, 0.8626] | 70.13 | $< 10^{-150}$ |
| **Sequence Length ($L$)** | -0.014516 | 0.000414 | [-0.015328, -0.013705] | -35.06 | 9.4361e-160 |

- **Coefficient of Determination ($R^2$):** 0.622983 (Explains 62.30% of the variance in Top-5 IoU).
- **Pearson Correlation ($r$):** -0.789293 (Strong inverse association).
- **Residual Standard Error:** $\sigma = 0.2409$.

> [!WARNING]
> **Extrapolation Constraint:** The linear regression line crosses zero at $L \approx 57.8$ packets. Because Top-5 IoU is bounded in $[0, 1]$, linear OLS exhibits negative predictions for $L > 58$. In publication figures, this trendline must be clipped to zero, or modeled using a beta regression / logit link function.

---

### 1.2 Disaggregated Sub-Cohort Regressions
To determine whether sequence length explains variance within classes rather than merely reflecting class imbalance, we evaluate OLS within each class independently:

| Sub-Cohort | Sample Size | Intercept | Slope ($\beta_1$) | $95\%$ CI for Slope | $R^2$ | $t$-Stat | $p$-Value |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **VPN Tunnel Flows ($y=1$)** | $N=373$ | 0.6515 | -0.010556 | [-0.011554, -0.009558] | **0.5369** | -20.74 | 5.4486e-64 |
| **Non-VPN Browsing ($y=0$)** | $N=373$ | 0.9286 | -0.017270 | [-0.019030, -0.015511] | **0.4994** | -19.24 | 1.0611e-57 |

**Key Statistical Takeaway:**  
The inverse relationship between flow length and attribution drift is **not** an artifact of pooled data (Simpson's Paradox). Sequence length independently explains **53.69% of IoU variance within VPN flows** ($p = 5.45 \times 10^{-64}$) and **49.94% of variance within Non-VPN flows** ($p = 1.06 \times 10^{-57}$).

---

### 1.3 Multivariate Regression (Confounding & Mediation Analysis)
$$\text{Top-5 IoU}_{\text{feat}} = \beta_0 + \beta_L \cdot L + \beta_{\text{VPN}} \cdot y + \varepsilon$$

| Variable | Coefficient | Std. Error | $t$-Statistic | $p$-Value | Partial Impact |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Intercept** | 0.886838 | 0.012438 | 71.30 | $< 10^{-150}$ | Baseline IoU at $L=0$ |
| **Length ($L$)** | -0.012285 | 0.000460 | -26.73 | $< 10^{-100}$ | Reduces IoU by 0.0123 per active packet |
| **VPN Status ($y$)** | -0.182388 | 0.019608 | -9.30 | 0.0000e+00 | Independent class penalty of -0.1824 |

- **Multivariate Model $R^2$:** **0.662308** (66.23% explained variance).
- **Incremental F-Test ($L$ vs. $L + \text{VPN}$):** $F(1, 743) = 86.52$, $p < 10^{-18}$.
- **Scientific Interpretation:** Active sequence length and VPN status are both statistically significant predictors. Long duration accounts for the majority of attribution drift, but encrypted VPN encapsulation introduces an additional penalty of -0.1824 on Top-5 IoU beyond duration alone.

---

## 2. Combinatorial Chance Baseline Derivation

A common reviewer critique of Top-$k$ IoU is that selecting $k$ items from a small candidate pool trivially inflates the IoU, whereas selecting from a large candidate pool deflates it.

### Mathematical Formulation
Let $L$ be the active sequence length, yielding $M = 2L$ active candidate features (lengths and inter-arrival times). Let two independent random agents select subsets $A, B \subset \{1, \dots, M\}$ of size $k = \min(5, M)$ uniformly at random without replacement.
The intersection size $X = |A \cap B|$ follows a Hypergeometric distribution:
$$P(X = x) = \frac{\binom{k}{x} \binom{M - k}{k - x}}{\binom{M}{k}}, \quad \max(0, 2k - M) \le x \le k$$

Since $|A \cup B| = 2k - X$, the exact expected Jaccard Index (IoU) is:
$$\mathbb{E}[\text{IoU}(A, B)] = \sum_{x} \frac{x}{2k - x} \cdot \frac{\binom{k}{x} \binom{M - k}{k - x}}{\binom{M}{k}}$$

### Empirical vs. Theoretical Chance Comparison

| Active Packets ($L$) | Active Features ($M=2L$) | Effective $k$ | Hypergeometric Chance IoU | Empirical Non-VPN IoU | Empirical VPN IoU | Empirical / Chance Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$L=4$ (Non-VPN Med)** | $8$ | $5$ | **$0.4707$** | **$0.7839$** | N/A | $1.67\times$ above chance |
| **$L=10$** | $20$ | $5$ | **$0.1481$** | $0.6201$ | $0.4812$ | $3.25\times$ above chance |
| **$L=24$ (VPN Median)** | $48$ | $5$ | **$0.0604$** | N/A | **$0.3281$** | **$5.43\times$ above chance** |
| **$L=64$ (Max Length)** | $128$ | $5$ | **$0.0220$** | N/A | **$0.1418$** | **$6.45\times$ above chance** |

> [!IMPORTANT]
> **Rigor Defense Conclusion:**  
> While the combinatorial contraction of the random baseline ($0.4707 \to 0.0220$) explains why short Non-VPN flows appear more stable, the empirical VPN IoU ($0.3281$) remains **over $5.4\times$ above chance**. This proves that the INT8 model retains genuine semantic attributions and that the observed drift reflects true structural boundary changes rather than random noise.

---

## 3. Two-Sample Distributional Hypothesis Testing

Because attribution distributions on network traffic exhibit heavy tails and skewness, parametric Student's $t$-tests are inappropriate. We report two-sided **Mann–Whitney $U$ tests** alongside the non-parametric **Rank-Biserial Correlation ($r_{rb}$)** effect size:

$$r_{rb} = 1 - \frac{2U}{n_1 n_2}$$

| Metric | Sample Sizes ($n_1, n_2$) | Mann–Whitney $U$ | Two-Sided $p$-Value | Rank-Biserial $r_{rb}$ | Effect Magnitude |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Top-5 Feature IoU** | $n_1=373, n_2=373$ | $U = 24917.5$ | **3.4343e-55** | **0.6418** | Large ($> 0.50$) |
| **Spearman Rank $\rho$** | $n_1=373, n_2=373$ | $U = 28243.5$ | **7.5467e-45** | **0.5940** | Large ($> 0.50$) |
| **Normalized $L_2$ Distance**| $n_1=373, n_2=373$ | $U = 111259.5$ | **1.4490e-45** | **-0.5994** | Large ($> 0.50$) |

All three attribution metrics exhibit large, statistically significant differences ($p < 10^{-44}$), confirming that attribution drift between VPN and Non-VPN traffic is a profound, reproducible phenomenon.
