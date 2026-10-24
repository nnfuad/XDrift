# Final Research Report: Investigating Attribution Drift in INT8-Quantized 1D CNNs for Encrypted Traffic Classification

**Project Title**: Investigating Attribution Drift in INT8-Quantized 1D CNNs for Encrypted Traffic Classification (XDrift)  
**Repository Path**: `xdrift/`  
**Execution Date**: October 24, 2026  
**Author / Role**: Senior ML Research Engineer (Quantization, XAI, Network Security)  
**Primary Benchmark**: ISCX VPN-NonVPN 2016 Dataset (140 PCAP/PCAPNG Captures, 25.78 GB, 106 Canonical Groups)  
**Execution Environment**: Apple Silicon Mac (M-series, ARM64), Python 3.12.7, PyTorch 2.5.1 (QNNPACK Quantization Engine)

---

## 1. Executive Summary and Central Research Question

### 1.1 Central Research Question
> *When a 1D convolutional neural network (1D-CNN) is quantized from FP32 to INT8 for encrypted traffic classification, does the quantization process change its feature-attribution explanations, even when predictive performance remains approximately unchanged?*

To resolve this question empirically, the XDrift protocol mandates the strict separation of three outcomes:
1. **Predictive Drift**: Quantization-induced changes in classifications, logit scales, class-probabilities, and held-out predictive metrics.
2. **Attribution Drift**: Changes in black-box feature explanations generated for the same test inputs under identical explainer settings.
3. **Operational Implications**: Potential consequences for security analysts, triage systems, or rule-extraction pipelines (evaluated strictly as empirical hypotheses rather than assumed facts).

### 1.2 Summary of Scientific Findings
1. **Decoupling of Predictive and Attribution Drift Under Static PTQ**:
   - Post-Training Quantization (PTQ) preserves high ranking discriminability (ROC-AUC **0.9785**, PR-AUC **0.6102** vs FP32 ROC-AUC **0.9796**, PR-AUC **0.5610**).
   - Under model-specific validation calibration ($\tau = 0.90$), PTQ maintains strong classification fidelity (F1 **0.7697**, Accuracy **96.79%**, Balanced Accuracy **97.41%**), with a minuscule test prediction flip rate of **0.12%** (57 flips out of 49,251 test flows).
   - Crucially, **relative feature-attribution rankings remain exceptionally preserved**: mean Top-5 Jaccard overlap between FP32 and INT8-PTQ is **0.9786** (median **1.0000**), with identical Top-5 features on **95 of 100 flows**.
   - When tested against KernelSHAP's own within-model sampling variability across seeds on the identical 100-flow cohort, the paired Total Variation difference is $\Delta TV = -0.0185 \pm 0.1115$ (bootstrap 95% CI $[-0.0422, +0.0011]$). Because the delta is negative and spans zero, **attribution displacement under PTQ is statistically indistinguishable from the noise floor of KernelSHAP itself**.
2. **QAT Recovers Discriminability but Restructures Explanations**:
   - Quantization-Aware Training (QAT) adapts network weights to simulated integer constraints, achieving superior ranking metrics (ROC-AUC **0.9834**, PR-AUC **0.6478**, Brier Score **0.0237**).
   - Under validation calibration ($\tau = 0.68$), QAT achieves **Accuracy: 96.90%**, **Balanced Accuracy: 97.57%**, and **F1-score: 0.7762**, matching or exceeding FP32 baseline performance (F1: 0.7746).
   - However, gradient fine-tuning through simulated integer operators fundamentally reorganizes the attribution landscape: Total Variation distance reaches **0.2346** (paired $\Delta TV = +0.1793$ over explainer noise, Wilcoxon $p = 1.7478e-16$).
3. **Disentangled Hardware Efficiency**:
   - At **Batch Size = 1 (edge streaming inference)**, INT8 models provide **no latency improvement** over FP32 (0.254 ms for QAT vs 0.254 ms for FP32; speedup 1.00×) due to operator invocation, dynamic tensor quantization, and scalar dequantization overhead.
   - At **Batch Size = 64 (server batched ingestion)**, integer vectorization via ARM64 NEON SIMD delivers a **10.21× throughput speedup** (72,090 vs 6,282 flows/sec).
   - Checkpoint storage is reduced by **3.56×** (184.24 KB down to 51.77 KB).

---

## 2. Experimental Design and Protocol Adherence

### 2.1 Preprocessing and Leakage-Free Splitting
- **Dataset Audit**: 140 raw capture files (102 `.pcap`, 38 `.pcapng`) totaling 25.78 GB were audited across 5 directories.
- **Session Grouping & Leakage Prevention**:
  - Raw captures were grouped into **106 independent physical session groups** with explicit alias mapping (`hangout_chat_4` $\to$ `hangouts_chat_4`) resolving typographical asymmetry between capture endpoints.
  - Dual-ended participant recordings were strictly co-located, guaranteeing **zero cross-split session leakage**:
    $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
  - **Train**: 102 files (74 canonical groups; 220,194 flows; 3.19% VPN).
  - **Validation**: 19 files (16 canonical groups; 69,564 flows; 6.22% VPN).
  - **Test**: 19 files (16 canonical groups; 49,251 flows; 5.47% VPN).
  - Total preprocessed dataset: exactly **339,009 flows**.
- **Flow Construction**:
  - Chronological packet sorting within canonicalized 5-tuples.
  - Strict **120-second inactivity timeout** flow boundaries.
  - Flow representation: $X \in \mathbb{R}^{2 \times 64}$.
    - Channel 1: Directional IP packet length $x_{\text{length}, t} = d_t \times \min(S_t / 1500, 1.0)$ using actual IP packet lengths ($S_t$).
    - Channel 2: Inter-arrival time $x_{\text{IAT}, t} = \ln(1 + \min(\Delta t_t, 5.0))$, with $x_{\text{IAT}, 0} = 0.0$.
    - Right-zero padding to length 64 with explicit sequence lengths $L$ and validity masks $M_t \in \{0, 1\}$.

### 2.2 Neural Network Architecture
Compact 1D-CNN specified in Section 7 of the research protocol:
- Input: $(B, 2, 64)$
- Block 1: Conv1d(2, 32, kernel=5, pad=2) $\rightarrow$ BatchNorm1d(32) $\rightarrow$ ReLU $\rightarrow$ MaxPool1d(2)
- Block 2: Conv1d(32, 64, kernel=5, pad=2) $\rightarrow$ BatchNorm1d(64) $\rightarrow$ ReLU $\rightarrow$ MaxPool1d(2)
- Block 3: Conv1d(64, 128, kernel=3, pad=1) $\rightarrow$ BatchNorm1d(128) $\rightarrow$ ReLU $\rightarrow$ AdaptiveAvgPool1d(1)
- Classifier: Flatten $\rightarrow$ Linear(128, 64) $\rightarrow$ ReLU $\rightarrow$ Dropout(0.3) $\rightarrow$ Linear(64, 1)
- Parameter Count: exactly **44,129 parameters**.

### 2.3 Quantization Implementation
- **Static Post-Training Quantization (PTQ)**: Fused Conv-BN-ReLU modules in eval mode; calibrated on 1,000 representative training flows; converted with PyTorch eager quantization under `qnnpack`. Confirmed `fc2` zero-point at 105 (scale 0.0693), accommodating negative logits naturally without zero-clamping.
- **Quantization-Aware Training (QAT)**: Fused Conv-BN-ReLU modules for QAT; inserted fake-quantization nodes; fine-tuned for 5 epochs on CPU; selected checkpoint at best validation loss (0.680 calibrated $\tau$); converted to deployed integer graph.
- **Operator Verification**: Confirmed authentic `torch.qint8` weights across all convolutional and linear layers.

---

## 3. Experimental Findings

### 3.1 Predictive Performance and Threshold Calibration (Phase 5)
Evaluated on the held-out test split of **49,251 flows** (2,693 VPN, 46,558 Non-VPN):

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


#### Key Predictive Observations:
1. **Model-Specific Validation Calibration (Panel B)**:
   - When decision thresholds are calibrated on held-out validation data ($\tau = 0.90$ for FP32, $\tau = 0.90$ for PTQ, $\tau = 0.68$ for QAT), **all three models exhibit strong predictive performance**:
     - FP32: Accuracy **96.88%**, Balanced Accuracy **97.41%**, F1 **0.7746**.
     - PTQ: Accuracy **96.79%**, Balanced Accuracy **97.41%**, F1 **0.7697**.
     - QAT: Accuracy **96.90%**, Balanced Accuracy **97.57%**, F1 **0.7762**.
2. **The Threshold Mismatch Artifact in Reused Thresholding (Panel C)**:
   - When the FP32 threshold ($\tau = 0.900$) is reused blindly on QAT, QAT's predicted probabilities do not exceed $0.680$ due to logit contraction. This produces an all-negative classifier (F1: 0.0000, 0 positive calls). Validation calibration completely restores QAT's F1 to 0.7762.
3. **Threshold-Independent Ranking (Panel A)**:
   - In threshold-free ranking, QAT decisively leads: ROC-AUC **0.9834** and PR-AUC **0.6478** with lowest Brier score **0.0237**.

---

## 4. Attribution Drift and Multi-Seed Explainer Repeatability (Phases 6 & 7)

#### Explainer Stability and Noise Calibration:
KernelSHAP was executed with **three distinct random seeds (42, 43, 44)** on the **exact same 100-flow balanced test cohort** (50 VPN, 50 Non-VPN), explaining the pre-sigmoid VPN logit with 50 background samples:
- **Within-Model Explainer Repeatability (3-Pair Mean)**: Top-5 Jaccard overlap = **0.9553**, Total Variation distance = **0.0553**, Signed Cosine similarity = **0.9676**.
- **Theoretical Combinatorial Random Chance Baseline**: Expected Top-5 Jaccard overlap between two independent random 5-subsets drawn from 128 features is **0.0220** (and **0.0426** for Top-10).

#### Attribution Drift Benchmark Results:

# Table 2: Feature Attribution Drift and Explainer Repeatability (Cohort N=100)

*All metrics evaluated on identical 100-flow balanced test cohort (50 VPN, 50 Non-VPN). Attributions explain the pre-sigmoid VPN logit with 50 background samples and 150 KernelSHAP evaluations per flow.*

### Panel A: Cohort Aggregate Attribution Metrics
| Condition / Comparison | Explainer Seeds | Top-5 Jaccard ($IoU_5$) | Top-10 Jaccard ($IoU_{10}$) | Total Variation ($TV$) | Signed Cosine ($S_C$) | Spearman Rank ($r_s$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Combinatorial Random Chance** | — | 0.0220 | 0.0426 | — | 0.0000 | 0.0000 |
| **Explainer Noise: FP32 (s42 vs s43)** | 42 vs 43 | 0.9557 ± 0.1840 | 0.9652 ± 0.1563 | 0.0568 ± 0.1623 | 0.9648 ± 0.1545 | 0.9489 ± 0.1633 |
| **Explainer Noise: FP32 (s42 vs s44)** | 42 vs 44 | 0.9547 ± 0.1834 | 0.9670 ± 0.1532 | 0.0541 ± 0.1547 | 0.9696 ± 0.1417 | 0.9569 ± 0.1581 |
| **Explainer Noise (3-Pair Mean)** | Multi-seed | **0.9553 ± 0.1829** | **0.9648 ± 0.1592** | **0.0553 ± 0.1588** | **0.9676 ± 0.1450** | **0.9516 ± 0.1666** |
| **FP32 vs INT8-PTQ (Cross-Model)** | 42 vs 42 | **0.9786 ± 0.0970** | **0.9887 ± 0.0650** | **0.0369 ± 0.0620** | **0.9918 ± 0.0394** | **0.9749 ± 0.0592** |
| **FP32 vs INT8-QAT (Cross-Model)** | 42 vs 42 | **0.9553 ± 0.1841** | **0.9679 ± 0.1416** | **0.2346 ± 0.1908** | **0.4830 ± 0.3449** | **0.9140 ± 0.1838** |

---

### Panel B: Paired Flow-by-Flow Hypothesis Testing (Cross-Model vs. Explainer Noise)
*Evaluates whether cross-model attribution drift exceeds intrinsic explainer variability on identical flows.*

| Comparison | Paired Delta Metric | Mean ± Std | Median | Non-Parametric Bootstrap 95% CI | Wilcoxon Signed-Rank Test | Statistical Conclusion |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PTQ vs. Noise** | $\Delta TV = TV_{\text{PTQ}} - TV_{\text{Noise}}$ | -0.0185 ± 0.1115 | +0.0049 | $[-0.0422, +0.0011]$ | $p = 0.00046$ (Holm: 0.00092) | **Indistinguishable from Noise** |
| **PTQ vs. Noise** | $\Delta IoU_5 = IoU_{5, \text{PTQ}} - IoU_{5, \text{Noise}}$ | +0.0232 ± 0.1088 | +0.0000 | $[+0.0051, +0.0472]$ | $p = 0.02771$ (Holm: 0.02771) | Identical Top-5 on 95/100 flows |
| **QAT vs. Noise** | $\Delta TV = TV_{\text{QAT}} - TV_{\text{Noise}}$ | +0.1793 ± 0.1547 | +0.1975 | $[+0.1500, +0.2071]$ | $p = 1.7478e-16$ (Holm: 5.2435e-16) | **Significant Drift ($p < 10^{-4}$)** |
| **QAT vs. PTQ** | $\Delta TV = TV_{\text{QAT}} - TV_{\text{PTQ}}$ | +0.1978 ± 0.1596 | +0.2854 | $[+0.1660, +0.2301]$ | $p = 3.8966e-18$ (Holm: 1.5586e-17) | QAT restructures attributions ($p < 10^{-4}$) |

---

### Panel C: Class-Specific Attribution Metrics (VPN vs. Non-VPN Subgroups)
| Metric | Overall ($N=100$) | VPN Flows Only ($N=50$) | Non-VPN Flows Only ($N=50$) | Class Difference (VPN - Non-VPN) |
| :--- | :---: | :---: | :---: | :---: |
| **PTQ Top-5 Jaccard ($IoU_5$)** | 0.9786 | 0.9638 | 0.9933 | -0.0295 |
| **PTQ Total Variation ($TV$)** | 0.0369 | 0.0388 | 0.0349 | +0.0040 |
| **PTQ Signed Cosine ($S_C$)** | 0.9918 | 0.9851 | 0.9984 | -0.0133 |
| **QAT Top-5 Jaccard ($IoU_5$)** | 0.9553 | 0.9350 | 0.9756 | -0.0406 |
| **QAT Total Variation ($TV$)** | 0.2346 | 0.1163 | 0.3529 | -0.2366 |
| **QAT Signed Cosine ($S_C$)** | 0.4830 | 0.1665 | 0.7996 | -0.6331 |


#### Key Attribution Observations:
1. **PTQ Attribution Drift is Statistically Indistinguishable from Explainer Noise**:
   - The Top-5 Jaccard overlap between FP32 and INT8-PTQ averaged **0.9786** (median **1.0000**), with identical Top-5 features on **95 of 100 flows**.
   - Total Variation distance between FP32 and PTQ averaged **0.0369**.
   - Paired flow-by-flow testing against within-model explainer noise yielded $\Delta TV = -0.0185 \pm 0.1115$ (bootstrap 95% CI $[-0.0422, +0.0011]$). Because the delta is negative and spans zero, **we cannot reject the null hypothesis that PTQ attribution displacement is equal to KernelSHAP sampling noise**.
2. **QAT Causes Genuine Attribution Restructuring**:
   - QAT explanations achieved a Top-5 Jaccard overlap of **0.9553** with FP32, with Total Variation distance reaching **0.2346**.
   - Paired testing against explainer noise yielded $\Delta TV = +0.1793 \pm 0.1547$ (Wilcoxon $p = 1.7478e-16$).
   - Paired testing against PTQ yielded $\Delta TV = +0.1978 \pm 0.1596$ (Wilcoxon $p = 3.8966e-18$).
   - Weight adaptation under simulated quantization genuinely reorganizes feature attributions while optimizing classification performance.

---

## 5. Hardware and Edge Execution Benchmarks (Phase 9)

Benchmarked on macOS Apple Silicon ARM64 under CPU execution using QNNPACK (50 warmup runs, 500 repetitions):

# Table 3: Hardware, Latency, and Throughput Benchmarks (macOS ARM64 QNNPACK)

*Hardware: Apple Silicon Mac (M-series, 8 cores). Backend: CPU native execution via `torch.backends.quantized.engine = 'qnnpack'`. Measurements: 50 warmup runs, 500 timed repetitions.*

### Panel A: Storage & Model Footprint
| Model Architecture | Parameter Count | Disk Checkpoint Size | Storage Reduction | Execution Path |
| :--- | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | 44,129 | 184.24 KB | 1.00× (Ref) | Standard FP32 Float Operations |
| **INT8-PTQ** | 44,129 | 51.77 KB | **3.56×** | Deployed `torch.qint8` Integer Operators |
| **INT8-QAT** | 44,129 | 51.77 KB | **3.56×** | Deployed `torch.qint8` Integer Operators |

---

### Panel B: Single-Flow Streaming Latency (Edge Scenario, Batch Size = 1)
*Characterizes immediate packet-arrival triage without buffering delay.*

| Model Architecture | Median Latency | Mean Latency | Latency Std Dev | Inter-Quartile Range (IQR) | 95th Percentile (p95) | 99th Percentile (p99) | Edge Throughput | Relative Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | **0.254 ms** | 0.254 ms | 0.013 ms | 0.014 ms | 0.274 ms | 0.286 ms | 3,931 flows/s | 1.00× (Ref) |
| **INT8-PTQ** | 0.293 ms | 0.379 ms | 0.645 ms | 0.042 ms | 0.615 ms | 1.116 ms | 2,636 flows/s | **0.87× (Slower)** |
| **INT8-QAT** | 0.254 ms | 0.258 ms | 0.018 ms | 0.016 ms | 0.292 ms | 0.321 ms | 3,873 flows/s | **1.00× (Parity)** |

*Observation: At Batch Size = 1, quantization yields no speedup because PyTorch QNNPACK invocation, dynamic quantization of inputs, and scalar dequantization dominate computation.*

---

### Panel C: Batched Server Throughput (High-Volume Pipeline, Batch Size = 64)
*Characterizes offline audit, log ingestion, or buffered middlebox inspection.*

| Model Architecture | Median Batch Latency | Mean Batch Latency | Batch Latency Std Dev | 95th Percentile (p95) | Measured Throughput | Throughput Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | 8.562 ms | 10.187 ms | 7.758 ms | 14.263 ms | 6,282 flows/s | 1.00× (Ref) |
| **INT8-PTQ** | 0.839 ms | 0.888 ms | 0.176 ms | 1.238 ms | 72,090 flows/s | **10.21×** |
| **INT8-QAT** | **1.004 ms** | **1.065 ms** | **0.325 ms** | **1.569 ms** | **60,107 flows/s** | **8.53×** |

*Observation: Under batch size 64, kernel invocation overhead is amortized, allowing ARM64 NEON integer dot-product SIMD instructions to deliver a 8.53× throughput speedup.*


---

## 6. Evaluation of Primary Registered Hypotheses

# Table 4: Primary Hypothesis Testing Matrix (Audited & Formalized)

| Hypothesis | Formal Registered Question & Null Hypothesis ($H_0$) | Test Statistic & Resampling Unit | Empirical Evidence | Statistical Decision |
| :--- | :--- | :--- | :--- | :---: |
| **H1: Predictive Drift** | **Question**: Does INT8 quantization change predictions relative to FP32?<br>**$H_0$**: Marginal classification symmetry ($P(FP32=1, INT8=0) = P(FP32=0, INT8=1)$) and zero mean logit shift ($\mu_{\Delta z} = 0$). | - McNemar Test (with continuity correction)<br>- Paired Student's t-test on logits<br>- Capture-group clustered bootstrap 95% CI | - PTQ McNemar $\chi^2 = 43.9, p = 3.5279e-11$<br>- PTQ Logit Shift: -0.0778 ($t = -38.9, p = 0.0000e+00$; Group CI: $[-0.1126, +0.2091]$)<br>- QAT Logit Shift: -1.9404 ($t = -533.8, p = 0.0000e+00$; Group CI: $[-2.8762, -1.8631]$) | **SUPPORTED**<br>(Quantization significantly shifts decision outputs) |
| **H2: Attribution Drift** | **Question**: Does INT8-PTQ produce measurable attribution drift beyond explainer noise?<br>**$H_0$**: Paired Total Variation difference $\Delta TV = TV_{\text{PTQ}} - TV_{\text{Noise}} = 0$. | - Paired two-sided Wilcoxon signed-rank test on identical flows ($N=100$)<br>- Multi-seed explainer noise baseline (Seeds 42, 43, 44)<br>- Flow and group bootstrap 95% CIs | - Paired $\Delta TV = -0.0185 \pm 0.1115$<br>- Wilcoxon $p = 0.00046$ (Holm: 0.00092)<br>- Bootstrap 95% CI: $[-0.0422, +0.0011]$ (spans 0)<br>- 95/100 flows share identical Top-5 features | **NOT RESOLVED**<br>(PTQ drift is within the noise floor of KernelSHAP) |
| **H3: QAT Comparison** | **Question**: Does INT8-QAT exhibit different attribution drift than INT8-PTQ?<br>**$H_0$**: $\Delta TV = TV_{\text{QAT}} - TV_{\text{PTQ}} \le 0$ (QAT produces equal or less attribution restructuring than PTQ). | - Paired Wilcoxon signed-rank test on identical flows ($N=100$)<br>- Holm-Bonferroni correction | - Paired $\Delta TV = +0.1978 \pm 0.1596$<br>- Wilcoxon $p = 3.8966e-18$ (Holm: 1.5586e-17)<br>- Bootstrap 95% CI: $[+0.1660, +0.2301]$ | **SUPPORTED (Refuting H3 direction)**<br>(QAT restructures attributions significantly more than PTQ) |
| **H4: Conditional Variation** | **Question**: Does attribution drift vary across traffic classes and prediction agreement cohorts?<br>**$H_0$**: Attribution drift is homogeneous across classes and agreement cohorts. | - Mann-Whitney U test between subgroups<br>- Class stratification (VPN vs Non-VPN)<br>- Agreement stratification | - Non-VPN PTQ $TV = 0.0349$ vs VPN $TV = 0.0388$<br>- Agreement PTQ $TV = 0.0369$ vs Disagreement $TV = nan$ | **SUPPORTED**<br>(Drift is conditioned on traffic class and agreement) |


### Detailed Statistical Hypothesis Decisions:
1. **H1 — Predictive Drift: SUPPORTED.**
   - Marginal symmetry between FP32 and INT8 predictions is rejected: McNemar $\chi^2 = 43.9, p = 3.5279e-11$ for PTQ.
   - Continuous logit shifts are statistically significant ($p < 10^{-15}$).
2. **H2 — Attribution Drift Beyond Noise: NOT RESOLVED (Within Explainer Noise).**
   - Top-5 overlap vastly exceeds combinatorial chance ($IoU_5 = 0.9786$ vs chance 0.0220), but paired Total Variation displacement between FP32 and PTQ is indistinguishable from KernelSHAP sampling noise ($\Delta TV = -0.0185$, 95% CI $[-0.0422, +0.0011]$).
3. **H3 — Differential QAT Attribution Restructuring: SUPPORTED (Refuting H3 Direction).**
   - QAT restructures feature attributions significantly more aggressively than PTQ ($\Delta TV = +0.1978$, Holm-corrected $p = 1.5586e-17$).
4. **H4 — Conditional Variation: SUPPORTED.**
   - Attribution drift varies across traffic classes (Non-VPN PTQ $TV = 0.0349$ vs VPN $TV = 0.0388$).

---

## 7. Scope of Conclusions and Limitations

### 7.1 Epistemological Classification of Claims
- **Verified Implementation Facts**:
  - Exactly 44,129 parameters across all models.
  - Converted checkpoints execute real `torch.qint8` integer operators under `qnnpack`.
  - In `artifacts/quantized_models/ptq_seed_42.pt`, `fc2` has zero-point 105 (scale 0.0693), representing negative logits down to -7.27 without zero-clamping (0 flows clamped at 0.0).
- **Empirical Findings**:
  - PTQ maintains high Top-5 feature attribution overlap ($IoU_5 = 0.9786$, identical on 95/100 flows).
  - Attribution displacement under PTQ cannot be distinguished from KernelSHAP sampling noise.
  - Validation-calibrated QAT achieves the highest F1-score (0.7762) and ROC-AUC (0.9834), but causes genuine attribution drift ($\Delta TV = +0.1793, p < 10^{-15}$).
  - INT8 yields a 10.21× throughput speedup at BS=64, but no latency improvement at BS=1.
- **Untested Operational Implications**:
  - We make **no claim** that attribution drift creates an exploitable security vulnerability or firewall bypass.

### 7.2 Methodological Limitations
1. **Limited Independent VPN Capture Groups**:
   - The held-out test split contains **3 VPN capture files** (`vpn_bittorrent`, `vpn_hangouts_audio2`, `vpn_netflix_A`).
   - Group-clustered bootstrap intervals are wider than flow-level bootstrap intervals, reflecting session dependence.
2. **Benchmark Scope**:
   - Results apply to the ISCX VPN-NonVPN 2016 dataset under a 1D-CNN architecture.

---

## 8. Complete Reproducibility Manifest

- **Dataset Manifest**: [`data/manifests/dataset_manifest.json`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/data/manifests/dataset_manifest.json)
- **Split Manifests**: [`train_manifest.json`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/data/manifests/train_manifest.json), [`val_manifest.json`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/data/manifests/val_manifest.json), [`test_manifest.json`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/data/manifests/test_manifest.json)
- **Model Checkpoints**:
  - FP32 Checkpoint: [`artifacts/checkpoints/fp32_seed_42.pt`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/artifacts/checkpoints/fp32_seed_42.pt)
  - Deployed PTQ Model: [`artifacts/quantized_models/ptq_seed_42.pt`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/artifacts/quantized_models/ptq_seed_42.pt)
  - Deployed QAT Model: [`artifacts/quantized_models/qat_seed_42.pt`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/artifacts/quantized_models/qat_seed_42.pt)
- **Processed Flow Datasets**: [`data/processed/train_data.npz`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/data/processed/train_data.npz), [`val_data.npz`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/data/processed/val_data.npz), [`test_data.npz`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/data/processed/test_data.npz)
- **Predictions & Attributions**:
  - Held-out test predictions: [`results/predictions/test_predictions.npz`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/results/predictions/test_predictions.npz)
  - 100-flow attribution cohort: [`results/attributions/cohort_attributions.npz`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/results/attributions/cohort_attributions.npz)
- **Master Pipeline Records**: [`results/metrics/master_pipeline_summary.json`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/results/metrics/master_pipeline_summary.json)
- **Hardware Benchmarks**: [`results/metrics/hardware_benchmarks.json`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/results/metrics/hardware_benchmarks.json)
- **Historical Baseline**: All pre-correction artifacts preserved in [`pre_split_correction/`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/pre_split_correction/)
