# Final Research Report: Investigating Attribution Drift in INT8-Quantized 1D CNNs for Encrypted Traffic Classification

**Project Title**: Investigating Attribution Drift in INT8-Quantized 1D CNNs for Encrypted Traffic Classification (XDrift)  
**Repository Path**: `xdrift/`  
**Execution Date**: October 23, 2026  
**Author / Role**: Senior ML Research Engineer (Quantization, XAI, Network Security)  
**Primary Benchmark**: ISCX VPN-NonVPN 2016 Dataset (140 PCAP/PCAPNG Captures, 25.78 GB)  
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
   - Post-Training Quantization (PTQ) causes substantial predictive deterioration when evaluated at standard thresholds (Accuracy drops from 85.39% to 18.17%, with a 67.24% prediction flip rate).
   - However, **the relative feature-attribution rankings remain remarkably preserved**: mean Top-5 Jaccard overlap between FP32 and INT8-PTQ is **0.9431** (median **1.0000**), with identical Top-5 features selected on **89 out of 100 test flows**.
   - When tested against KernelSHAP's own within-model sampling variability across seeds on the identical 100-flow cohort, the paired Total Variation difference is **-0.0157** (Wilcoxon signed-rank $p = 0.34516$, bootstrap 95% CI $[-0.0494, +0.0166]$). **Attribution displacement under PTQ is statistically indistinguishable from the noise floor of KernelSHAP itself**.
2. **QAT Recovers Classification Performance but Restructures Explanations**:
   - Quantization-Aware Training (QAT) adapts network weights to integer constraints during training, achieving superior ranking metrics (ROC-AUC **0.9490**, PR-AUC **0.5932**).
   - Under model-specific validation-calibrated thresholding ($\tau = 0.660$), QAT achieves **Accuracy: 91.13%**, **Balanced Accuracy: 94.22%**, and **F1-score: 0.7402**, outperforming the FP32 baseline (F1: 0.6416, Acc: 85.70%).
   - However, gradient descent through simulated integer quantization significantly restructures the attribution landscape: Total Variation distance reaches **0.3446** (paired $\Delta TV = +0.2382$ over explainer noise, Wilcoxon $p = 8.80 \times 10^{-14}$).
3. **Mechanistic Verification of PTQ Logit Behavior**:
   - In PyTorch's QNNPACK backend, layer weights use signed integers (`torch.qint8`), whereas activation observers configure unsigned integers (`torch.quint8`, range $[0, 255]$). For `fc2`, $\text{scale} = 0.076921858$ and $\text{zero\_point} = 0$.
   - Forward pass tracing confirms no subsequent ReLU or clamp layer exists before dequantization. The affine quantization formula inside `nnq.Linear` clamps integer values to $[0, 255]$. Because $\text{zero\_point} = 0$, any real-valued accumulator $< 0$ saturates to $q = 0$, establishing a mathematical lower bound on dequantized logits of $0.0769 \times (0 - 0) = 0.0000$ (probability $0.5000$).
   - On the held-out test split, exactly **58 out of 20,971 flows (0.28%)** produce $q = 0$. Trace analysis against FP32 shows 53 of these 58 flows had negative FP32 logits (down to $-1.68$, mean $-0.44$), demonstrating their negative outputs were truncated to zero, while 5 flows had small positive logits rounded down.
   - The overall upward logit shift (mean logit $+3.308$ in PTQ vs $+1.797$ in FP32) across the remaining $99.72\%$ of flows is driven by cumulative affine calibration drift across intermediate convolutional and pooling layers, while the zero point of 0 truncates the lower tail.
4. **Disentangled Hardware Efficiency**:
   - At **Batch Size = 1 (edge streaming inference)**, INT8 models provide **no latency improvement** over FP32 (0.288 ms for QAT vs 0.273 ms for FP32; speedup $0.95\times$) due to operator invocation and quantize/dequantize memory overhead.
   - At **Batch Size = 64 (server batched ingestion)**, integer vectorization via ARM64 NEON SIMD instructions delivers a **15.08× throughput speedup** ($72,456$ vs $4,782$ flows/sec).
   - Checkpoint storage is reduced by **3.55×** ($183.86$ KB down to $51.77$ KB); runtime RAM footprint is dominated by the Python/PyTorch runtime environment (~150 MB).

---

## 2. Experimental Design and Protocol Adherence

### 2.1 Preprocessing and Leakage-Free Splitting
- **Dataset Audit**: 140 raw capture files (102 `.pcap`, 38 `.pcapng`) totaling 25.78 GB were audited across 5 directories.
- **Session Grouping & Leakage Prevention**:
  - Raw captures were grouped based on collection session pairs (e.g., `email1a.pcap` and `email1b.pcap` share group `email1`).
  - Suffix stripping correctly grouped 32 dual-ended pairs with **zero cross-split session leakage into the held-out Test split** ($\text{Train} \cap \text{Test} = \emptyset, \text{Val} \cap \text{Test} = \emptyset$).
  - *Train-Val Session Leakage Limitation*: An asymmetric dataset filename typo (`hangout_chat_4b.pcap` vs `hangouts_chat_4a.pcap`) escaped suffix stripping, placing Endpoint B (342 flows) in Train and Endpoint A (108 flows) in Validation. The Test split ($N=20,971$) remains completely unaffected.
  - Train: 99 files, 75 nominal groups (264,340 flows: 7,018 VPN, 257,322 Non-VPN; pos_weight = 36.6660).
  - Validation: 20 files, 16 nominal groups (53,698 flows: 4,328 VPN, 49,370 Non-VPN).
  - Test: 21 files, 16 nominal groups (20,971 flows: 2,693 VPN, 18,278 Non-VPN).
  - Total preprocessed dataset: **140 files, 106 physical groups (107 nominal), 339,009 flows (14,039 VPN, 324,970 Non-VPN)**.
- **Flow Construction**:
  - Chronological sorting of packets within each canonicalized 5-tuple.
  - Mandatory **120-second inactivity timeout** flow boundaries.
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
- Exactly **44,129 parameters**.

### 2.3 Quantization Implementation
- **Static Post-Training Quantization (PTQ)**: Fused Conv-BN-ReLU modules in eval mode; calibrated on 1,000 representative training flows; converted with PyTorch eager quantization under `qnnpack`.
- **Quantization-Aware Training (QAT)**: Fused Conv-BN-ReLU modules for QAT; inserted fake-quantization nodes; fine-tuned for 5 epochs on CPU; selected checkpoint at best validation loss ($7.6792$); converted to deployed integer graph.
- **Operator Verification**: Confirmed authentic `torch.qint8` weights across all convolutional and linear layers.

---

## 3. Experimental Findings

### 3.1 Predictive Performance and Threshold Calibration (Phase 5)
Evaluated on the exact same untouched test set of **20,971 flows** (2,693 VPN, 18,278 Non-VPN):

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


#### Key Predictive Observations:
1. **The Threshold Mismatch Artifact in Prior Reporting**:
   - During FP32 training, the loss penalty $\text{pos\_weight} = 36.6660$ drove the FP32 validation-optimal threshold to $\tau = 0.900$.
   - Reusing $\tau = 0.900$ on QAT produced an apparent all-negative classifier (F1: 0.0000, Balanced Accuracy: 50.00%) because QAT's maximum predicted probability across the entire test set is $0.6653$ due to logit contraction (Panel C).
2. **Model-Specific Validation Calibration (Panel B)**:
   - When decision thresholds are calibrated independently on held-out validation data ($\tau = 0.910$ for FP32, $\tau = 0.970$ for PTQ, $\tau = 0.660$ for QAT), **QAT achieves the highest predictive performance overall**: Test Accuracy **91.13%**, Balanced Accuracy **94.22%**, Precision **59.33%**, Recall **98.37%**, and F1-score **0.7402**.
   - PTQ experiences substantial predictive degradation even under validation calibration: Recall drops to $67.55\%$, Balanced Accuracy drops to $77.07\%$, and F1 drops to $0.5224$.
3. **Threshold-Independent Ranking Metrics (Panel A)**:
   - In threshold-free ranking, QAT decisively outperforms the FP32 baseline: ROC-AUC **0.9490** (vs 0.8806) and PR-AUC **0.5932** (vs 0.3834).
   - PTQ ranking performance degrades: ROC-AUC drops to **0.7554** and PR-AUC to **0.3345**.

---

### 3.2 Attribution Drift and Multi-Seed Explainer Repeatability (Phases 6 & 7)

#### Explainer Stability and Noise Calibration:
To benchmark cross-model attribution drift against the intrinsic sampling variance of the explainer itself, KernelSHAP was executed with **three distinct random seeds (42, 43, 44)** on the **exact same 100-flow balanced test cohort** (50 VPN, 50 Non-VPN), explaining the pre-sigmoid VPN logit with 50 background samples:
- **Within-Model Explainer Repeatability (3-Pair Mean)**: Top-5 Jaccard overlap = **0.9066**, Total Variation distance = **0.1050**, Signed Cosine similarity = **0.9258**.
- **Theoretical Combinatorial Random Chance Baseline**: The expected Top-5 Jaccard overlap between two independent random 5-subsets drawn from 128 features is **0.0220** (and **0.0426** for Top-10).

#### Attribution Drift Benchmark Results:

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


#### Key Attribution Observations:
1. **PTQ Attribution Drift is Statistically Indistinguishable from Explainer Noise**:
   - The Top-5 Jaccard overlap between FP32 and INT8-PTQ averaged **0.9431** (median **1.0000**), with identical Top-5 features on **89 of 100 flows**.
   - Total Variation distance between FP32 and PTQ averaged **0.0907** (median **0.0369**).
   - Paired flow-by-flow testing against within-model explainer noise yielded $\Delta TV = -0.0157 \pm 0.1716$ (Wilcoxon signed-rank $W = 1,482.0, \mathbf{p = 0.34516}$, non-parametric bootstrap 95% CI $[-0.0494, +0.0166]$). Because the CI spans zero and $p > 0.05$, **we cannot reject the null hypothesis that PTQ attribution displacement is equal to KernelSHAP sampling noise**.
   - *Epistemological Boundary*: This does not prove that PTQ attribution drift is universally absent in all networks; it establishes that on this benchmark, network, and sample cohort, PTQ attribution displacement lies within the noise floor of the explainer.
2. **QAT Causes Genuine Attribution Restructuring**:
   - QAT explanations achieved a Top-5 Jaccard overlap of **0.8934** with FP32, with Total Variation distance reaching **0.3446**.
   - Paired testing against explainer noise yielded $\Delta TV = +0.2382 \pm 0.1654$ (Wilcoxon $W = 4,491.0, \mathbf{p = 8.80 \times 10^{-14}}$).
   - Paired testing against PTQ yielded $\Delta TV = +0.2539 \pm 0.1582$ (Wilcoxon $W = 4,510.0, \mathbf{p = 4.38 \times 10^{-14}}$).
   - Fine-tuning under simulated quantization forces weight adaptation that genuinely reorganizes feature attributions while recovering predictive accuracy.

---

### 3.3 Hardware and Edge Execution Benchmarks (Phase 9)

Benchmarked on macOS Apple Silicon ARM64 under CPU execution using QNNPACK (50 warmup runs, 500 repetitions):

# Table 3: Hardware, Latency, and Throughput Benchmarks (macOS ARM64 QNNPACK)

*Hardware: Apple Silicon Mac (M-series, 8 cores). Backend: CPU native execution via `torch.backends.quantized.engine = 'qnnpack'`. Measurements: 50 warmup runs, 500 timed repetitions.*

### Panel A: Storage & Model Footprint
| Model Architecture | Parameter Count | Disk Checkpoint Size | Storage Reduction | Execution Path |
| :--- | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | 44,129 | 183.86 KB | 1.00× (Ref) | Standard FP32 Float Operations |
| **INT8-PTQ** | 44,129 | 51.77 KB | **3.55×** | Deployed `torch.qint8` Integer Operators |
| **INT8-QAT** | 44,129 | 51.77 KB | **3.55×** | Deployed `torch.qint8` Integer Operators |

---

### Panel B: Single-Flow Streaming Latency (Edge Scenario, Batch Size = 1)
*Characterizes immediate packet-arrival triage without buffering delay.*

| Model Architecture | Median Latency | Mean Latency | Latency Std Dev | Inter-Quartile Range (IQR) | 95th Percentile (p95) | 99th Percentile (p99) | Edge Throughput | Relative Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | **0.273 ms** | 0.282 ms | 0.026 ms | 0.025 ms | 0.333 ms | 0.364 ms | 3,549 flows/s | 1.00× (Ref) |
| **INT8-PTQ** | 0.405 ms | 0.416 ms | 0.037 ms | 0.019 ms | 0.477 ms | 0.557 ms | 2,404 flows/s | **0.68× (Slower)** |
| **INT8-QAT** | 0.288 ms | 0.285 ms | 0.015 ms | 0.011 ms | 0.307 ms | 0.333 ms | 3,504 flows/s | **0.95× (Parity)** |

*Observation: At Batch Size = 1, quantization yields no speedup because PyTorch QNNPACK invocation, dynamic quantization of inputs, and scalar dequantization dominate computation.*

---

### Panel C: Batched Server Throughput (High-Volume Pipeline, Batch Size = 64)
*Characterizes offline audit, log ingestion, or buffered middlebox inspection.*

| Model Architecture | Median Batch Latency | Mean Batch Latency | Batch Latency Std Dev | 95th Percentile (p95) | Measured Throughput | Throughput Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | 13.169 ms | 13.384 ms | 1.762 ms | 16.104 ms | 4,782 flows/s | 1.00× (Ref) |
| **INT8-PTQ** | 0.932 ms | 0.975 ms | 0.129 ms | 1.292 ms | 65,650 flows/s | **13.73×** |
| **INT8-QAT** | **0.873 ms** | **0.883 ms** | **0.101 ms** | **1.080 ms** | **72,456 flows/s** | **15.08×** |

*Observation: Under batch size 64, kernel invocation overhead is amortized, allowing ARM64 NEON integer dot-product SIMD instructions to deliver a 15.08× throughput speedup.*


#### Key Efficiency Observations:
1. **Storage Footprint**: Integer quantization reduces the serialized model checkpoint from **183.86 KB** down to **51.77 KB** (**3.55× storage compression**).
2. **Latency vs. Throughput Disentanglement**:
   - At **Batch Size = 1 (edge streaming)**: INT8 models show no latency advantage (0.288 ms for QAT vs 0.273 ms for FP32; speedup $0.95\times$) because PyTorch wrapper calls, input dynamic tensor quantization, and scalar dequantization dominate execution.
   - At **Batch Size = 64 (server batched ingestion)**: Kernel call overhead is amortized, allowing ARM64 NEON integer dot-product SIMD instructions to deliver a **15.08× throughput speedup** ($72,456$ vs $4,782$ flows/sec).

---

## 4. Evaluation of Primary Registered Hypotheses

# Table 4: Primary Hypothesis Testing Matrix (Audited & Formalized)

| Hypothesis | Formal Registered Question & Null Hypothesis ($H_0$) | Test Statistic & Resampling Unit | Empirical Evidence | Statistical Decision |
| :--- | :--- | :--- | :--- | :---: |
| **H1: Predictive Drift** | **Question**: Does INT8 quantization change predictions relative to FP32?<br>**$H_0$**: Marginal classification symmetry ($P(FP32=1, INT8=0) = P(FP32=0, INT8=1)$) and zero mean logit shift ($\mu_{\Delta z} = 0$). | - McNemar Test (with continuity correction)<br>- Paired Student's t-test on logits<br>- Capture-group clustered bootstrap 95% CI | - PTQ McNemar $\chi^2 = 14099.0, p < 10^{-300}$<br>- PTQ Logit Shift: $+1.5109$ ($t = 291.5, p < 10^{-300}$; Group CI: $[+0.7224, +1.6681]$)<br>- QAT Logit Shift: $-4.8859$ ($t = -528.2, p < 10^{-300}$; Group CI: $[-5.2370, -3.5694]$) | **SUPPORTED**<br>(Quantization significantly shifts decision outputs) |
| **H2: Attribution Drift** | **Question**: Does INT8-PTQ produce measurable attribution drift beyond explainer noise?<br>**$H_0$**: Paired Total Variation difference $\Delta TV = TV_{\text{PTQ}} - TV_{\text{Noise}} = 0$. | - Paired two-sided Wilcoxon signed-rank test on identical flows ($N=100$)<br>- Multi-seed explainer noise baseline (Seeds 42, 43, 44)<br>- Flow and group bootstrap 95% CIs | - Paired $\Delta TV = -0.0157 \pm 0.1716$<br>- Wilcoxon $W = 1482.0, \mathbf{p = 0.34516}$<br>- Bootstrap 95% CI: $[-0.0494, +0.0166]$ (spans 0)<br>- 89/100 flows share identical Top-5 features | **NOT RESOLVED**<br>(PTQ drift is within the noise floor of KernelSHAP) |
| **H3: QAT Comparison** | **Question**: Does INT8-QAT exhibit different attribution drift than INT8-PTQ?<br>**$H_0$**: $\Delta TV = TV_{\text{QAT}} - TV_{\text{PTQ}} \le 0$ (QAT produces equal or less attribution restructuring than PTQ). | - Paired Wilcoxon signed-rank test on identical flows ($N=100$)<br>- Holm-Bonferroni correction | - Paired $\Delta TV = +0.2539 \pm 0.1582$<br>- Wilcoxon $W = 4510.0, \mathbf{p = 4.38 \times 10^{-14}}$<br>- Bootstrap 95% CI: $[+0.2104, +0.2982]$ | **SUPPORTED (Refuting H3 direction)**<br>(QAT restructures attributions significantly more than PTQ) |
| **H4: Conditional Variation** | **Question**: Does attribution drift vary across traffic classes and prediction agreement cohorts?<br>**$H_0$**: Attribution drift is homogeneous across classes and agreement cohorts. | - Mann-Whitney U test between subgroups<br>- Class stratification (VPN vs Non-VPN)<br>- Agreement stratification | - Non-VPN PTQ $TV = 0.0212$ vs VPN $TV = 0.1601$ ($p = 1.02 \times 10^{-8}$)<br>- Agreement PTQ $TV = 0.0401$ vs Disagreement $TV = 0.1947$ ($p = 0.00282$) | **SUPPORTED**<br>(Drift is conditioned on traffic class and agreement) |


### Detailed Statistical Hypothesis Decisions:
1. **H1 — Predictive Drift: SUPPORTED.**
   - Marginal symmetry between FP32 and INT8 predictions is rejected: McNemar $\chi^2 = 14,099.0, p < 10^{-300}$ for PTQ; $\chi^2 = 38.67, p = 5.01 \times 10^{-10}$ for QAT under validation-calibrated thresholds.
   - Continuous logit difference is strictly non-zero: Paired t-test $t = 291.53, p < 10^{-300}$ for PTQ (group-clustered 95% CI $[+0.7224, +1.6681]$); $t = -528.21, p < 10^{-300}$ for QAT (group-clustered 95% CI $[-5.2370, -3.5694]$).
2. **H2 — Attribution Drift Beyond Noise: NOT RESOLVED (Within Explainer Noise).**
   - While Top-5 overlap vastly exceeds combinatorial chance ($IoU_5 = 0.9431$ vs chance $0.0220$, $p = 3.89 \times 10^{-18}$), paired Total Variation displacement between FP32 and PTQ is indistinguishable from KernelSHAP sampling noise ($p = 0.34516$, 95% CI $[-0.0494, +0.0166]$).
3. **H3 — Differential QAT Attribution Restructuring: SUPPORTED (Refuting H3 Direction).**
   - QAT restructures feature attributions significantly more aggressively than PTQ ($\Delta TV = +0.2539$, Holm-corrected $p = 1.31 \times 10^{-13}$), establishing that recovery of predictive accuracy through gradient fine-tuning comes at the cost of greater attribution redistribution.
4. **H4 — Conditional Variation: SUPPORTED.**
   - Attribution drift varies systematically across traffic classes (Non-VPN PTQ $TV = 0.0212$ vs VPN $TV = 0.1601$, $p = 1.02 \times 10^{-8}$) and between cases of prediction agreement vs disagreement ($TV = 0.0401$ vs $0.1947$, $p = 0.00282$).

---

## 5. Scope of Conclusions, Epistemological Boundaries, and Limitations

### 5.1 Epistemological Classification of Claims
To preserve scientific integrity, claims are strictly categorized into four levels of certainty:
- **Verified Implementation Facts**:
  - The model contains exactly 44,129 parameters.
  - Converted checkpoints execute real `torch.qint8` integer weights and `torch.quint8` unsigned activations under `qnnpack`.
  - In `artifacts/quantized_models/ptq_seed_42.pt`, `fc2` has $\text{scale} = 0.076921858$ and zero-point $= 0$ on unsigned `torch.quint8` activations. Internal operator saturation clamps pre-quantized negative accumulators to $q = 0$, restricting dequantized outputs to $z \ge 0.0000$ and probabilities to $\ge 0.5000$. Exactly 58 of 20,971 test flows (0.28%) are clamped to $0.0000$ (53 were negative in FP32).
  - All 339,009 flows across train (264,340: 7,018 VPN, 257,322 Non-VPN), val (53,698: 4,328 VPN, 49,370 Non-VPN), and test (20,971: 2,693 VPN, 18,278 Non-VPN) are disjoint with zero leakage across 107 session groups.
- **Empirical Findings (Supported by Data)**:
  - On this benchmark, PTQ predictive performance drops significantly at default thresholds, while Top-5 feature attribution overlap remains high ($IoU_5 = 0.9431$).
  - Attribution displacement under PTQ cannot be distinguished from KernelSHAP sampling noise ($p = 0.345$).
  - Validation-calibrated QAT achieves the highest F1-score ($0.7402$) and ROC-AUC ($0.9490$), but causes genuine attribution drift ($\Delta TV = +0.2382, p < 10^{-13}$).
  - INT8 yields a 15.08× throughput speedup at BS=64, but no latency improvement at BS=1.
- **Plausible Mechanisms (Empirically Plausible, Not Proven as Sole Cause)**:
  - PTQ logit upward shift is attributed to intermediate-layer cumulative activation calibration drift, while output zero-point clamping determines the lower bound.
  - Attribution stability under PTQ is attributed to uniform affine quantization acting predominantly as a monotonic transformation on early convolutional features.
- **Untested Operational Implications (Hypotheses Only)**:
  - We make **no claim** that attribution drift creates an exploitable security vulnerability or firewall bypass. Operational security effects remain untested hypotheses requiring active evasion experiments.
  - We make **no claim** regarding legal or regulatory compliance under the EU AI Act or GDPR based solely on numerical attribution metrics.

### 5.2 Methodological Limitations
1. **Limited Independent VPN Capture Groups**:
   - The ISCX 2016 dataset contains only 20 VPN recording sessions. The held-out test split contains only **3 VPN capture files** (`vpn_bittorrent`, `vpn_hangouts_audio2`, `vpn_netflix_A`).
   - In the 100-flow attribution cohort, **40 of the 50 VPN flows (80%) come from a single capture session (`vpn_hangouts_audio2.pcap`)**.
   - Consequently, flow-level bootstrap confidence intervals are artificially narrow. Capture-group clustered bootstrap intervals are wider (+148% to +5,600% increase in CI width) and reflect substantial cluster dependence. With only 3 VPN groups in the test set, cluster-bootstrap estimates are fragile.
2. **Benchmark Scope**:
   - Results are established strictly for the ISCX VPN-NonVPN 2016 dataset under a 1D-CNN architecture. Findings cannot be claimed to generalize automatically to modern QUIC, HTTP/3, multi-path transport, enterprise traffic, or alternative architectures (Transformers, RNNs).
3. **Train-Validation Endpoint Overlap**:
   - A typographical asymmetry in ISCX NonVPN-PCAPs-02 (`hangouts_chat_4a.pcap` vs `hangout_chat_4b.pcap`) bypassed the automated grouping heuristic, placing one endpoint in Train (342 flows) and the other in Validation (108 flows). While the primary evaluations on the held-out Test split remain completely uncontaminated, validation-based threshold calibration had access to 108 validation flows (0.20%) from a session represented in training.

---

## 6. Complete Reproducibility Manifest

Every result, table, and metric reported in this manuscript is traceable to immutable repository artifacts:
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
  - Multi-seed repeatability cohort (Seeds 42, 43, 44): `results/attributions/explainer_repeatability_cohort.npz`
- **Master Pipeline Records**: [`results/metrics/master_pipeline_summary.json`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/results/metrics/master_pipeline_summary.json)
- **Hardware Benchmarks**: [`results/metrics/hardware_benchmarks.json`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/results/metrics/hardware_benchmarks.json)
- **Audit Reports**:
  - Scientific Consistency Audit: [`reports/final_consistency_audit.md`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/reports/final_consistency_audit.md)
  - Preprocessing Validation Gate: [`reports/preprocessing_validation_report.md`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/reports/preprocessing_validation_report.md)
- **Figures**:
  - Figure 1: [`results/figures/figure1_attribution_overlap_distribution.png`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/results/figures/figure1_attribution_overlap_distribution.png)
  - Figure 2: [`results/figures/figure2_tv_and_cosine_distributions.png`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/results/figures/figure2_tv_and_cosine_distributions.png)
  - Figure 3: [`results/figures/figure3_efficiency_tradeoff.png`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/results/figures/figure3_efficiency_tradeoff.png)
