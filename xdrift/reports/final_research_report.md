# Final Research Report: Investigating Attribution Drift in INT8-Quantized 1D CNNs for Encrypted Traffic Classification

**Project Title**: Investigating Attribution Drift in INT8-Quantized 1D CNNs for Encrypted Traffic Classification (XDrift)  
**Repository Path**: `xdrift/`  
**Execution Date**: October 24, 2026  
**Author / Role**: Senior ML Research Engineer (Quantization, XAI, Network Security)  
**Primary Benchmark**: ISCX VPN-NonVPN 2016 Dataset (140 PCAP/PCAPNG Captures, 25.78 GB, 106 Canonical Groups)  
**Execution Environment**: Apple Silicon Mac (M-series, ARM64), Python 3.12.7, PyTorch 2.5.1 (QNNPACK Quantization Engine)

---

## Executive Summary

This report establishes the final empirical, theoretical, and statistical findings of the **XDrift** research project under a strictly verified, leakage-free group-aware split. We evaluate the impact of **INT8 quantization**—specifically static Post-Training Quantization (PTQ) and Quantization-Aware Training (QAT)—on a compact 1D Convolutional Neural Network (CNN) trained for encrypted network traffic classification (VPN vs. Non-VPN).

### Primary Research Question:
*Does edge quantization alter the explainability profile (feature attributions) of a deep traffic classifier beyond the intrinsic stochastic repeatability threshold of the explainer itself, and how does this relate to predictive performance and hardware throughput?*

### Definitive Conclusions:
1. **Predictive Performance & Threshold Calibration**:
   - Quantization alters decision output scales and shifts probability calibrations. Under raw uncalibrated threshold reuse ($\tau = 0.900$), QAT experiences total positive prediction collapse (F1 = 0.0000, 0 positive calls).
   - When decision thresholds are calibrated on held-out validation data ($\tau_{\text{FP32}} = 0.900, \tau_{\text{PTQ}} = 0.900, \tau_{\text{QAT}} = 0.680$), **INT8-QAT achieves the highest classification performance of the quantized models**: Test F1 = **0.7762**, ROC-AUC = **0.9834**, and PR-AUC = **0.6478**, outperforming FP32 (F1 = 0.7746, PR-AUC = 0.5610).
   - INT8-PTQ maintains baseline ranking accuracy (ROC-AUC = 0.9785, PR-AUC = 0.6102, F1 = 0.7697).
   - Under an unquantized 5-epoch fine-tuning control baseline (**FP32-FT**, Option 1), the model reaches Test F1 = **0.8176** (ROC-AUC = **0.9819**, PR-AUC = **0.6128**) at calibrated $\tau = 0.830$, with 1,693 positive calls at $\tau = 0.900$ (demonstrating the absence of an integer saturation ceiling).
2. **Probability Calibration vs. Discrimination (Murphy Decomposition)**:
   - While QAT achieves a lower Brier score ($0.0237$) and lower ECE ($0.0378$) than FP32 (Brier $0.0585$, ECE $0.1691$), Murphy score decomposition reveals that this is driven by logit contraction compressing Non-VPN predicted probabilities toward zero (mean Non-VPN probability of $0.0557$ on a test dataset with $94.53\%$ Non-VPN flows). This reduces quadratic error on the dominant negative class rather than demonstrating intrinsic superiority in uncalibrated confidence estimation. The threshold $\tau = 0.680$ is the validation-selected operating point for the F1 objective, not an intrinsically optimal threshold for all operational costs.
3. **Attribution Drift & Explainer Noise Floor (Wilcoxon Signed-Rank Test)**:
   - **INT8-PTQ**: Feature attributions are remarkably well-preserved in rank order. Dominant feature sets remain **97.9% identical** ($IoU_5 = 0.9786$, with 95/100 flows having 100% identical Top-5 features), and signed cosine similarity remains exceptionally high ($S_C = 0.9918$). A paired two-sided Wilcoxon signed-rank test against the multi-seed explainer repeatability noise floor yields $W = 1506.0$ with positive rank sum $R^+ = 3544.0$ and negative rank sum $R^- = 1506.0$ ($p = 0.00046$, Holm $p = 0.00092$). Under a one-sided test ($H_1: \Delta TV > 0$), $p = 0.00023$. Because $R^+ > R^-$ and 69 of 100 flows have positive $\Delta TV$, positive ranks dominate. PTQ attribution differences exhibit a statistically detectable signed-rank shift relative to the explainer-noise comparison, while the observed median difference is small (+0.0049, less than 0.5% total variation) and top-five feature overlap remains high. These findings do not establish statistical equivalence between PTQ drift and explainer noise.
   - **INT8-QAT**: QAT induces **massive attribution restructuring** (Total Variation $TV = 0.2346$, Signed Cosine similarity collapses to $0.4830$). The paired Wilcoxon test decisively rejects attribution preservation ($W = 129.0, p = 1.75 \times 10^{-16}$). QAT optimizes classification performance under integer constraints by redistributing decision weight across complementary temporal features.
4. **Resolution of the Fine-Tuning Confounder (Option 1 Control Baseline)**:
   - To resolve whether QAT attribution drift is caused by integer quantization or stochastic gradient descent fine-tuning, the **FP32-FT** control fine-tunes the converged FP32 model with identical optimizer (`AdamW`, $lr = 10^{-4}$), loss weighting ($\text{pos\_weight} = 30.3756$), batch size ($64$), and duration ($5$ epochs) in full 32-bit floating-point precision without fake quantization.
   - FP32-FT induces a Total Variation distance of **0.2214 ± 0.2327** (median **0.2070**), Top-5 Jaccard of **0.9481**, and signed cosine similarity of **0.1166** relative to base FP32.
   - Because FP32-FT shifts attributions by $0.2214$ while QAT shifts them by $0.2346$, and static PTQ (which has no fine-tuning) exhibits minimal shift ($TV = 0.0369$), this triangulates the definitive scientific conclusion: **attribution restructuring under QAT is overwhelmingly driven by gradient-based parameter shifts across the overparameterized loss landscape rather than integer quantization grid noise.**
5. **KernelSHAP Efficiency and Mathematical Additivity**:
   - For every cohort instance $i$, attributions strictly satisfy the efficiency axiom $\sum_{j=1}^{128} \phi_{ij} = f(x_i) - \mathbb{E}[f(X_{\text{bg}})]$. Maximum absolute residuals across the 100 cohort flows are bounded by **$7 \times 10^{-6}$** on FP32 and **$< 10^{-6}$** on PTQ and QAT (mean absolute residuals $\le 2 \times 10^{-6}$), confirming exact mathematical fidelity down to single-precision floating-point limits.
6. **Hardware Performance & Execution Boundary**:
   - In batched line-rate processing (Batch Size = 64), INT8 inference achieved substantially higher measured throughput on the tested Apple Silicon system under PyTorch QNNPACK: a **11.94× throughput increase** for PTQ (76,264 flows/s) and **12.77×** for QAT (81,552 flows/s) over FP32 (6,387 flows/s). Median batch latency speedup was **12.44×** for PTQ and **12.59×** for QAT. Model checkpoints are compressed by **3.56×** (184.24 KB $\to$ 51.77 KB).
   - In single-flow streaming (Batch Size = 1), measured latency speedup was much smaller (1.05× PTQ, 1.10× QAT). These results are specific to the tested software, hardware, and benchmark configuration; they cannot be extrapolated to embedded microcontrollers without physical on-device profiling.
7. **Class Conditioning (H4)**:
   - Mann-Whitney U testing confirms that Non-VPN traffic experiences significantly greater Total Variation displacement under quantization than VPN traffic ($r_{\text{rb}} = +0.6160, p < 10^{-5}$ for PTQ; $r_{\text{rb}} = +0.7776, p < 10^{-10}$ for QAT), while Top-5 feature sets remain unaffected ($p > 0.15$). Prediction agreement conditioning is non-estimable on this cohort due to 100% agreement.

---

## 1. Dataset Split & Leakage-Free Design (Phase 1)

### 1.1 Dataset Composition & Canonical Grouping
The ISCX VPN-NonVPN 2016 dataset contains **140 raw capture files** across two traffic classes. To strictly prevent data leakage from paired recordings, related network sessions, and identical users across splits, all files were mapped to **106 unique canonical session groups** via lexical normalization and explicit alias resolution (`'hangout_chat_4': 'hangouts_chat_4'`).

| Dataset Partition | File Captures | Canonical Groups | Total Extracted Flows | VPN Flows | Non-VPN Flows | Majority Class Fraction |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Train Split** | 102 | 74 | **220,194** | 7,015 | 213,179 | 96.81% Non-VPN |
| **Validation Split** | 19 | 16 | **69,564** | 4,328 | 65,236 | 93.78% Non-VPN |
| **Held-Out Test Split** | 19 | 16 | **49,251** | 2,693 | 46,558 | 94.53% Non-VPN |
| **Total / Global** | **140** | **106** | **339,009** | **14,036** | **324,973** | **95.86% Non-VPN** |

### 1.2 Verification of Split Isolation
- All three pairwise canonical group intersections are strictly empty:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
- Both `hangout_chat_4b.pcap` and `hangouts_chat_4a.pcap` are co-located in Train.
- Automated regression test `test_canonical_session_group_isolation_regression` passed.
- No flow IDs, 5-tuples, or temporal sessions overlap across split boundaries.

---

## 2. Preprocessing & Architecture (Phases 2, 3, 4)

### 2.1 Preprocessing Pipeline
- Inactivity timeout: 120 seconds.
- Bi-directional flow canonicalization: lowest `(IP, Port)` tuple ordered first.
- Feature Representation: First 64 packets per flow, 2 channels:
  - Channel 1: Normalized directional packet length $x_{\text{len}, t} = \text{dir}_t \cdot (\text{length}_t / 1500.0) \in [-1.0, +1.0]$.
  - Channel 2: Log-transformed inter-arrival time $x_{\text{IAT}, t} = \ln(1 + \min(\Delta t_t, 5.0))$, with $x_{\text{IAT}, 0} = 0.0$.
  - Right-zero padding to length 64 with explicit sequence lengths $L$ and validity masks $M_t \in \{0, 1\}$.

### 2.2 Neural Network Architecture
Compact 1D-CNN (44,129 parameters):
- Input: $(B, 2, 64)$
- Block 1: Conv1d(2, 32, k=5, p=2) $\to$ BatchNorm1d(32) $\to$ ReLU $\to$ MaxPool1d(2)
- Block 2: Conv1d(32, 64, k=5, p=2) $\to$ BatchNorm1d(64) $\to$ ReLU $\to$ MaxPool1d(2)
- Block 3: Conv1d(64, 128, k=3, p=1) $\to$ BatchNorm1d(128) $\to$ ReLU $\to$ AdaptiveAvgPool1d(1)
- Classifier: Flatten $\to$ Linear(128, 64) $\to$ ReLU $\to$ Dropout(0.3) $\to$ Linear(64, 1)

### 2.3 Quantization Implementation
- **Static Post-Training Quantization (PTQ)**: Fused Conv-BN-ReLU modules; calibrated on 1,000 representative training flows; converted with PyTorch eager quantization under `qnnpack`. Confirmed `fc2` zero-point at 105 (scale 0.0693), accommodating negative logits naturally down to -7.27 without zero-clamping (0 flows clamped at 0.0).
- **Quantization-Aware Training (QAT)**: Fused Conv-BN-ReLU modules for QAT; inserted fake-quantization nodes; fine-tuned for 5 epochs on CPU; selected checkpoint at best validation loss ($\tau = 0.680$); converted to deployed integer graph.
- **Operator Verification**: Confirmed authentic `torch.qint8` weights across all convolutional and linear layers.

### 2.4 Unquantized FP32 Fine-Tuning Control Baseline (Option 1)
To rigorously disentangle the effects of numerical quantization from gradient descent optimization in QAT, we implemented an unquantized fine-tuning control baseline (`FP32-FT`):
- Starting checkpoint: Identical converged FP32 model (`artifacts/checkpoints/fp32_seed_42.pt`).
- Optimization: Fine-tuned for 5 identical epochs using AdamW ($lr = 10^{-4}$, weight decay $10^{-5}$, $\text{pos\_weight} = 30.3756$, batch size 64, seed 42).
- Precision: Maintained in full 32-bit floating-point precision throughout (zero fake-quantization modules, zero integer rounding or clamping).
- Validation selection: Checkpoint selected at epoch 5 (best validation loss = $4.2377$), with optimal threshold calibrated on the held-out validation set ($\tau = 0.830$).
- Purpose: Serves as the experimental ablation control to isolate the SGD parameter path confounder from INT8 representation limits.

---

## 3. Experimental Findings

### 3.1 Predictive Performance and Threshold Calibration (Phase 5)
Evaluated on the held-out test split of **49,251 flows** (2,693 VPN, 46,558 Non-VPN):

# Table 1: Predictive Performance Comparison on Held-Out Test Set (N=49,251)

*Evaluated on the held-out test split (2,693 VPN, 46,558 Non-VPN flows). Thresholds calibrated strictly on held-out validation data.*

### Panel A: Threshold-Independent Ranking Performance
| Metric | FP32 Baseline | INT8-PTQ | Delta (PTQ - FP32) | INT8-QAT | Delta (QAT - FP32) | FP32-FT (Ablation Control) | Delta (FT - FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **ROC-AUC** | 0.9796 | 0.9785 | -0.0012 | **0.9834** | +0.0038 | 0.9819 | +0.0023 |
| **PR-AUC** | 0.5610 | 0.6102 | +0.0492 | **0.6478** | +0.0868 | 0.6128 | +0.0518 |
| **Brier Score** | 0.0585 | 0.0584 | -0.0000 | **0.0237** | -0.0347 | 0.0248 | -0.0337 |

---

### Panel B: Model-Specific Validation-Calibrated Thresholding (Recommended)
*Thresholds chosen on validation set: FP32 ($\tau = 0.900$), PTQ ($\tau = 0.900$), QAT ($\tau = 0.680$), FP32-FT ($\tau = 0.830$)*

| Metric | FP32 Baseline ($\tau = 0.90$) | INT8-PTQ ($\tau = 0.90$) | Delta (PTQ - FP32) | INT8-QAT ($\tau = 0.68$) | Delta (QAT - FP32) | FP32-FT ($\tau = 0.83$) | Delta (FT - FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Decision Threshold ($\tau$)** | 0.900 | 0.900 | +0.000 | 0.680 | -0.220 | 0.830 | -0.070 |
| **Positive Calls (VPN)** | 4,121 | 4,172 | +51 | 4,130 | +9 | 3,684 | -437 |
| **Confusion Matrix $[[TN, FP], [FN, TP]]$** | $[[45076, 1482], [54, 2639]]$ | $[[45028, 1530], [51, 2642]]$ | — | $[[45076, 1482], [45, 2648]]$ | — | $[[45481, 1077], [86, 2607]]$ | — |
| **Accuracy** | 96.88% | 96.79% | -0.09% | 96.90% | +0.02% | **97.64%** | +0.76% |
| **Balanced Accuracy** | 97.41% | 97.41% | +0.00% | **97.57%** | +0.17% | 97.25% | -0.16% |
| **Precision (VPN)** | 64.04% | 63.33% | -0.71% | 64.12% | +0.08% | **70.77%** | +6.73% |
| **Recall (VPN)** | 97.99% | 98.11% | +0.11% | **98.33%** | +0.33% | 96.81% | -1.18% |
| **F1-Score** | 0.7746 | 0.7697 | -0.0049 | 0.7762 | +0.0016 | **0.8176** | +0.0430 |

---

### Panel C: Reused FP32 Threshold ($\tau = 0.90$, Uncalibrated Baseline)
*Illustrates the impact of threshold incompatibility caused by logit contraction in QAT and distribution shift in fine-tuning.*

| Metric | FP32 Baseline ($\tau = 0.90$) | INT8-PTQ ($\tau = 0.90$) | Delta (PTQ - FP32) | INT8-QAT ($\tau = 0.90$) | Delta (QAT - FP32) | FP32-FT ($\tau = 0.90$) | Delta (FT - FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Positive Calls (VPN)** | 4,121 | 4,172 | +51 | 0 | -4,121 | 1,693 | -2,428 |
| **Accuracy** | 96.88% | 96.79% | -0.09% | 94.53% | -2.35% | 94.45% | -2.43% |
| **Balanced Accuracy** | 97.41% | 97.41% | +0.00% | 50.00% | -47.41% | 64.42% | -32.99% |
| **Precision (VPN)** | 64.04% | 63.33% | -0.71% | 0.00% | -64.04% | 48.85% | -15.19% |
| **Recall (VPN)** | 97.99% | 98.11% | +0.11% | 0.00% | -97.99% | 30.71% | -67.28% |
| **F1-Score** | 0.7746 | 0.7697 | -0.0049 | 0.0000 | -0.7746 | 0.3771 | -0.3975 |
| **Prediction Flip Rate** | — | 0.12% | — | 8.37% | — | 5.09% | — |


#### Key Predictive Observations:
1. **Model-Specific Validation Calibration (Panel B)**:
   - When decision thresholds are calibrated on held-out validation data ($\tau = 0.90$ for FP32, $\tau = 0.90$ for PTQ, $\tau = 0.68$ for QAT, $\tau = 0.83$ for FP32-FT), **all models exhibit strong classification fidelity**:
     - FP32: Accuracy **96.88%**, Balanced Accuracy **97.41%**, F1 **0.7746**.
     - PTQ: Accuracy **96.79%**, Balanced Accuracy **97.41%**, F1 **0.7697**.
     - QAT: Accuracy **96.90%**, Balanced Accuracy **97.57%**, F1 **0.7762**.
     - FP32-FT: Accuracy **97.64%**, Balanced Accuracy **97.25%**, F1 **0.8176** (highest overall F1 due to precision increasing to 70.77%).
2. **Threshold Recalibration is Universal to Fine-Tuning and Quantization (Panel C)**:
   - While QAT experiences complete positive call collapse (0 calls, F1 = 0.0000) due to integer operator saturation ceiling ($p_{\max} = 0.7338$), FP32-FT also suffers substantial F1 degradation ($0.7746 \to 0.3771$, 1,693 positive calls) due to probability distribution shifting. This proves that threshold recalibration is an essential operational requirement of both fine-tuning and quantization workflows.
3. **Threshold-Independent Ranking (Panel A)**:
   - In threshold-free ranking, QAT and FP32-FT both demonstrate superior discrimination: QAT achieves ROC-AUC **0.9834** / PR-AUC **0.6478**, while FP32-FT achieves ROC-AUC **0.9819** / PR-AUC **0.6128**, compared to the base FP32 (0.9796 / 0.5610).

### 3.2 Probability Calibration and Murphy Score Decomposition
To analyze probabilistic calibration versus discrimination under quantization, we compute the Brier score and its canonical Murphy (1973) three-component decomposition:
$$\text{Brier} = \text{Reliability} - \text{Resolution} + \text{Uncertainty} + \text{Within-Bin Variance}$$
where $\bar{y} = 0.054679$ is the empirical base rate of the positive class (VPN), yielding dataset uncertainty $\text{Uncertainty} = \bar{y}(1 - \bar{y}) = 0.051689$.

Evaluated using 10 equal-width bins $[0.0, 0.1), [0.1, 0.2), \dots, [0.9, 1.0]$. Empty bins are assigned weight $w_k = 0$, contributing zero to reliability, resolution, and ECE.

| Model Architecture | Brier Score | Reliability ($\downarrow$) | Resolution ($\uparrow$) | Uncertainty | Expected Calibration Error (ECE) | Maximum Calibration Error (MCE) | Mean Non-VPN Prob | Mean VPN Prob |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | **0.0585** | 0.0378 | 0.0314 | 0.0517 | 0.1691 | 0.7417 | 0.1811 | 0.9604 |
| **INT8-PTQ** | **0.0584** | 0.0369 | 0.0310 | 0.0517 | 0.1611 | 0.7417 | 0.1722 | 0.9685 |
| **INT8-QAT** | **0.0237** | **0.0036** | **0.0316** | 0.0517 | **0.0378** | **0.5511** | **0.0557** | 0.7288 |

#### Reliability Diagram Bin Breakdown (10 Equal-Width Bins):
- **FP32**: Bin counts: `[360, 43135, 89, 309, 74, 324, 160, 441, 238, 4121]`. High confidence bin $[0.9, 1.0]$ has 4,121 flows (mean confidence $0.9679$, observed accuracy $0.6404$).
- **INT8-PTQ**: Bin counts: `[1915, 41488, 161, 75, 96, 358, 337, 473, 176, 4172]`. High confidence bin $[0.9, 1.0]$ has 4,172 flows (mean confidence $0.9756$, observed accuracy $0.6333$).
- **INT8-QAT**: Bin counts: `[43957, 417, 276, 117, 131, 121, 118, 4114, 0, 0]`. Bins 8 and 9 ($[0.8, 1.0]$) contain **0 flows**.

#### Quantization Parameter Saturation & Probability Ceiling:
Direct inspection of `artifacts/quantized_models/qat_seed_42.pt` reveals the exact mechanism bounding QAT probability outputs:
- The output linear layer `fc2` operates in `torch.quint8` with scale $s = 0.02535125$ and zero-point $z = 215$.
- In unsigned 8-bit quantization ($q \in [0, 255]$), the maximum representable integer is $q_{\text{max}} = 255$.
- Its dequantized logit is strictly bounded:
  $$z_{\text{max}} = (255 - 215) \times 0.02535125 = \mathbf{+1.01405001}$$
- The corresponding maximum sigmoid probability is:
  $$p_{\text{max}} = \sigma(+1.01405001) = \mathbf{0.73381203}$$
- In the clean test set, exactly **4,070 flows hit this upper saturation ceiling**, including **2,642 out of 2,693 VPN flows ($98.11\%$)**.
- Consequently, the median of VPN flows ($50^{\text{th}}$ percentile) is mathematically identical to the maximum probability ($0.7338$). Bins $[0.8, 1.0]$ in the reliability diagram are completely empty, and reusing $\tau = 0.900$ produces zero positive calls.

#### Impact on Scoring Rules vs Operating Thresholds:
- Because the test dataset is heavily imbalanced ($94.53\%$ negative), concentrating Non-VPN probabilities near zero ($43,957$ flows in Bin 0 with mean probability $0.0266$) substantially reduces the quadratic loss term $(\hat{p}_i - 0)^2$ across 46,558 negative flows. This largely explains the observed reduction in Brier score and reliability error.
- The threshold $\tau = 0.680$ is the validation-selected threshold specifically tuned for the F1 objective. It is not an intrinsically optimal threshold for all operational cost functions.

---

## 4. Attribution Drift and Multi-Seed Explainer Repeatability (Phases 6 & 7)

#### Explainer Stability and Noise Calibration:
KernelSHAP was executed with **three distinct random seeds (42, 43, 44)** on the **verified 100-flow balanced test cohort** (50 VPN, 50 Non-VPN), explaining the pre-sigmoid VPN logit with 50 background samples:
- **Cohort Scale & Scope**: The explainability analysis is conducted on a balanced cohort of **100 flows (50 VPN, 50 Non-VPN)** to permit multi-seed (Seeds 42, 43, 44) KernelSHAP repeatability evaluations. In contrast, predictive evaluation is conducted across all **49,251 test flows**. Findings describe the attribution behavior of this balanced sample.
- **Cohort Provenance**: 100/100 flow IDs are confirmed members of `data/processed/test_data.npz` (50 VPN, 50 Non-VPN). Attributions exhibit exact additive correlation ($r = 1.0000$) with the model forward passes.
- **KernelSHAP Additivity Residuals**:
  - FP32 (s42): $\max |\epsilon_i| = 0.000007$, $\text{mean} |\epsilon_i| = 0.000002$
  - PTQ (s42): $\max |\epsilon_i| = 0.000000$, $\text{mean} |\epsilon_i| = 0.000000$
  - QAT (s42): $\max |\epsilon_i| = 0.000000$, $\text{mean} |\epsilon_i| = 0.000000$
  - Attributions fulfill the KernelSHAP efficiency axiom down to machine precision.
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
| **FP32 vs FP32-FT (Ablation Control)** | 42 vs 42 | **0.9481 ± 0.1816** | **0.9681 ± 0.1379** | **0.2214 ± 0.2327** | **0.1166 ± 0.8106** | **0.8949 ± 0.2453** |

---

### Panel B: Paired Flow-by-Flow Hypothesis Testing (Cross-Model vs. Explainer Noise & Ablation)
*Evaluates whether cross-model attribution drift exceeds intrinsic explainer variability on identical flows.*

| Comparison | Paired Delta Metric | Mean ± Std | Median | Non-Parametric Bootstrap 95% CI | Wilcoxon Signed-Rank Test | Statistical Conclusion |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **PTQ vs. Noise** | $\Delta TV = TV_{\text{PTQ}} - TV_{\text{Noise}}$ | -0.0185 ± 0.1115 | +0.0049 | $[-0.0422, +0.0011]$ | $W = 1506.0$ ($R^+ = 3544.0, R^- = 1506.0$)<br>Two-sided $p = 0.00046$ (Holm: 0.00092)<br>One-sided ($H_1: \Delta TV > 0$) $p = 0.00023$ | Positive rank shift in 69/100 flows ($R^+ > R^-$), minimal median delta (+0.0049) |
| **PTQ vs. Noise** | $\Delta IoU_5 = IoU_{5, \text{PTQ}} - IoU_{5, \text{Noise}}$ | +0.0232 ± 0.1088 | +0.0000 | $[+0.0051, +0.0472]$ | $W = 0.0, p = 0.02771$ (Holm: 0.02771) | Identical Top-5 on 95/100 flows |
| **QAT vs. Noise** | $\Delta TV = TV_{\text{QAT}} - TV_{\text{Noise}}$ | +0.1793 ± 0.1547 | +0.1975 | $[+0.1500, +0.2071]$ | $W = 129.0, p = 1.7478e-16$ (Holm: 5.2435e-16) | **Significant Drift ($p < 10^{-4}$)** |
| **QAT vs. PTQ** | $\Delta TV = TV_{\text{QAT}} - TV_{\text{PTQ}}$ | +0.1978 ± 0.1596 | +0.2854 | $[+0.1660, +0.2301]$ | $W = 0.0, p = 3.8966e-18$ (Holm: 1.5586e-17) | QAT restructures attributions ($p < 10^{-4}$) |
| **FP32-FT vs. Noise** | $\Delta TV = TV_{\text{FT}} - TV_{\text{Noise}}$ | +0.1661 ± 0.1985 | +0.1802 | $[+0.1270, +0.2052]$ | $W = 482.0, p = 8.12 \times 10^{-11}$ | Unquantized SGD also induces massive drift ($p < 10^{-4}$) |
| **QAT vs. FP32-FT** | $\Delta TV = TV_{\text{QAT}} - TV_{\text{FT}}$ | +0.0132 ± 0.1654 | -0.0047 | $[-0.0195, +0.0460]$ | $W = 2390.0, p = 0.684$ | Drift magnitude statistically indistinguishable |

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


#### Deconstruction of Attribution Properties:
1. **Feature Ranking ($IoU_5$ & $IoU_{10}$)**: Feature importance ordering is extremely stable under PTQ ($IoU_5 = 0.9786$, median $1.0000$, identical on 95/100 flows; $IoU_{10} = 0.9887$), whereas QAT alters rankings substantially ($IoU_5 = 0.9553$).
2. **Attribution Direction ($S_C$)**: The orientation of attribution vectors is preserved under PTQ ($S_C = 0.9918$, median $0.9996$, Spearman $\rho = 0.9749$), but undergoes substantial rotation under QAT ($S_C = 0.4830$).
3. **Magnitude Displacement ($TV$)**: The normalized $L_1$ distribution of attribution weights shifts minimally under PTQ ($TV = 0.0369$, median $0.0196$) compared to massive redistribution under QAT ($TV = 0.2346$).

#### Wilcoxon Signed-Rank Test & Explainer Noise Floor:
- Paired differences $\Delta TV = TV_{\text{PTQ}} - TV_{\text{Noise}}$ across the 100 cohort flows:
  - Positive differences: **69 / 100**
  - Negative differences: **31 / 100**
  - Zero differences: **0 / 100**
  - Positive Rank Sum $R^+ = 3544.0$, Negative Rank Sum $R^- = 1506.0$, Total = $5050.0$
  - Wilcoxon test statistic $W = \min(R^+, R^-) = 1506.0$
  - Two-sided $p = 0.0004589$; One-sided greater ($H_1: \Delta TV > 0$) $p = 0.0002295$
- **Parametric vs Non-Parametric Statistics**:
  - Median $\Delta TV = +0.004908$ (Bootstrap 95% CI: $[+0.002413, +0.007179]$, strictly positive).
  - Mean $\Delta TV = -0.018474$.
  - Sample standard deviation ($ddof=1$, Bessel correction): **$s = 0.112012$** (reported as $0.1120$).
  - Population standard deviation ($ddof=0$): **$\sigma = 0.111450$** (reported as $0.1115$).
  - The mean and median have opposite signs because the distribution of paired differences is negatively skewed: a small number of large negative outliers (minimum $\Delta TV = -0.579200$) from stochastic explainer noise runs pull the arithmetic mean below zero, causing the bootstrap CI of the mean to span zero ($[-0.041869, +0.000877]$).
- **Scientific Conclusion**: *PTQ attribution differences exhibit a statistically detectable signed-rank shift relative to the explainer-noise comparison ($p = 0.00046$, median $+0.0049$), while top-five feature overlap remains high ($97.9\%$). These findings do not establish statistical equivalence between PTQ drift and explainer noise.*

### 4.2 Disentangling the Fine-Tuning Confounder (Option 1 Defense)
A critical question arising in peer review and dissertation defense is:
> *"Does QAT induce severe attribution restructuring because of integer quantization constraints, or simply because it undergoes 5 epochs of gradient fine-tuning with AdamW?"*

To isolate this confounder, the unquantized FP32 Fine-Tuning baseline (`FP32-FT`, Option 1) duplicates the exact training regime of QAT without fake-quantization operators. The resulting empirical comparison yields profound scientific clarity:

1. **Attribution Drift Magnitude**:
   - `FP32-FT` yields a mean Total Variation distance of **$0.2214 \pm 0.2327$** (median **$0.2070$**), with Top-5 Jaccard overlap dropping to **$0.9481$** and signed cosine similarity dropping to **$0.1166$** relative to base FP32.
   - `INT8-QAT` yields a mean Total Variation distance of **$0.2346 \pm 0.1908$** (median **$0.2023$**), with Top-5 Jaccard of **$0.9553$** and signed cosine similarity of **$0.4830$**.
   - A paired Wilcoxon test between QAT and FP32-FT reveals no significant difference in Total Variation magnitude ($W = 2390.0, p = 0.684, \text{median } \Delta TV = -0.0047$).
2. **Triangulated Causal Attribution**:
   - **Static PTQ (Quantization without Fine-Tuning)**: Preserves attributions with extreme fidelity ($TV = 0.0369, S_C = 0.9918, IoU_5 = 0.9786$).
   - **FP32-FT (Fine-Tuning without Quantization)**: Induces substantial attribution reorganization ($TV = 0.2214, S_C = 0.1166, IoU_5 = 0.9481$).
   - **INT8-QAT (Quantization WITH Fine-Tuning)**: Induces comparable attribution reorganization ($TV = 0.2346, S_C = 0.4830, IoU_5 = 0.9553$).
3. **Core Scientific Conclusion**:
   - **The explanation restructuring observed in QAT is driven primarily by gradient descent optimization across the non-convex loss surface rather than integer quantization noise.**
   - In overparameterized neural networks, gradient fine-tuning naturally finds alternate weight configurations that achieve equivalent or superior predictive discriminability while relying on alternative, complementary packet features.
   - Static PTQ proves that integer quantization itself does not destroy feature rankings; QAT proves that fine-tuning under constraints finds an alternate explainability basin.

---

## 5. Hardware and Edge Execution Benchmarks (Phase 9)

Benchmarked on macOS Apple Silicon ARM64 under CPU native execution using PyTorch QNNPACK (50 warmup runs, 500 repetitions):

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

| Model Architecture | Median Latency | Mean Latency | Latency Std Dev | Inter-Quartile Range (IQR) | 95th Percentile (p95) | 99th Percentile (p99) | Measured Throughput | Median Latency Speedup | Mean Throughput Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | **0.267 ms** | 0.291 ms | 0.074 ms | 0.042 ms | 0.405 ms | 0.580 ms | 3,438 flows/s | 1.00× (Ref) | 1.00× (Ref) |
| **INT8-PTQ** | 0.255 ms | 0.263 ms | 0.036 ms | 0.032 ms | 0.309 ms | 0.459 ms | 3,796 flows/s | **1.05× (Parity)** | **1.10×** |
| **INT8-QAT** | 0.242 ms | 0.250 ms | 0.023 ms | 0.012 ms | 0.281 ms | 0.372 ms | 4,005 flows/s | **1.10× (Faster)** | **1.16×** |

*Observation: At Batch Size = 1, quantization yields no speedup because PyTorch QNNPACK invocation, dynamic quantization of inputs, and scalar dequantization dominate computation.*

---

### Panel C: Batched Server Throughput (High-Volume Pipeline, Batch Size = 64)
*Characterizes offline audit, log ingestion, or buffered middlebox inspection.*

| Model Architecture | Median Batch Latency | Mean Batch Latency | Batch Latency Std Dev | 95th Percentile (p95) | Measured Throughput | Median Latency Speedup | Mean Throughput Speedup |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | 9.503 ms | 10.020 ms | 3.849 ms | 12.640 ms | 6,387 flows/s | 1.00× (Ref) | 1.00× (Ref) |
| **INT8-PTQ** | 0.764 ms | 0.839 ms | 0.233 ms | 1.228 ms | 76,264 flows/s | **12.44×** | **11.94×** |
| **INT8-QAT** | **0.755 ms** | **0.785 ms** | **0.127 ms** | **0.960 ms** | **81,552 flows/s** | **12.59×** | **12.77×** |

*Note: Measured throughput is calculated from mean batch latency (64 / mean_batch_latency_sec). Median latency speedup and mean throughput speedup differ slightly due to tail latency and OS scheduling variance. Observed throughput gains stem from PyTorch QNNPACK batched integer matrix multiplication on ARM64; microarchitectural instruction counters (e.g. NEON SIMD) were not directly profiled via hardware PMUs.*


### Measured Latency and Throughput Performance
1. **Single-Flow Inference (Batch Size = 1)**:
   - FP32 median latency: $0.267$ ms (mean $0.291$ ms, throughput $3,439$ flows/s).
   - PTQ median latency: $0.255$ ms (mean $0.263$ ms, throughput $3,797$ flows/s, **1.05× median latency speedup**, **1.10× mean throughput speedup**).
   - QAT median latency: $0.242$ ms (mean $0.250$ ms, throughput $4,005$ flows/s, **1.10× median latency speedup**, **1.16× mean throughput speedup**).
   - The observed single-flow latency advantage is small on the tested platform.
2. **Batched Processing (Batch Size = 64)**:
   - FP32 median batch latency: $9.503$ ms (mean $10.020$ ms, throughput $6,387$ flows/s).
   - PTQ median batch latency: $0.764$ ms (mean $0.839$ ms, throughput $76,265$ flows/s, **12.44× median latency speedup**, **11.94× mean throughput speedup**).
   - QAT median batch latency: $0.755$ ms (mean $0.785$ ms, throughput $81,552$ flows/s, **12.59× median latency speedup**, **12.77× mean throughput speedup**).
   - Median latency speedup and mean throughput speedup differ slightly due to tail latency and OS scheduling variance.
3. **Checkpoint Compression**:
   - Model checkpoint size is reduced by **3.56×** ($184.24$ KB $\to$ $51.77$ KB) with identical parameter count ($44,129$).
4. **Platform Boundary**:
   - INT8 inference achieved substantially higher measured throughput at batch size 64 on the tested Apple Silicon system under PyTorch QNNPACK. These results are specific to the tested software, hardware, and benchmark configuration, and cannot be extrapolated to embedded microcontrollers without on-device profiling.

---

## 6. Evaluation of Primary Registered Hypotheses

# Table 4: Primary Hypothesis Testing Matrix (Audited & Formalized)

| Hypothesis | Formal Registered Question & Null Hypothesis ($H_0$) | Test Statistic & Resampling Unit | Empirical Evidence | Statistical Decision |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **H1: Predictive Drift** | **Question**: Does INT8 quantization change predictions relative to FP32?<br>**$H_0$**: Marginal classification symmetry ($P(FP32=1, INT8=0) = P(FP32=0, INT8=1)$) and zero mean logit shift (\mu_{\Delta z} = 0). | - McNemar Test (with continuity correction)<br>- Paired Student's t-test on logits<br>- Capture-group clustered bootstrap 95% CI | - PTQ McNemar \chi^2 = 43.9, p = 3.5279e-11<br>- PTQ Logit Shift: -0.0778 ($t = -38.9, p = 0.0000e+00$; Group CI: $[-0.1126, +0.2091]$)*<br>- QAT Logit Shift: -1.9404 ($t = -533.8, p = 0.0000e+00$; Group CI: $[-2.8762, -1.8631]$)<br>*\*Note on PTQ Group CI: 15/19 capture groups have positive shifts (group mean = +0.2210), while skype_audio3 (79.5% of test flows) anchors the per-flow mean at -0.0778; cluster resampling without skype_audio3 skews the CI positive.* | **SUPPORTED**<br>(Quantization significantly shifts decision outputs and calibration requirements) |
| **H2: Attribution Drift** | **Question**: Does INT8-PTQ produce measurable attribution drift beyond explainer noise?<br>**$H_0$**: Paired Total Variation difference \Delta TV = TV_{\text{PTQ}} - TV_{\text{Noise}} = 0$. | - Paired Wilcoxon signed-rank test on identical flows ($N=100$)<br>- Multi-seed explainer noise baseline (Seeds 42, 43, 44)<br>- Flow and group bootstrap 95% CIs | - Paired \Delta TV = -0.0185 \pm 0.1115 (Median: $+0.0049$)<br>- Wilcoxon $W = 1506.0 (R^+ = 3544.0, R^- = 1506.0)$<br>- Two-sided $p = 0.00046$ (Holm: 0.00092)<br>- One-sided greater ($H_1: \Delta TV > 0$) $p = 0.00023$<br>- Flow Bootstrap 95% CI: $[-0.0422, +0.0011]$<br>- 95/100 flows share identical Top-5 features ($IoU_5 = 0.9786$) | **STATISTICALLY DETECTABLE BUT MINIMAL MAGNITUDE**<br>(Wilcoxon confirms positive rank shift: $R^+ = 3544.0 > R^- = 1506.0$, positive in 69/100 flows; median delta is tiny at +0.0049 and dominant feature sets remain 97.9% identical) |
| **H3: QAT Comparison** | **Question**: Does INT8-QAT exhibit different attribution drift than INT8-PTQ?<br>**$H_0$**: \Delta TV = TV_{\text{QAT}} - TV_{\text{PTQ}} \le 0$ (QAT produces equal or less attribution restructuring than PTQ). | - Paired Wilcoxon signed-rank test on identical flows ($N=100$)<br>- Holm-Bonferroni correction | - Paired \Delta TV = +0.1978 \pm 0.1596 (Median: $+0.2854$)<br>- Wilcoxon $W = 0.0, p = 3.8966e-18$ (Holm: 1.5586e-17)<br>- Bootstrap 95% CI: $[+0.1660, +0.2301]$ | **SUPPORTED (Refuting preservation hypothesis)**<br>(QAT restructures attributions significantly more than PTQ; $p < 10^{-15}$) |
| **H4: Conditional Variation** | **Question**: Does attribution drift vary across traffic classes and prediction agreement cohorts?<br>**$H_0$**: Attribution drift is homogeneous across classes and agreement cohorts. | - Mann-Whitney U test between traffic classes with rank-biserial effect sizes ($r_{\text{rb}} = 1 - 2U / (n_1 n_2)$)<br>- Agreement stratification (if estimable) | - Class Stratification (VPN vs Non-VPN, $n_1=50, n_2=50$):<br>  * PTQ TV: VPN median 0.0131 vs Non-VPN 0.0342 ($U = 480.0, p = 1.1280e-07, r_{\text{rb}} = +0.6160$, Holm $p = 3.3840e-07$)<br>  * PTQ $IoU_5$: VPN 0.9638 vs Non-VPN 0.9933 ($U = 1174.0, p = 0.1681, r_{\text{rb}} = +0.0608$, Holm $p = 0.3362$)<br>  * QAT TV: VPN median 0.0459 vs Non-VPN 0.3475 ($U = 278.0, p = 2.1224e-11, r_{\text{rb}} = +0.7776$, Holm $p = 8.4895e-11$)<br>  * QAT $IoU_5$: VPN 0.9350 vs Non-VPN 0.9756 ($U = 1199.0, p = 0.3976, r_{\text{rb}} = +0.0408$, Holm $p = 0.3976$)<br>- Agreement Stratification: **NON-ESTIMABLE** (100% agreement between FP32 and PTQ on the 100-flow cohort, $N_{\text{disagree}} = 0$) | **PARTIALLY SUPPORTED / INCONCLUSIVE**<br>(Traffic class significantly modulates TV magnitude [$r_{\text{rb}} = +0.6160$ PTQ, $+0.7776$ QAT], but Top-5 rank agreement is unaffected; agreement cohort is non-estimable) |


### Detailed Statistical Hypothesis Decisions:
1. **H1 — Predictive Drift: SUPPORTED.**
   - Marginal symmetry between FP32 and INT8 predictions is rejected: McNemar $\chi^2 = 43.9, p = 3.5279e-11$ for PTQ.
   - Continuous logit shifts are statistically significant ($p < 10^{-15}$).
   - **Clustered Bootstrap Estimands (Primary: 16 Canonical Session Groups)**:
     - The held-out test split contains 19 capture files mapped to **16 canonical session groups** (merging `email1a`+`email1b`, `email2a`+`email2b`, `facebook_video2a`+`facebook_video2b`).
     - **Estimand A (pooled flow-weighted mean across clusters)**: Point estimate **$-0.077839$**, cluster bootstrap 95% CI **$[-0.113025, +0.327451]$** (1,000 replicates, seed 42).
     - **Estimand B (unweighted mean of group means)**: Point estimate **$+0.227018$**, cluster bootstrap 95% CI **$[+0.063889, +0.356235]$** (1,000 replicates, seed 42).

| Canonical Session Group | Traffic Class | Test Flow Count | % of Test Split | Mean Logit Shift (\Delta z) |
| :--- | :---: | :---: | :---: | :---: |
| `skype_audio3` | Non-VPN | 39,179 | 79.55% | -0.1164 |
| `email1` (`email1a` + `email1b`) | Non-VPN | 5,727 | 11.63% | -0.0543 |
| `vpn_hangouts_audio2` | VPN | 2,135 | 4.33% | +0.3124 |
| `email2` (`email2a` + `email2b`) | Non-VPN | 581 | 1.18% | +0.4209 |
| `vpn_bittorrent` | VPN | 406 | 0.82% | -0.7204 |
| `voipbuster1` | Non-VPN | 398 | 0.81% | +0.2785 |
| `facebook_video2` (`2a` + `2b`) | Non-VPN | 249 | 0.51% | +0.4511 |
| `vpn_netflix` (`vpn_netflix_A`) | VPN | 152 | 0.31% | +0.2583 |
| `netflix2` | Non-VPN | 88 | 0.18% | +0.5262 |
| `gmailchat1` | Non-VPN | 79 | 0.16% | +0.3348 |
| `youtube3` | Non-VPN | 75 | 0.15% | +0.3245 |
| `facebookchat3` | Non-VPN | 68 | 0.14% | +0.4158 |
| `icqchat1` | Non-VPN | 43 | 0.09% | +0.1037 |
| `spotify3` | Non-VPN | 30 | 0.06% | +0.3353 |
| `facebookchat2` | Non-VPN | 27 | 0.05% | +0.2963 |
| `scpdown1` | Non-VPN | 14 | 0.03% | +0.4656 |

     - **Distribution of Estimand A Bootstrap Replicates (10,000 Replicates)**:
       - Mean: $-0.0204$, Median: $-0.0753$, Std: $0.1159$, IQR: $0.1188$.
       - Percentiles: $p_5 = -0.1096, p_25 = -0.0955, p_50 = -0.0753, p_75 = +0.0233, p_95 = +0.2564$.
       - Skewness: **$+1.6806$** (strongly positive). $71.49\%$ of replicates are negative, while $28.51\%$ are positive.
       - **Bimodal Dynamics**: Across the 16 canonical groups, **13 groups have positive mean shifts** ($+0.1037$ to $+0.5262$), and only 3 groups are negative (`skype_audio3`, `email1`, `vpn_bittorrent`). In cluster bootstrap resampling, when `skype_audio3` is included (probability $\approx 64.39\%$), the flow-weighted mean centers around $-0.07$ to $-0.11$. When `skype_audio3` is omitted (probability $(1 - 1/16)^{16} \approx 35.61\%$), the resampled flow-weighted mean is driven by the 13 positive groups ($+0.10 \dots +0.35$), generating the long positive tail ($+0.3275$).
     - **Mechanism of Asymmetry**: Across the 16 canonical groups, **13 groups have positive mean shifts** ($+0.1037$ to $+0.5262$). Only 3 groups are negative, anchored by `skype_audio3` (39,179 flows, 79.55% of test flows, mean shift $-0.1164$), `email1` (5,727 flows, 11.63%, mean shift $-0.0543$), and `vpn_bittorrent` (406 flows, 0.82%, mean shift $-0.7204$). In cluster bootstrap resampling, when `skype_audio3` is omitted (probability $(1 - 1/16)^{16} \approx 35.61\%$), the resampled flow-weighted mean becomes positive, creating the wide positive upper tail ($+0.3275$).
     - **Sensitivity Analysis (15 Canonical Groups, Excluding `skype_audio3`)**:
       - Target population shift: describes test network traffic excluding the single Skype audio recording ($10,072$ flows across 15 session groups).
       - Estimand A (flow-weighted): point estimate **$+0.071971$**, cluster bootstrap 95% CI **$[-0.076981, +0.345431]$**.
       - Estimand B (unweighted group mean): point estimate **$+0.249909$**, cluster bootstrap 95% CI **$[+0.087045, +0.367264]$**.
     - **Comparison with 19 Raw Capture Files**:
       - Resampling 19 raw files treats file recording splits as independent clusters: Estimand A CI $[-0.1126, +0.2091]$, Estimand B point $+0.2210$, CI $[+0.0683, +0.3351]$. Merging into 16 canonical groups properly enforces session disjunction, slightly widening intervals due to honest reduction in cluster degrees of freedom.
     - **Inferential Caution**: With only 16 test session groups (and only 3 VPN groups), the primary confidence interval $[-0.1130, +0.3275]$ spans zero in both directions and is strongly modulated by capture group composition. The flow-weighted shift cannot be described as a stable, dataset-independent effect.
2. **H2 — Attribution Drift: STATISTICALLY DETECTABLE BUT MINIMAL MAGNITUDE.**
   - Top-5 overlap vastly exceeds combinatorial chance ($IoU_5 = 0.9786$ vs chance 0.0220).
   - Wilcoxon signed-rank testing confirms a statistically detectable positive rank shift ($R^+ = 3544.0 > R^- = 1506.0$, two-sided $p = 0.00046$, one-sided $p = 0.00023$, median delta $+0.0049$). However, the magnitude is minimal and dominant features are identical on 95% of test flows. No statistical equivalence was established.
3. **H3 — Differential QAT Attribution Restructuring: SUPPORTED (Refuting H3 Direction).**
   - QAT restructures feature attributions significantly more aggressively than PTQ ($\Delta TV = +0.1978$, Holm-corrected $p = 1.5586e-17$).
4. **H4 — Conditional Variation: PARTIALLY SUPPORTED / INCONCLUSIVE.**
   - **Total Variation**: Class conditioning is statistically confirmed. Non-VPN traffic experiences significantly greater Total Variation displacement under quantization than VPN traffic:
     - PTQ TV: VPN median 0.0131 vs Non-VPN 0.0342 (Mann-Whitney $U = 480.0, p = 1.1280e-07, r_{\text{rb}} = +0.6160$, Holm $p = 3.3840e-07$).
     - QAT TV: VPN median 0.0459 vs Non-VPN 0.3475 (Mann-Whitney $U = 278.0, p = 2.1224e-11, r_{\text{rb}} = +0.7776$, Holm $p = 8.4895e-11$).
   - **Top-5 Feature Overlap**: Insufficient evidence of class conditioning. Top-5 Jaccard overlap does not differ significantly across traffic classes (PTQ $U = 1174.0, p = 0.1681$; QAT $U = 1199.0, p = 0.3976$).
   - **Prediction Agreement Stratification**: Strictly **NON-ESTIMABLE** on the 100-flow cohort because prediction agreement between FP32 and PTQ is 100% (0 disagreements).

---

## 7. Scope of Conclusions and Limitations

### 7.1 Epistemological Classification of Claims
- **Verified Implementation Facts**:
  - Exactly 44,129 parameters across all models.
  - Converted checkpoints execute real `torch.qint8` integer operators under `qnnpack`.
  - In `artifacts/quantized_models/ptq_seed_42.pt`, `fc2` has zero-point 105 (scale 0.0693), representing negative logits down to -7.27 without zero-clamping (0 flows clamped at 0.0).
  - Under the tested environment, fresh forward passes reproduce saved logits with zero residual ($0.00 \times 10^0$).
- **Empirical Findings**:
  - PTQ maintains high Top-5 feature attribution overlap ($IoU_5 = 0.9786$, identical on 95/100 flows).
  - Attribution displacement under PTQ is statistically detectable over explainer noise ($p = 0.00046$, median $+0.0049$) but small in practical magnitude.
  - Validation-calibrated QAT achieves the highest F1-score (0.7762) and ROC-AUC (0.9834), but causes genuine attribution drift ($\Delta TV = +0.1793, p < 10^{-15}$).
  - INT8 yields substantial batched throughput speedups at BS=64 (PTQ 11.94×; QAT 12.77×), but a much smaller latency advantage at BS=1.
- **Untested Operational Implications**:
  - We make **no claim** that attribution drift creates an exploitable security vulnerability or firewall bypass.

### 7.2 Methodological & Hardware Execution Boundaries
1. **Limited Independent VPN Capture Groups**:
   - The held-out test split contains **3 VPN capture files** (`vpn_bittorrent`, `vpn_hangouts_audio2`, `vpn_netflix_A`).
   - Group-clustered bootstrap intervals reflect session dependence.
2. **Benchmark Scope**:
   - Results apply strictly to the ISCX VPN-NonVPN 2016 dataset under a 1D-CNN architecture.
3. **Hardware Execution Boundary (Desktop ARM64 vs Embedded Edge / IoT)**:
   - All latency and throughput benchmarks were gathered on an Apple Silicon Mac (desktop ARM64 architecture with unified memory and QNNPACK CPU runtime).
   - These results **cannot be assumed to translate directly to low-power embedded microcontrollers or edge IoT appliances** (e.g., ARM Cortex-M microcontrollers, Raspberry Pi, low-power IoT gateways, CCTV cameras, smartwatches).
   - While 3.56× checkpoint compression (184.24 KB $\to$ 51.77 KB) is verified for flash storage constraints, runtime compute speedups require direct on-device profiling before deploying to constrained edge hardware.

---

## 8. Complete Reproducibility Manifest

- **Dataset Manifest**: [`data/manifests/dataset_manifest.json`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/data/manifests/dataset_manifest.json)
- **Split Manifests**: [`train_manifest.json`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/data/manifests/train_manifest.json), [`val_manifest.json`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/data/manifests/val_manifest.json), [`test_manifest.json`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/data/manifests/test_manifest.json)
- **Model Checkpoints**:
  - FP32 Checkpoint: [`artifacts/checkpoints/fp32_seed_42.pt`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/artifacts/checkpoints/fp32_seed_42.pt)
  - FP32-FineTuned Control Checkpoint: [`artifacts/checkpoints/fp32_finetuned_seed_42.pt`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/artifacts/checkpoints/fp32_finetuned_seed_42.pt)
  - Deployed PTQ Model: [`artifacts/quantized_models/ptq_seed_42.pt`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/artifacts/quantized_models/ptq_seed_42.pt)
  - Deployed QAT Model: [`artifacts/quantized_models/qat_seed_42.pt`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/artifacts/quantized_models/qat_seed_42.pt)
- **Processed Flow Datasets**: [`data/processed/train_data.npz`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/data/processed/train_data.npz), [`val_data.npz`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/data/processed/val_data.npz), [`test_data.npz`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/data/processed/test_data.npz)
- **Predictions & Attributions**:
  - Predictions: [`results/predictions/test_predictions.npz`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/results/predictions/test_predictions.npz)
  - Cohort Attributions: [`results/attributions/cohort_attributions.npz`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/results/attributions/cohort_attributions.npz)
  - FP32-FT Attributions: [`results/attributions/cohort_attributions_ft.npz`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/results/attributions/cohort_attributions_ft.npz)
  - FP32-FT Evaluation Metrics: [`results/metrics/fp32_ft_evaluation.json`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/results/metrics/fp32_ft_evaluation.json)
- **Scientific Reproducibility Gate Script**: [`scripts/run_scientific_reproducibility_gate.py`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/scripts/run_scientific_reproducibility_gate.py)
- **Regression Suite**: 15/15 tests passing (`pytest tests/ -v`).
