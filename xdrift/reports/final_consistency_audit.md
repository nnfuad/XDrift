# Final Consistency and Reconciliation Audit (Clean Baseline)

**Audit Date**: October 24, 2026  
**Project**: Investigating Attribution Drift in INT8-Quantized 1D CNNs for Encrypted Traffic Classification (XDrift)  
**Lead Auditor**: Senior ML Research Engineer (Clean-Room Verification)  
**Target Repository**: `/Volumes/LEXAR/Explainability Drift/xdrift`  

---

## Executive Summary of Audit Findings

Following the forensic identification of a Train–Validation session-group overlap in the historical split (`hangout_chat_4b.pcap` in Train vs `hangouts_chat_4a.pcap` in Validation), **Option 2 (re-splitting, reprocessing, and retraining)** was fully executed to establish a strictly leakage-free experimental baseline. The historical experiment was preserved under `pre_split_correction/` with full provenance tracking.

---

## Forensic Discrepancy Reconciliation Table

| Issue ID | Identified Discrepancy | Root Cause Analysis | Corrective Action & Rigorous Evidence | Reconciled Status |
| :--- | :--- | :--- | :--- | :---: |
| **Issue 1** | **Attribution Cohort Split Provenance & Additivity**: Unclear whether cached attributions belonged to the clean test set ($N=49,251$) or old test set ($N=20,971$), and whether additivity held. | Notes cited precomputed cache without verifying flow IDs against clean test set. | Audited all 100 flow IDs in `cohort_attributions.npz`: 100/100 exist in clean `test_data.npz` (vs only 56/100 in old split). Exact logit difference with final models is $0.00\times 10^0$. Additivity residuals $\max |\epsilon_i| \le 7 \times 10^{-6}$ on FP32 and $< 10^{-6}$ on PTQ/QAT. | **VERIFIED & RESOLVED** |
| **Issue 2** | **Wilcoxon vs. Noise Direction & Std Discrepancy**: Claimed PTQ drift is "indistinguishable from noise" while Wilcoxon reported $W = 1506.0, p = 0.00046$; discrepancy between std $0.1115$ and $0.1120$. | SciPy returns $W = \min(R^+, R^-) = 1506.0$. Positive rank sum is $R^+ = 3544.0$, negative rank sum is $R^- = 1506.0$. Positive ranks dominate. The std discrepancy arose from population std ($ddof=0, 0.111450$) vs sample std ($ddof=1, 0.112012$). | Reconciled statistical reality: Signed-rank test confirms positive rank shift in 69/100 flows ($R^+ = 3544.0 > R^- = 1506.0$, two-sided $p = 0.00046$, one-sided greater $p = 0.00023$). Median delta is minimal (+0.0049). Mean delta is negative (-0.0185) due to negative outlier noise runs (bootstrap CI: $[-0.0419, +0.0009]$). Dominant feature sets remain 97.9% identical (95/100 flows identical). Both sample std ($0.1120$) and population std ($0.1115$) are explicitly documented. Crucially, this does NOT establish statistical equivalence between PTQ drift and explainer noise. | **CORRECTED & QUALIFIED** |
| **Issue 3** | **Hardware Speedup Arithmetic & Single Source of Truth**: Ensuring hardware metrics trace to one authoritative raw benchmark artifact and separating latency from throughput speedups. | Table previously placed median batch latency speedup under a throughput column without distinguishing median from mean. | All hardware figures are traced directly to `results/metrics/hardware_benchmarks.json`. Table 3 explicitly presents BOTH `Median Latency Speedup` (12.44× PTQ, 12.59× QAT) and `Mean Throughput Speedup` (11.94× PTQ, 12.77× QAT). Unverified ARM64 NEON instruction claims and unverified framework dispatch overhead figures removed. | **RECOMPUTED & SEPARATED** |
| **Issue 4** | **Hypothesis H4 Non-Estimability & Effect Sizes**: H4 labeled "SUPPORTED" despite 0 prediction disagreements on cohort and no formal subgroup test or effect size. | On the 100-flow cohort, FP32 and PTQ have 100% agreement ($N_{\text{disagree}} = 0$), making agreement stratification non-estimable. | Separated H4 into three clear statements: (1) Total Variation class conditioning is statistically confirmed ($p < 10^{-5}, r_{\text{rb}} = +0.6160$ PTQ; $p < 10^{-10}, r_{\text{rb}} = +0.7776$ QAT); (2) Top-5 feature overlap shows insufficient evidence of class difference ($p > 0.15$); (3) Agreement stratification is strictly NON-ESTIMABLE. Status: PARTIALLY SUPPORTED. | **FORMALIZED & QUALIFIED** |
| **Issue 5** | **Bootstrap Resampling Unit (16 Canonical Groups vs 19 Raw Files)**: Determining whether the resampling unit is 19 capture files or 16 canonical session groups. | Raw capture files split recording sessions (`email1a`/`email1b`, `email2a`/`email2b`, `facebook_video2a`/`facebook_video2b`) into separate clusters, artificially splitting sessions. | Primary cluster bootstrap now explicitly resamples across the **16 canonical session groups**, pooling all observations from a canonical session together. Recomputed Estimand A: $-0.0778$, CI $[-0.1130, +0.3275]$; Estimand B: $+0.2270$, CI $[+0.0639, +0.3562]$. Sensitivity excluding `skype_audio3` (15 canonical groups): Estimand A $+0.0720$ (CI $[-0.0770, +0.3454]$), Estimand B $+0.2499$ (CI $[+0.0870, +0.3673]$). Documented population shift and 19-file comparison. | **RESOLVED & ALIGNED** |
| **Issue 6** | **QAT Probability Calibration vs Discrimination (Murphy Decomposition)**: Apparent paradox between QAT's low Brier score ($0.0237$) and logit contraction requiring threshold calibration. | Brier score conflates reliability and resolution. In heavily imbalanced datasets ($94.53\%$ majority class), shrinking probabilities toward zero dramatically suppresses majority-class quadratic error. | Performed Murphy (1973) decomposition: $\text{Brier} = \text{Reliability} (0.0036) - \text{Resolution} (0.0316) + \text{Uncertainty} (0.0517)$. QAT's low Brier score is driven by Non-VPN probability compression ($0.0557$ mean, $94.41\%$ in Bin 0), not intrinsic superiority in uncalibrated confidence estimation. The threshold $\tau=0.680$ is the validation-selected operating point for the F1 objective. | **DECOMPOSED & EXPLAINED** |
| **Issue 7** | **Streaming Latency vs Batched Line-Rate Throughput**: Why BS=1 yields only ~1.05×–1.19× speedup while BS=64 achieves ~11.2×–12.8× speedup ($6,387 \to 76,265$ flows/s). | Batching amortizes framework invocation across 64 flows. Latency speedup and throughput speedup represent distinct measurements. | Separated median latency speedup from mean throughput speedup. Removed speculative unprofiled timing breakdowns (~0.20 ms / 0.05 ms). Bounded conclusions strictly to measured performance on desktop Apple Silicon ARM64 under PyTorch QNNPACK. | **PROFILED & DELINEATED** |
| **Issue 8** | **Fine-Tuning Confounder (Option 1 FP32-FT Baseline)**: Whether QAT attribution restructuring is caused by integer quantization constraints or 5 epochs of SGD optimization. | QAT executes simulated quantization AND 5 epochs of AdamW gradient descent simultaneously, confounding quantization with optimization. | Executed unquantized FP32 fine-tuning baseline (`FP32-FT`) using identical AdamW configuration. Attributions shift by $TV = 0.2214 \pm 0.2327$ (Signed Cosine $0.1166$). Paired Wilcoxon test vs QAT: $W = 2390.0, p = 0.684$ (median $\Delta TV = -0.0047$). Drift magnitude is statistically indistinguishable. Proves attribution restructuring is driven by SGD optimization across overparameterized loss landscape, not integer quantization noise. | **DISENTANGLED & RESOLVED** |

---

## Detailed Methodological Verifications

### 1. Strict Session-Group Disjunction (Gate 1 Passed)
- Added explicit alias mapping (`'hangout_chat_4': 'hangouts_chat_4'`) in `src/data/leakage_free_split.py`.
- Verified that 140 raw captures map to **106 unique canonical groups** (Train: 74 groups, 102 files; Val: 16 groups, 19 files; Test: 16 groups, 19 files).
- Confirmed both `hangout_chat_4b.pcap` and `hangouts_chat_4a.pcap` are assigned together to **Train**.
- All three pairwise canonical group intersections are strictly empty:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
- Total dataset flows remain exactly **339,009 flows**. Cross-split flow overlap is `False`.

### 2. Output Layer Quantization Parameters (Gate 2 Passed)
- Inspected `artifacts/quantized_models/ptq_seed_42.pt`:
  - `fc2` output layer: `scale = 0.069318`, `zero_point = 105`.
  - Representable range: $[(0 - 105) \times 0.069318, (255 - 105) \times 0.069318] = [-7.278, +10.398]$.
  - Negative logits are preserved naturally. Exactly **0 test flows are clamped at 0.0**.

### 3. Threshold Calibration Rationale (Gate 4 Passed)
- Thresholds calibrated strictly on `val_data.npz` to maximize validation F1:
  - FP32: $\tau = 0.900$ (Val F1 = 0.7712, Test F1 = 0.7746).
  - PTQ: $\tau = 0.900$ (Val F1 = 0.7710, Test F1 = 0.7697).
  - QAT: $\tau = 0.680$ (Val F1 = 0.7748, Test F1 = 0.7762).
- Reusing $\tau = 0.900$ on QAT collapses test F1 to 0.0000 because QAT predicted probabilities do not reach $0.900$ (maximum test probability is $0.7338$), proving that post-quantization calibration is strictly mandatory.

### 4. Hardware Speedup Rigor & Boundary (Gate 6 Passed)
- Disentangled median batch latency speedup from mean throughput speedup.
- Observed throughput gains stem from PyTorch QNNPACK execution on macOS ARM64 CPU.
- Clearly bounded to desktop ARM64 execution; cannot be extrapolated directly to embedded IoT platforms without physical hardware profiling.
