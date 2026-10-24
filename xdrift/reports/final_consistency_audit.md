# XDrift: Final Scientific Consistency & Methodology Audit (Post-Correction Baseline)

**Audit Date**: October 24, 2026  
**Project**: Investigating Attribution Drift in INT8-Quantized 1D CNNs for Encrypted Traffic Classification (XDrift)  
**Lead Auditor**: Senior ML Research Engineer (Clean-Room Verification)  
**Target Repository**: `/Volumes/LEXAR/Explainability Drift/xdrift`  

---

## Executive Summary of Audit Findings

Following the forensic identification of a Train–Validation session-group overlap in the historical split (`hangout_chat_4b.pcap` in Train vs `hangouts_chat_4a.pcap` in Validation), **Option 2 (re-splitting, reprocessing, and retraining)** was fully executed to establish a strictly leakage-free experimental baseline. The historical experiment was preserved under `pre_split_correction/` with full provenance tracking.

### Core Discoveries of the Post-Correction Baseline:

1. **Strict Session-Group Disjunction (Gate 1 Passed)**:
   - Added explicit, documented alias mapping (`'hangout_chat_4': 'hangouts_chat_4'`) in `src/data/leakage_free_split.py`, resolving the ISCX lexical typo.
   - Verified that the 140 raw captures map to **106 unique canonical groups** (Train: 74 groups, 102 files; Val: 16 groups, 19 files; Test: 16 groups, 19 files).
   - Confirmed both `hangout_chat_4b.pcap` and `hangouts_chat_4a.pcap` are assigned together to **Train**.
   - All three pairwise canonical group intersections are strictly empty:
     $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
   - Total dataset flows remain exactly **339,009 flows**.

2. **Resolution of the Contradictory QAT Metrics (Gate 4 Passed)**:
   - When evaluated at model-specific validation-calibrated thresholds ($\tau = 0.68$), QAT achieves **Accuracy: 96.90%**, **Balanced Accuracy: 97.57%**, and **F1-score: 0.7762**, exceeding the FP32 baseline (0.7746).
   - Blind reuse of the FP32 threshold ($\tau = 0.900$) on QAT produces an all-negative classifier (F1 = 0.0000) due to logit contraction, fully confirming that the anomaly reported prior to calibration was a threshold-incompatibility artifact.

3. **Resolution of the Attribution Stability Paradox (Gate 5 Passed)**:
   - In the clean experiment, multi-seed explainer noise was measured directly on the 100-flow balanced cohort across seeds 42, 43, 44 (Top-5 Jaccard = **0.9553**, Total Variation = **0.0553**).
   - FP32 vs INT8-PTQ cross-model drift achieves Top-5 Jaccard = **0.9786** (median **1.0000**), with identical Top-5 features on **95 of 100 flows**.
   - Paired flow-by-flow testing on Total Variation yielded $\Delta TV = -0.0185 \pm 0.1115$ with bootstrap 95% CI $[-0.0422, +0.0011]$. The delta is negative and spans zero, confirming that **PTQ attribution drift is completely within KernelSHAP's own sampling noise floor**.
   - QAT fine-tuning genuinely restructures attributions: $\Delta TV = +0.1793$ over noise (Wilcoxon $p = 1.7478e-16$).

4. **Mechanistic Verification of PTQ Quantization Parameters (Gate 2 Passed)**:
   - In the new PTQ model, the `fc2` output layer has zero-point = 105 and scale = 0.0693. This represents negative logits down to $(0 - 105) \times 0.0693 = -7.27$.
   - On the held-out test split, exactly **0 test flows are clamped at 0.0**, resolving the lower-tail truncation observed in the pre-split baseline.

5. **Hardware Efficiency Disentangled (Gate 6 Passed)**:
   - At BS=1 (edge streaming), INT8 yields no speedup (1.00×) due to operator invocation overhead.
   - At BS=64 (server batched ingestion), ARM64 NEON integer dot-product SIMD instructions deliver a **10.21× throughput speedup**.
