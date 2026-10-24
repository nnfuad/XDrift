# XDrift: Final Release Checklist & Reproducibility Gate

**Audit Date**: October 23, 2026  
**Project**: Investigating Attribution Drift in INT8-Quantized 1D CNNs for Encrypted Traffic Classification (XDrift)  
**Lead Auditor**: Senior Machine Learning Research Engineer  
**Repository Path**: `/Volumes/LEXAR/Explainability Drift/xdrift`  
**Gate Status**: **HOLD PENDING USER DECISION (GATE 1 TRAIN-VAL DEFECT IDENTIFIED; TEST ISOLATION VERIFIED)**  

---

## 1. Executive Gate Summary

This document certifies that the complete XDrift research pipeline, artifacts, and documentation have undergone an exhaustive reproducibility, consistency, and session-isolation audit. All reported metrics, counts, and architectural mechanisms have been independently reproduced from raw model binaries, dataset arrays, and forward execution hooks.

**Crucial Audit Finding**: Gate 1 is marked as **HOLD** because an unhandled typographical variation in the ISCX dataset (`hangout_chat_4b.pcap` vs. `hangouts_chat_4a.pcap`) caused the two communicating endpoints of Hangouts Chat Session 4 to be partitioned across Train (342 flows) and Validation (108 flows). Crucially, **the held-out Test split remains 100% isolated and uncontaminated** (0 overlapping files, 0 overlapping groups, 0 overlapping flows).

| Gate | Verification Area | Primary Evidence | Status |
| :---: | :--- | :--- | :---: |
| **Gate 1** | Leakage-Free Capture Splitting | Train-Val leakage found in 1 session (hangouts_chat_4); Test split strictly 100% isolated | **HOLD (DEFECT IDENTIFIED)** |
| **Gate 2** | Preprocessing & Flow Invariants | 120s timeout, canonical 5-tuple, 339,009 flows across splits; 7,018 VPN / 257,322 Non-VPN | **PASSED** |
| **Gate 3** | Architecture & Quantized Execution | 44,129 parameters; real `torch.qint8` weights & `torch.quint8` activations | **PASSED** |
| **Gate 4** | Predictive Performance Calibration | Validation-tuned thresholds ($\tau=0.91, 0.97, 0.66$); QAT F1 = 0.7402 | **PASSED** |
| **Gate 5** | Mechanistic Parameter Verification | Trace of `fc2` `torch.quint8` unsigned saturation clamp (58 flows; 53 negative in FP32) | **PASSED** |
| **Gate 6** | Attribution Cohort & Noise Baseline | Multi-seed 100-flow cohort (seeds 42, 43, 44); paired Wilcoxon $p = 0.345$ | **PASSED** |
| **Gate 7** | Disentangled Hardware Benchmarks | BS=1 streaming latency (no speedup) vs BS=64 batched (15.08× speedup) | **PASSED** |
| **Gate 8** | Epistemological Boundaries | 3 VPN test captures documented; zero unsubstantiated security claims | **PASSED** |

---

## 2. Gate-by-Gate Verification Details

### Gate 1: Leakage-Free Dataset Partitioning
- **Raw Capture Inventory**: 140 files (102 `.pcap`, 38 `.pcapng`; 25.78 GB total). Verified in `data/manifests/dataset_manifest.json`.
- **Session Group Heuristic & Defect Analysis**:
  - The production heuristic in `src/data/leakage_free_split.py` stripped terminal `_a`, `_b`, `_A`, `_B` suffixes.
  - While it successfully grouped 32 dual-ended pairs, it failed to normalize an asymmetric singular/plural naming convention in ISCX NonVPN-PCAPs-02:
    - `hangout_chat_4b.pcap` (singular "hangout") $\rightarrow$ mapped to group ID `'hangout_chat_4'` (assigned to **Train**; 342 flows).
    - `hangouts_chat_4a.pcap` (plural "hangouts") $\rightarrow$ mapped to group ID `'hangouts_chat_4'` (assigned to **Validation**; 108 flows).
  - Packet timestamp and IP analysis confirms they are two endpoints of the same session recorded 3.45 seconds apart on April 29, 2015 (`131.202.240.242` $\leftrightarrow$ `131.202.240.87`).
  - Therefore, **genuine session leakage occurred between Train and Validation** (1 session, 108 validation flows = 0.20% of validation).
- **Test Split Isolation**:
  - `test_manifest.json` contains 21 files and 16 session groups.
  - Neither `hangout_chat_4b` nor `hangouts_chat_4a` appears in `test_manifest.json` or `test_data.npz`.
  - Canonical disjunction: $\text{TrainCanonical} \cap \text{TestCanonical} = \emptyset$, $\text{ValCanonical} \cap \text{TestCanonical} = \emptyset$.
  - **The test split ($N=20,971$ flows) is 100% isolated and uncontaminated**.
- **Split Manifests**: Verified in `data/manifests/{train,val,test}_manifest.json`.

---

### Gate 2: Preprocessing Pipeline & Flow Invariants
- **Flow Construction Rules**:
  - Transport 5-tuple bidirectional canonicalization: Complete $(IP, port)$ endpoint pairs ordered deterministically.
  - Chronological sorting of packets before flow segmentation.
  - Inactivity timeout: 120.0 seconds strictly enforced.
  - Maximum sequence length: 64 packets with right-zero padding.
  - Features: Channel 1 = Directional IP length $d_t \times \min(S_t / 1500, 1.0)$; Channel 2 = Log IAT $\ln(1 + \min(\Delta t_t, 5.0))$.
  - Validity masks: $M_t = 1$ for observed packets, $0$ for padded positions.
- **Processed Datasets (Reconciled from Raw NPZ Arrays)**:
  - `data/processed/train_data.npz`: $X \in \mathbb{R}^{264340 \times 2 \times 64}$, $y \in \{0, 1\}^{264340}$ (**7,018 VPN, 257,322 Non-VPN**).
  - `data/processed/val_data.npz`: $X \in \mathbb{R}^{53698 \times 2 \times 64}$, $y \in \{0, 1\}^{53698}$ (**4,328 VPN, 49,370 Non-VPN**).
  - `data/processed/test_data.npz`: $X \in \mathbb{R}^{20971 \times 2 \times 64}$, $y \in \{0, 1\}^{20971}$ (**2,693 VPN, 18,278 Non-VPN**).
  - Total: 339,009 flows (**14,039 VPN, 324,970 Non-VPN**). All 339,009 flow IDs are completely disjoint across splits.
- **Unit Test Suite**: All 14 unit tests in `tests/` executed and passed (`pytest tests/ -v`).

---

### Gate 3: Neural Network Architecture & Quantized Execution Traceability
- **Architecture**: Compact 1D-CNN with 3 Conv-BN-ReLU-Pool blocks and AdaptiveAvgPool1d(1).
- **Parameter Count**: Exactly **44,129 parameters** verified.
- **Model Checkpoints**:
  - FP32 Baseline: `artifacts/checkpoints/fp32_seed_42.pt` (Best epoch: 4, Val Loss: 1.7758, Train time: 666.1s, $\text{pos\_weight} = 36.6660$).
  - INT8-PTQ: `artifacts/quantized_models/ptq_seed_42.pt` (Calibrated on 1,000 training flows).
  - INT8-QAT: `artifacts/quantized_models/qat_seed_42.pt` (Fine-tuned for 5 epochs on CPU, Best epoch: 5, Val Loss: 7.6792).
- **Execution Verification**: Forward passes execute real `torch.qint8` integer weights and `torch.quint8` integer activations with `torch.ao.nn.intrinsic.quantized.ConvReLU1d` and `torch.ao.nn.quantized.Linear` modules under the `qnnpack` backend.

---

### Gate 4: Predictive Performance & Threshold Calibration
- **Held-Out Test Set**: Exactly 20,971 flows (2,693 VPN, 18,278 Non-VPN).
- **Threshold Calibration Protocol**:
  - In FP32 training, $\text{pos\_weight} = 36.6660$ pushed validation-optimal threshold to $\tau = 0.900$.
  - Reusing $\tau = 0.900$ on QAT produced an apparent all-negative classifier because QAT test probabilities peak at $0.6653$ due to logit contraction.
  - Thresholds calibrated independently on validation data:
    - FP32: $\tau = 0.910 \implies$ Accuracy: 85.70%, Balanced Accuracy: 91.64%, F1: 0.6416, Precision: 47.31%, Recall: 99.63%.
    - PTQ: $\tau = 0.970 \implies$ Accuracy: 84.14%, Balanced Accuracy: 77.07%, F1: 0.5224, Precision: 42.59%, Recall: 67.55%.
    - QAT: $\tau = 0.660 \implies$ Accuracy: **91.13%**, Balanced Accuracy: **94.22%**, F1: **0.7402**, Precision: 59.33%, Recall: 98.37%.
- **Ranking Metrics (Threshold-Independent)**:
  - FP32: PR-AUC = 0.3834, ROC-AUC = 0.8806, Brier = 0.5000.
  - PTQ: PR-AUC = 0.3345, ROC-AUC = 0.7554, Brier = 0.7746.
  - QAT: PR-AUC = **0.5932**, ROC-AUC = **0.9490**, Brier = **0.0656**.

---

### Gate 5: Mechanistic Verification of Output Quantization Parameters
- **Forward Hook and Trace Evidence**:
  - Executed a forward hook on `fc2` across all 20,971 test flows.
  - Output tensor properties:
    - Module class: `torch.ao.nn.quantized.modules.linear.Linear`
    - Tensor `dtype`: `torch.quint8` (Unsigned 8-bit integer, range $[0, 255]$).
    - Tensor `qscheme`: `torch.per_tensor_affine`.
    - Quantization parameters: $\text{scale} = 0.076921858$, $\text{zero\_point} = 0$.
    - Weight tensor properties: `weight().dtype = torch.qint8` (Signed 8-bit integer, range $[-128, 127]$).
  - Graph Topology: Forward definition is `x = self.fc2(x); return self.dequant(x)`. No ReLU, BatchNorm, or explicit `torch.clamp` exists after `fc2`.
  - Saturation Mechanism: Inside `nnq.Linear`, the real-valued accumulator $y$ is mapped to integer $q$ via the hardware affine quantization clamp:
    $$q = \text{clamp}\left(\left\lfloor \frac{y}{\text{scale}} + 0.5 \right\rfloor + \text{zero\_point},\, 0,\, 255\right)$$
    Because activations are unsigned `torch.quint8` and $\text{zero\_point} = 0$, any negative accumulator $y < 0$ is saturated to $q = 0$.
  - Dequantization Formula: `DeQuantStub()` evaluates $z = \text{scale} \times (q - 0) = 0.076921858 \times q$. Since $q \in [0, 255]$, $z \ge 0.0000$ identically for all possible inputs, bounding sigmoid probabilities to $\ge \sigma(0.0) = 0.5000$.
  - Exact Flow Analysis on Test Set ($N=20,971$):
    - Exactly **58 flows (0.2766%)** produce integer output $q = 0$ ($z = 0.0000, p = 0.5000$).
    - Comparison against baseline FP32 logits for these 58 flows:
      - 53 flows (91.4%) had negative FP32 logits (range $[-1.6797, -0.0537]$, mean $-0.4390$), demonstrating that their negative pre-quantization outputs were clamped to zero by the unsigned integer saturation boundary.
      - 5 flows (8.6%) had small positive FP32 logits ($+0.0232$ to $+0.0521$), which rounded down to 0 because they fell below the half-step rounding threshold ($\text{scale}/2 \approx 0.0385$).
    - In FP32, exactly 59 test flows had negative logits ($0.28\%$).
- **Epistemological Distinction**:
  - *Verified Fact*: Saturation clamping at $q=0$ on unsigned `torch.quint8` activations bounds the lower tail at $0.0000$ on 0.28% of test flows.
  - *Plausible Mechanism*: The broader upward distribution shift (+1.51 mean logit shift across the test set) is caused by cumulative affine calibration drift across intermediate Conv1D and MaxPool layers.

---

### Gate 6: Attribution Cohort Traceability & Explainer Repeatability
- **Cohort Composition**: 100 balanced test flows (50 VPN, 50 Non-VPN).
  - Fully traced to test set indices in `results/metrics/cohort_indices.json`.
  - Capture distribution:
    - VPN (50 flows): `vpn_hangouts_audio2` (40 flows = 80%), `vpn_bittorrent` (8 flows = 16%), `vpn_netflix_A` (2 flows = 4%).
    - Non-VPN (50 flows): `scp1` (26 flows = 52%), `email1b` (11 flows = 22%), `email1a` (9 flows = 18%), `email2b` (1 flow), `icq_chat_3b` (1 flow), `email2a` (1 flow), `vimeo4` (1 flow).
- **Multi-Seed Repeatability Benchmark**:
  - Evaluated KernelSHAP with seeds 42, 43, 44 on identical 100 flows explaining pre-sigmoid VPN logit.
  - Mean explainer noise: Top-5 Jaccard = **0.9066**, Total Variation = **0.1050**, Signed Cosine = **0.9258**.
- **Paired Hypothesis Tests on Identical Flows**:
  - FP32 vs. PTQ: Mean Top-5 Jaccard = **0.9431** (median 1.0, 89/100 flows identical).
  - Paired $\Delta TV = TV_{\text{PTQ}} - TV_{\text{Noise}} = -0.0157 \pm 0.1716$.
  - Wilcoxon signed-rank test: $W = 1482.0, \mathbf{p = 0.34516}$; Bootstrap 95% CI: $[-0.0494, +0.0166]$ (spans 0).
  - Decision: **PTQ attribution drift is statistically indistinguishable from KernelSHAP sampling noise on this cohort**.
  - FP32 vs. QAT: Paired $\Delta TV = +0.2382 \pm 0.1654$; Wilcoxon $W = 4491.0, \mathbf{p = 8.80 \times 10^{-14}}$.
  - Decision: **QAT induces genuine, statistically significant attribution restructuring**.

---

### Gate 7: Disentangled Hardware Benchmarks
- **Environment**: macOS Apple Silicon ARM64, CPU QNNPACK execution, 50 warmup runs, 500 repetitions.
- **Edge Streaming Latency (Batch Size = 1)**:
  - FP32: Median 0.273 ms (Mean 0.282 ms, p95 0.333 ms, Throughput 3,549 flows/s).
  - PTQ: Median 0.405 ms (Mean 0.416 ms, p95 0.477 ms, Throughput 2,404 flows/s; 0.68× slower).
  - QAT: Median 0.288 ms (Mean 0.285 ms, p95 0.307 ms, Throughput 3,504 flows/s; 0.95× parity).
  - *Observation*: Quantization provides no speedup at BS=1 due to runtime invocation overhead.
- **Server Batched Ingestion (Batch Size = 64)**:
  - FP32: Median 13.169 ms (Throughput 4,782 flows/s).
  - PTQ: Median 0.932 ms (Throughput 65,650 flows/s; 13.73× speedup).
  - QAT: Median 0.873 ms (Throughput **72,456 flows/s**; **15.08× speedup**).
  - *Observation*: Integer vectorization via ARM64 NEON SIMD instructions dominates at BS=64.
- **Storage vs. Memory**: Model checkpoint reduced by **3.55×** (183.86 KB down to 51.77 KB). Peak runtime process RAM is ~150 MB (dominated by PyTorch runtime).

---

### Gate 8: Epistemological Boundaries & Preserved Limitations
- **Capture Sample Limitation**: The held-out test split contains only **3 independent VPN capture files**. The 100-flow attribution cohort has **40 of 50 VPN flows from a single capture**.
- **Uncertainty Estimates**: Flow-level bootstrap CIs are artificially narrow. Group-clustered bootstrap CIs are wider (+148% to +5,600%) and fragile with so few VPN groups.
- **Claims Boundary**:
  - Findings apply strictly to the **ISCX 2016 benchmark test split** and the evaluated 1D-CNN architecture.
  - Zero claims of universal generalization to modern QUIC, HTTP/3, or enterprise traffic.
  - Zero claims of exploitable firewall vulnerabilities or security breaches.
  - Zero claims of legal or regulatory compliance.

---

## 3. Disputed Items Resolution & Provable Evidence Matrix

| Disputed Item | Original Claim / Issue | Artifact / Test Used for Verification | Observed Empirical Evidence | Resulting Conclusion & Action Taken | Affected Reports & Tables |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Item 1: PTQ Output Clamping Mechanism** | Checklist stated `fc2` zero-point $=0$ mathematically truncates negative logits, without addressing that `torch.qint8` is signed. | Runtime hook on `fc2` output tensor and inspection of exported graph under QNNPACK. | `fc2` activations are unsigned `torch.quint8` ($[0, 255]$), whereas weights are signed `torch.qint8`. QNNPACK internal affine saturation clamp forces accumulators $< 0$ to $q=0$. Exactly 58 test flows produce $q=0$; 53 had negative FP32 logits (mean $-0.44$). | Replaced generic zero-point claim with the verified `torch.quint8` activation saturation mechanism. Truncation verified on 58 flows (0.28%); broader upward shift explicitly attributed to intermediate-layer calibration drift. | `final_release_checklist.md`, `final_consistency_audit.md`, `final_research_report.md` |
| **Item 2: Training Label Counts Discrepancy** | Earlier text quoted 6,996 VPN and 257,344 Non-VPN flows; another audit quoted 7,018 and 257,322 for the same 264,340 training flows. | Direct loading of `data/processed/train_data.npz` and inspection of `artifacts/checkpoints/fp32_seed_42.pt`. | Array `y` has exactly 7,018 ones (VPN) and 257,322 zeros (Non-VPN). Checkpoint `pos_weight` is $36.666000366$, which matches $257,322 / 7,018 = 36.6660017$ exactly (whereas $257,344 / 6,996 = 36.7844$). | Discrepancy identified as a typographical drafting error in step 406. Ground-truth verified as 7,018 VPN and 257,322 Non-VPN. Updated all checklist and audit documents to the true counts. | `final_release_checklist.md`, `final_consistency_audit.md`, `final_research_report.md` |
| **Item 3: Dataset Provenance & Split Disjointness** | Needed to verify that all 339,009 flows and 107 groups are disjoint and match split manifests. | Cross-split verification script loading all 3 NPZ files and JSON manifests. | Total flows: $264,340 + 53,698 + 20,971 = 339,009$. File overlap: 0. Group overlap: 0. Flow ID overlap: 0. Test capture IDs in data match test manifest exactly. | Provenance confirmed. All 339,009 flows accounted for exactly once. Cohort flow IDs match test set indices exactly (saved to `results/metrics/cohort_indices.json`). | `final_release_checklist.md`, `cohort_indices.json`, `master_pipeline_summary.json` |
| **Item 4: Execution vs. Output Distinction** | Checklist must distinguish tests that actually ran successfully from steps that merely started or produced output. | Execution logs, exit codes, and `pytest tests/ -v` test runner. | All 14 unit tests in `tests/` completed with exit code 0 (`14 passed in 2.06s`). Forward traces completed on all 20,971 test flows. Repeatability cohort ran 3 seeds to full completion. | Documented that all unit tests and empirical traces executed to termination with verified exit code 0. | `final_release_checklist.md` |
| **Item 5: Split Group Counts Discrepancy** | Previous response in Step 494 quoted 72 train, 14 val, and 21 test groups; manifest audit reports 75, 16, and 16 groups. | Manifest SHA-256 hashes, creation timestamps (`17:12:07`), and [`preprocessing_validation_report.md`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/reports/preprocessing_validation_report.md#L30-L36). | Manifests were generated simultaneously at 17:12:07 and have never been modified. Step 494 conflated test set file count (21 files) with group count (16 groups) and split remaining 86 groups into 72 and 14 in descriptive text. Original report from 17:25 confirms 75/16/16 ground truth. | Provenance verified: Train has 75 groups, Val has 16 groups, Test has 16 groups. Discrepancy documented as a step-494 conversational hallucination. | `final_release_checklist.md`, `final_consistency_audit.md` |
| **Item 6: Session-Group Overlap (hangouts_chat_4)** | Train contained `hangout_chat_4` and Val contained `hangouts_chat_4`; audit required verifying whether this represents cross-split leakage. | Packet timestamp and IP analysis of raw PCAPs (`hangouts_chat_4a.pcap` vs `hangout_chat_4b.pcap`) and canonical regression test. | In ISCX NonVPN-PCAPs-02, Endpoint A is spelled `hangouts_chat_4a` (plural) and Endpoint B is `hangout_chat_4b` (singular). The grouping function failed to normalize this typo, generating 2 groups instead of 1. Consequently, Endpoint B (342 flows) went to Train and Endpoint A (108 flows) to Val. Packet timestamps (April 29, 2015, 3.45s gap) confirm they are two endpoints of the same session. Leakage is strictly between Train and Val; Test split is 100% disjoint. | Identified as a grouping function defect on asymmetric dataset filenames. Marked Gate 1 on HOLD. Test set results (Phases 5–9) remain valid because neither endpoint touched Test. | `final_release_checklist.md`, `final_consistency_audit.md`, `test_preprocessing.py` |

---

## 4. Master Release Traceability Matrix

| Artifact Category | Primary Repository Path | Verified Checkpoint / File Hash | Last Audited |
| :--- | :--- | :--- | :---: |
| **FP32 Baseline Checkpoint** | `artifacts/checkpoints/fp32_seed_42.pt` | File size: 188,273 bytes, Seed: 42, pos_weight: 36.6660 | Oct 23, 2026 |
| **INT8-PTQ Converted Model** | `artifacts/quantized_models/ptq_seed_42.pt` | File size: 53,017 bytes, Engine: QNNPACK | Oct 23, 2026 |
| **INT8-QAT Converted Model** | `artifacts/quantized_models/qat_seed_42.pt` | File size: 53,017 bytes, Engine: QNNPACK | Oct 23, 2026 |
| **Processed Training Data** | `data/processed/train_data.npz` | 264,340 flows (7,018 VPN, 257,322 Non-VPN) | Oct 23, 2026 |
| **Processed Validation Data** | `data/processed/val_data.npz` | 53,698 flows (4,328 VPN, 49,370 Non-VPN) | Oct 23, 2026 |
| **Processed Test Flows** | `data/processed/test_data.npz` | 20,971 flows (2,693 VPN, 18,278 Non-VPN) | Oct 23, 2026 |
| **Per-Example Test Predictions** | `results/predictions/test_predictions.npz` | 20,971 predictions across 3 models | Oct 23, 2026 |
| **Attribution Cohort Tensors** | `results/attributions/cohort_attributions.npz` | 100 flows, 128 SHAP features per flow | Oct 23, 2026 |
| **Cohort Index Mapping** | `results/metrics/cohort_indices.json` | 100 items mapping cohort index to test flow | Oct 23, 2026 |
| **Multi-Seed Repeatability Tensors** | `results/attributions/explainer_repeatability_cohort.npz` | Seeds 42, 43, 44 on identical 100 flows | Oct 23, 2026 |
| **Master Summary Records** | `results/metrics/master_pipeline_summary.json` | Reconciled JSON matching all tables & traces | Oct 23, 2026 |
| **Audited Table 1 (Predictive)** | `results/tables/table1_predictive_performance.md` | Panels A, B, C | Oct 23, 2026 |
| **Audited Table 2 (Attribution)** | `results/tables/table2_attribution_drift.md` | Panels A, B, C | Oct 23, 2026 |
| **Audited Table 3 (Hardware)** | `results/tables/table3_hardware_benchmarks.md` | Panels A, B, C | Oct 23, 2026 |
| **Audited Table 4 (Hypotheses)** | `results/tables/table4_hypothesis_matrix.md` | Audited formal hypotheses H1-H4 | Oct 23, 2026 |
| **Scientific Consistency Audit** | `reports/final_consistency_audit.md` | Comprehensive 9-section audit | Oct 23, 2026 |
| **Reconciled Final Report** | `reports/final_research_report.md` | Complete manuscript | Oct 23, 2026 |

---

## 5. Certification Statement & Hold Status

**Status: RELEASE HOLD (GATE 1 DEFECT DETECTED)**

I certify that:
1. Every numerical metric reported in `reports/final_research_report.md` is mathematically reproduced from the underlying data and matches `results/metrics/master_pipeline_summary.json` and the audited tables without exception.
2. The output-clamping mechanism for PTQ `fc2` has been traced to the unsigned `torch.quint8` activation representation under QNNPACK and verified empirically against the 58 clamped flows (53 negative in FP32) and intermediate layer drift.
3. The training-label counts have been reconciled directly against `data/processed/train_data.npz` and checkpoint metadata as **7,018 VPN and 257,322 Non-VPN flows** (resolving an earlier drafting typo).
4. **Gate 1 Defect Identified**: The held-out Test split ($N=20,971$) is 100% strictly disjoint from both Train and Validation. However, a 1-session dual-ended endpoint leakage defect between Train and Validation (`hangout_chat_4b` in Train, `hangouts_chat_4a` in Val; 108 validation flows) has been diagnosed, reproduced via regression test, and documented.
5. All 14 original unit tests pass, and the canonical session-group regression test correctly fails on the current manifests as expected until the user decides whether to repartition or proceed.
6. All limitations regarding capture session sample size (3 VPN test captures), cohort concentration (40 of 50 from 1 capture), and benchmark scope are explicitly preserved.

The research release certification is placed on **HOLD** pending user guidance on the Train-Validation defect.
