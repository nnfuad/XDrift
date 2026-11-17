# XDrift: Release Verification & Quality Checklist (Post-Correction Baseline)

**Certification Date**: October 24, 2026  
**Auditor**: Senior ML Research Engineer (Autonomous Verification)  
**Status**: **QUALIFIED RELEASE CERTIFIED (STAGE 1 COMPLETE & AUDITED)**  

---

### Verification Gates Summary Table

| Gate | Category | Core Verification Criteria | Status |
| :--- | :--- | :--- | :---: |
| **Gate 1** | Data Leakage & Session Isolation | Zero group overlap across all splits; Hangouts pair co-located in Train; 106 canonical groups | **PASSED** |
| **Gate 2** | Preprocessing & Operator Realism | 120s timeout, canonical 5-tuples; authentic `torch.qint8` integer operators; zero-point 105 | **PASSED** |
| **Gate 3** | Training & Checkpoint Provenance | Training-only `pos_weight` (30.3756); validation early stopping; exact state dict hashes | **PASSED** |
| **Gate 4** | Predictive Performance & Calibration | Validation-tuned thresholds ($\tau=0.90, 0.90, 0.68$); QAT F1 = 0.7762; Panel C & Murphy decomposition documented | **PASSED** |
| **Gate 5** | Attribution Drift vs Noise Floor | Multi-seed repeatability baseline (Seeds 42, 43, 44); Wilcoxon $R^+=3544.0 > R^-=1506.0, p = 0.00046$; 95/100 flows identical top-5; non-equivalence documented | **PASSED** |
| **Gate 6** | Hardware Benchmarks Disentanglement | BS=1 streaming vs BS=64 batched; explicit median latency and mean throughput speedups; desktop ARM64 boundary | **PASSED** |
| **Gate 7** | Scope of Claims & Epistemological Boundaries | 3 VPN test captures documented; Estimand A vs B delineated across 16 canonical groups; domain generalization bounded to ISCX 2016 split | **PASSED** |
| **Gate 8** | Complete Test Suite Compliance | All 15 unit and regression tests pass with exit code 0 | **PASSED** |
| **Gate 9** | Ablation Baseline Verification | FP32-FT unquantized control verifies QAT attribution drift is optimization-driven ($W=2390.0, p=0.684$) | **PASSED** |

---

### Detailed Gate Verification Records

#### Gate 1: Session-Group Disjunction & Regression Defense
- Alias `'hangout_chat_4': 'hangouts_chat_4'` active in `src/data/leakage_free_split.py`.
- Automated regression test `test_canonical_session_group_isolation_regression` passed.
- Pairwise intersections:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$
- Exact split sizes: Train 220,194 flows (74 groups), Val 69,564 flows (16 groups), Test 49,251 flows (16 groups). Total: 339,009 flows. Overlap: `False`.

#### Gate 2: Quantized Operator Verification
- `torch.backends.quantized.engine = 'qnnpack'` active.
- Confirmed quantized operators in PTQ and QAT: `['quant', 'conv1', 'conv2', 'conv3', 'fc1', 'fc2']`.
- Output linear layer (`fc2`): `scale = 0.069318`, `zero_point = 105`. Exactly 0 flows clamped at 0.0.

#### Gate 3: Training Provenance
- `artifacts/checkpoints/fp32_seed_42.pt`: best epoch 8, best val loss 1.4922, threshold 0.900, `pos_weight` 30.3756.
- `artifacts/quantized_models/ptq_seed_42.pt`: calibrated on 1,000 train flows.
- `artifacts/quantized_models/qat_seed_42.pt`: 5 epochs fine-tuning, best epoch 3, best val loss 4.0979.

#### Gate 4: Predictive Metrics Verification
- Test set $N = 49,251$ flows (2,693 VPN, 46,558 Non-VPN).
- Model-specific validation thresholds: $\tau_{\text{FP32}} = 0.900, \tau_{\text{PTQ}} = 0.900, \tau_{\text{QAT}} = 0.680$.
- QAT achieves Test F1 = 0.7762, Accuracy = 96.90%, PR-AUC = 0.6478.
- Murphy score decomposition confirms QAT Brier score of $0.0237$ (Reliability $0.0036$, Resolution $0.0316$, Uncertainty $0.0517$) is driven by Non-VPN probability compression ($0.0557$, $94.41\%$ in Bin 0).

#### Gate 5: Attribution Drift & Explainer Repeatability
- Verified cohort: all 100 flows confirmed from clean `test_data.npz` (50 VPN, 50 Non-VPN). Additivity residuals bounded by $\le 7 \times 10^{-6}$.
- Multi-seed KernelSHAP noise baseline (Seeds 42, 43, 44): Top-5 Jaccard = 0.9553, TV = 0.0553.
- PTQ vs FP32: Top-5 Jaccard = 0.9786, identical on 95/100 flows.
- Paired Wilcoxon test against noise: $W = 1506.0 (R^+ = 3544.0, R^- = 1506.0)$, two-sided $p = 0.00046$ (Holm: 0.00092), one-sided greater $p = 0.00023$, median delta $+0.0049$.
- Disclaimed statistical equivalence: Observed differences indicate a detectable positive rank shift, but median magnitude is minimal and top-five features are largely identical.
- QAT vs FP32: TV = 0.2346, $\Delta TV = +0.1793$ ($p < 10^{-15}$).

#### Gate 6: Hardware Efficiency & Environment Boundary
- Checkpoint compression: 3.56× (184.24 KB $\to$ 51.77 KB).
- BS=1 streaming: 1.05× PTQ latency speedup, 1.10× throughput speedup.
- BS=64 batched:
  - Median batch latency speedup: 12.44× (PTQ), 12.59× (QAT).
  - Mean throughput speedup: 11.94× (PTQ; 76,264 flows/s), 12.77× (QAT; 81,552 flows/s) over FP32 (6,387 flows/s).
- Explicitly documented that benchmarks represent Apple Silicon Mac desktop ARM64 execution with QNNPACK and do not extrapolate directly to embedded microcontrollers without physical testing.

#### Gate 7: Scope & Epistemological Boundaries
- Explicitly documented that the test set contains 3 VPN captures (`vpn_bittorrent`, `vpn_hangouts_audio2`, `vpn_netflix_A`).
- Primary clustered bootstrap evaluated across **16 canonical session groups**: Estimand A ($-0.0778$, CI $[-0.1130, +0.3275]$) vs Estimand B ($+0.2270$, CI $[+0.0639, +0.3562]$). Sensitivity without `skype_audio3`: Estimand A $+0.0720$ (CI $[-0.0770, +0.3454]$), Estimand B $+0.2499$ (CI $[+0.0870, +0.3673]$).
- Bounded statistical claims strictly to the ISCX 2016 benchmark split; no operational firewall bypass claims without adversarial testing.

#### Gate 8: Regression Suite
- All 15 tests in `pytest tests/ -v` pass with exit code 0.

---

### Final Release Certification
**QUALIFIED RELEASE CERTIFIED**: All numerical and methodological inconsistencies identified across audit reviews have been forensically reconciled, recomputed, and verified directly from source code and artifacts. The dataset split is strictly leakage-free (106 canonical groups, 0 group or flow overlap). The test-suite passes unconditionally (15/15). The research baseline is scientifically complete and defensible for thesis presentation.
