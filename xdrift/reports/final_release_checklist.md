# XDrift: Release Verification & Quality Checklist (Post-Correction Baseline)

**Certification Date**: October 24, 2026  
**Auditor**: Senior ML Research Engineer (Autonomous Verification)  
**Status**: **ALL GATES PASSED — RELEASE CERTIFIED**  

---

### Verification Gates Summary Table

| Gate | Category | Core Verification Criteria | Status |
| :--- | :--- | :--- | :---: |
| **Gate 1** | Data Leakage & Session Isolation | Zero group overlap across all splits; Hangouts pair co-located in Train | **PASSED** |
| **Gate 2** | Preprocessing & Operator Realism | 120s timeout, canonical 5-tuples; authentic `torch.qint8` integer operators | **PASSED** |
| **Gate 3** | Training & Checkpoint Provenance | Training-only `pos_weight` (30.3756); validation early stopping (epoch 8) | **PASSED** |
| **Gate 4** | Predictive Performance Calibration | Validation-tuned thresholds ($	au=0.90, 0.90, 0.68$); QAT F1 = 0.7762 | **PASSED** |
| **Gate 5** | Attribution Drift vs Noise Floor | Multi-seed repeatability baseline (Seeds 42, 43, 44); PTQ within noise floor | **PASSED** |
| **Gate 6** | Hardware Benchmarks Disentanglement | BS=1 streaming vs BS=64 batched throughput; disk footprint reduction 3.56× | **PASSED** |
| **Gate 7** | Scope of Claims & Epistemological Boundaries | 3 VPN test captures documented; no unproven operational security claims | **PASSED** |
| **Gate 8** | Complete Test Suite Compliance | All 15 unit and regression tests pass with exit code 0 | **PASSED** |

---

### Detailed Gate Verification Records

#### Gate 1: Session-Group Disjunction & Regression Defense
- Alias `'hangout_chat_4': 'hangouts_chat_4'` active in `src/data/leakage_free_split.py`.
- Automated regression test `test_canonical_session_group_isolation_regression` passed.
- Pairwise intersections:
  $$\text{Train} \cap \text{Val} = \emptyset, \quad \text{Train} \cap \text{Test} = \emptyset, \quad \text{Val} \cap \text{Test} = \emptyset$$

#### Gate 2: Quantized Operator Verification
- `torch.backends.quantized.engine = 'qnnpack'` active.
- Confirmed quantized operators in PTQ and QAT: `['quant', 'conv1', 'conv2', 'conv3', 'fc1', 'fc2']`.
- Output linear layer (`fc2`): `scale = 0.069318`, `zero_point = 105`.

#### Gate 3: Training Provenance
- `artifacts/checkpoints/fp32_seed_42.pt`: best epoch 8, best val loss 1.4922, threshold 0.900, `pos_weight` 30.3756.
- `artifacts/quantized_models/ptq_seed_42.pt`: calibrated on 1,000 train flows.
- `artifacts/quantized_models/qat_seed_42.pt`: 5 epochs fine-tuning, best epoch 3, best val loss 4.0979.

#### Gate 4: Predictive Metrics Verification
- Test set $N = 49,251$ flows (2,693 VPN, 46,558 Non-VPN).
- Model-specific validation thresholds: $\tau_{\text{FP32}} = 0.900, \tau_{\text{PTQ}} = 0.900, \tau_{\text{QAT}} = 0.680$.
- QAT achieves Test F1 = 0.7762, Accuracy = 96.90%, PR-AUC = 0.6478.

#### Gate 5: Attribution Drift & Explainer Repeatability
- Multi-seed KernelSHAP noise baseline (Seeds 42, 43, 44 on identical cohort $N=100$): Top-5 Jaccard = 0.9553, TV = 0.0553.
- PTQ vs FP32: Top-5 Jaccard = 0.9786, identical on 95/100 flows.
- Paired Wilcoxon test against noise: $\Delta TV = -0.0185$, 95% CI $[-0.0422, +0.0011]$ (spans 0). PTQ drift is within explainer noise floor.
- QAT vs FP32: TV = 0.2346, $\Delta TV = +0.1793$ ($p < 10^{-15}$).

#### Gate 6: Hardware Efficiency
- Checkpoint compression: 3.56× (184.24 KB $\to$ 51.77 KB).
- BS=1 latency: 0.254 ms (FP32) vs 0.293 ms (PTQ) vs 0.254 ms (QAT) — no latency advantage.
- BS=64 throughput: 6,282 flows/s (FP32) vs 72,090 flows/s (PTQ; 10.21× speedup) vs 60,107 flows/s (QAT; 8.53× speedup).

#### Gate 7: Scope & Epistemological Boundaries
- Explicitly documented that the test set contains 3 VPN captures (`vpn_bittorrent`, `vpn_hangouts_audio2`, `vpn_netflix_A`).
- No claims of operational security vulnerabilities without active adversarial testing.

#### Gate 8: Regression Suite
- All 15 tests in `pytest tests/ -v` pass.

---

### Final Release Certification
**CERTIFIED**: The experimental pipeline, data splits, models, attribution benchmarks, and documentation satisfy all rigor requirements. Option 2 execution is complete with zero remaining session leakage.
