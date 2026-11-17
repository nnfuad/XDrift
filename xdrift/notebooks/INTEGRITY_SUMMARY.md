# XDrift Experimental Artifact Integrity & Provenance Summary

## 1. Cryptographic Baseline Audit

To ensure the scientific validity of the XDrift study, strict integrity controls were enforced during notebook construction and execution. **No frozen baseline experimental artifacts were modified, regenerated, or overwritten.**

Below is the verification table comparing the audited baseline SHA-256 checksums (recorded in [`reports/final_forensic_audit_table.md`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/reports/final_forensic_audit_table.md)) against the post-execution checksums calculated directly from disk:

| Artifact Description | File Path | Baseline SHA-256 Checksum | Post-Execution SHA-256 Checksum | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Train Dataset** | `data/processed/train_data.npz` | `e0e07bf22c18877180fdd53469df60988f93b0bb570263290da12247b95d29ef` | `e0e07bf22c18877180fdd53469df60988f93b0bb570263290da12247b95d29ef` | **IDENTICAL** |
| **Validation Dataset** | `data/processed/val_data.npz` | `9b3d3e44b8b77151de5dbdb8b096bbc8ab37be2b0aaf67a64135655c362813e5` | `9b3d3e44b8b77151de5dbdb8b096bbc8ab37be2b0aaf67a64135655c362813e5` | **IDENTICAL** |
| **Test Dataset** | `data/processed/test_data.npz` | `9d238910e51feb629c2adaa0951592bdb9031e9675ee36265f02042ee03fc73e` | `9d238910e51feb629c2adaa0951592bdb9031e9675ee36265f02042ee03fc73e` | **IDENTICAL** |
| **FP32 Model Weights** | `artifacts/checkpoints/fp32_seed_42.pt` | `a6b5d65f80e05f8a8847ec261cef71499a4f732d168e8c7e8158bc413fcfed93` | `a6b5d65f80e05f8a8847ec261cef71499a4f732d168e8c7e8158bc413fcfed93` | **IDENTICAL** |
| **INT8-PTQ Quantized Model** | `artifacts/quantized_models/ptq_seed_42.pt` | `6b77806383723c40645172eeef23747866510c0b7507307b9bdc8c4c16926410` | `6b77806383723c40645172eeef23747866510c0b7507307b9bdc8c4c16926410` | **IDENTICAL** |
| **INT8-QAT Quantized Model** | `artifacts/quantized_models/qat_seed_42.pt` | `13fcfde7938c1787e6e8e15bafd1e3723bca0ca06b1e24bea210e0b3d783b8ca` | `13fcfde7938c1787e6e8e15bafd1e3723bca0ca06b1e24bea210e0b3d783b8ca` | **IDENTICAL** |
| **Test Split Predictions** | `results/predictions/test_predictions.npz` | `a6fe7ee3816b9c38761f618f66008dc0d1cf9404c4c0e56958525204066a9dd8` | `a6fe7ee3816b9c38761f618f66008dc0d1cf9404c4c0e56958525204066a9dd8` | **IDENTICAL** |
| **Cohort Attributions (100 Flows)** | `results/attributions/cohort_attributions.npz` | `b17ccf97c381f9f11f56a3368966fed53902b937cfd6854f3308cf95f8b89b13` | `b17ccf97c381f9f11f56a3368966fed53902b937cfd6854f3308cf95f8b89b13` | **IDENTICAL** |
| **Explainer Repeatability Cohort** | `results/attributions/explainer_repeatability_cohort.npz` | `51edf732c24266a475288ed3a04ba7b38c6170119596bd8f52d39fac001fc611` | `51edf732c24266a475288ed3a04ba7b38c6170119596bd8f52d39fac001fc611` | **IDENTICAL** |
| **Hardware Benchmarks Metrics** | `results/metrics/hardware_benchmarks.json` | `6ebd56d6f0d8a7d912c03f8b95108794b0e9a74845aff9caf3bfab40cb11d80a` | `6ebd56d6f0d8a7d912c03f8b95108794b0e9a74845aff9caf3bfab40cb11d80a` | **IDENTICAL** |
| **Master Pipeline Summary** | `results/metrics/master_pipeline_summary.json` | `91b8b7d76a36cfc8f87c9d70e649715c246aacda3580c80d4eb8b03a1385a4dd` | `91b8b7d76a36cfc8f87c9d70e649715c246aacda3580c80d4eb8b03a1385a4dd` | **IDENTICAL** |

---

## 2. Zero-Residual Checkpoint Execution Verification

In Section 7 of the notebook and Task 1 of the reproducibility gate, raw checkpoints were instantiated and executed through PyTorch forward passes across all 49,251 held-out test flows. The maximum absolute logit differences between newly computed logits and the cached predictions in `test_predictions.npz` were:
- **FP32 Model**: $\max |\Delta z| = 0.00000000 \times 10^0$ (Exact bitwise match)
- **INT8-PTQ Model**: $\max |\Delta z| = 0.00000000 \times 10^0$ (Exact bitwise match)
- **INT8-QAT Model**: $\max |\Delta z| = 0.00000000 \times 10^0$ (Exact bitwise match)

---

## 3. Automated Test Suite Status

- **Unit & Regression Suite**: `pytest tests/ -v`  
  - Total Tests: 15
  - Passed: 15 / 15 (100.0%)
  - Failed: 0
  - Execution Time: 2.11 seconds
- **Scientific Reproducibility Gate**: `python scripts/run_scientific_reproducibility_gate.py`  
  - Gate 1 (Predictive Recomputation): **PASS**
  - Gate 2 (Probability Calibration & Murphy Decomposition): **PASS**
  - Gate 3 (Attribution Provenance & KernelSHAP Additivity): **PASS**
  - Gate 4 (Hypothesis Testing & Wilcoxon Signed-Rank Direction): **PASS**
  - Gate 5 (Clustered Inference Across 16 Canonical Groups): **PASS**
  - Gate 6 (Hardware Benchmarking & Speedup Ratios): **PASS**
  - Gate 7 (H4 Subgroup Stratification): **PASS**
  - Gate 8 (Regression Suite): **PASS**

---

## 4. Certification Conclusion

The research walkthrough notebook operates directly against authoritative repository artifacts without reliance on mocked, simulated, or artificially generated data. All scientific calculations, confidence intervals, test statistics, and visualizations reflect exact empirical measurements from the certified experimental pipeline.
