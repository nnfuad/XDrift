# XDrift Master Research Notebook Execution Summary

## 1. Execution Overview

| Property | Value |
| :--- | :--- |
| **Notebook File** | [`XDrift_A_to_Z_Research_Walkthrough.ipynb`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/notebooks/XDrift_A_to_Z_Research_Walkthrough.ipynb) |
| **Execution Date** | 2026-10-24 |
| **Execution Environment** | macOS Apple Silicon (ARM64), Python 3.12.7, PyTorch 2.5.1, QNNPACK backend |
| **Total Notebook Cells** | 78 cells |
| **Markdown Narrative Cells** | 42 cells |
| **Executable Code Cells** | 36 cells |
| **Code Cells Executed** | 36 / 36 (100.0%) |
| **Code Cells Failed** | 0 / 36 (0.0%) |
| **Total Wall-Clock Execution Time** | 341.28 seconds (~5.69 minutes) |
| **Final Automated Verification Cell** | **PASS (All 19 Figures generated, all 22 Invariant checks satisfied)** |

---

## 2. Generated Figures Catalog

All 19 figures were generated programmatically and saved simultaneously in high-resolution PNG (300 DPI) and publication-ready vector PDF formats in [`notebooks/figures/`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/notebooks/figures/):

| Figure Number | Filename | Topic / Description | Formats Verified |
| :--- | :--- | :--- | :--- |
| **Figure 1** | `figure1_flow_construction_pipeline` | Raw PCAP to Bidirectional $(2, 64)$ Tensor Transformation Pipeline | PNG (300 DPI), Vector PDF |
| **Figure 2** | `figure2_cnn1d_architecture` | 1D-CNN Layer Connectivity, Kernel Sizes, and Channel Progression | PNG (300 DPI), Vector PDF |
| **Figure 3** | `figure3_quantization_workflows` | Post-Training Quantization (PTQ) vs Quantization-Aware Training (QAT) Flowchart | PNG (300 DPI), Vector PDF |
| **Figure 4** | `figure4_confusion_matrices` | Confusion Matrices for FP32 ($\tau=0.9$), PTQ ($\tau=0.9$), and QAT ($\tau=0.68$) | PNG (300 DPI), Vector PDF |
| **Figure 5** | `figure5_roc_curves` | Receiver Operating Characteristic (ROC) Curves across all three models | PNG (300 DPI), Vector PDF |
| **Figure 6** | `figure6_pr_curves` | Precision-Recall (PR) Curves with baseline prevalence ($5.47\%$) | PNG (300 DPI), Vector PDF |
| **Figure 7** | `figure7_calibration_reliability` | Calibration Reliability Diagrams (10 Bins) and Brier Score Decompositions | PNG (300 DPI), Vector PDF |
| **Figure 8** | `figure8_qat_probability_saturation` | QAT Predicted Probability Histograms showing the $p_{\max} = 0.7338$ Ceiling | PNG (300 DPI), Vector PDF |
| **Figure 9** | `figure9_qat_logit_ceiling` | Forensic Derivation of the QAT Unsigned Quantized Logit Representation Clamping | PNG (300 DPI), Vector PDF |
| **Figure 10** | `figure10_explainer_noise_baseline` | Multi-Seed KernelSHAP Explainer Repeatability Noise Distribution ($N=100$) | PNG (300 DPI), Vector PDF |
| **Figure 11** | `figure11_ptq_attribution_drift` | Paired PTQ Attribution Drift ($\Delta \text{TV}$) Histogram and Paired Scatter Plot | PNG (300 DPI), Vector PDF |
| **Figure 12** | `figure12_attribution_stability_comparison` | Grouped Bar Chart of Attribution Stability Metrics (FP32 vs PTQ vs QAT) | PNG (300 DPI), Vector PDF |
| **Figure 13** | `figure13_class_conditional_drift` | Boxplots with Individual Flow Jitter for VPN vs Non-VPN Attribution Drift | PNG (300 DPI), Vector PDF |
| **Figure 14** | `figure14_group_logit_shifts` | Mean Logit Shift Bar Chart by Canonical Session Group ($K=16$) with Sample Sizes | PNG (300 DPI), Vector PDF |
| **Figure 15** | `figure15_cluster_bootstrap_distribution` | Bimodal Flow-Weighted Cluster Bootstrap Distribution ($B=10,000$) | PNG (300 DPI), Vector PDF |
| **Figure 16** | `figure16_leave_one_group_out` | Leave-One-Group-Out Sensitivity Analysis showing Outsized Leverage of `skype_audio3` | PNG (300 DPI), Vector PDF |
| **Figure 17** | `figure17_inference_latency` | Streaming (BS=1) vs Batched (BS=64) Median Inference Latency (Log Scale) | PNG (300 DPI), Vector PDF |
| **Figure 18** | `figure18_inference_throughput` | Streaming vs Batched Inference Throughput ($11.94\times$ PTQ Batched Acceleration) | PNG (300 DPI), Vector PDF |
| **Figure 19** | `figure19_checkpoint_sizes` | Model Checkpoint Compression Comparison ($188.7\text{ KB} \to 53.0\text{ KB}$, $3.56\times$) | PNG (300 DPI), Vector PDF |

---

## 3. Automated Verification Gate Results (Cell #78 / Code Cell #36)

The notebook includes a terminal automated verification cell that checks both file existence and exact numerical invariants against audited project baselines:

```
=== Running Final Notebook Automated Verification Gate ===
1. Figure Generation Audit:        PASS (All 19 Figures generated in PNG and PDF)
2. Numerical Invariant Audit:     PASS (All 22 Invariant checks satisfied)

================================================================================
     ALL VERIFICATION CHECKS PASSED: NOTEBOOK IS FULLY REPRODUCIBLE
================================================================================
```

### Numerical Invariant Audit Breakdown

| Check Item | Tested Condition | Observed Value | Status |
| :--- | :--- | :--- | :--- |
| **1. Total Test Flows** | $N_{\text{test}} == 49,251$ | $49,251$ | **PASS** |
| **2. Test VPN Flows** | $N_{\text{vpn}} == 2,693$ | $2,693$ | **PASS** |
| **3. Canonical Session Groups** | $K == 16$ | $16$ | **PASS** |
| **4. FP32 Parameter Count** | $\text{Params} == 44,129$ | $44,129$ | **PASS** |
| **5. FP32 ROC-AUC** | $\text{ROC-AUC} \approx 0.9796$ | $0.9796$ | **PASS** |
| **6. PTQ ROC-AUC** | $\text{ROC-AUC} \approx 0.9785$ | $0.9785$ | **PASS** |
| **7. QAT ROC-AUC** | $\text{ROC-AUC} \approx 0.9834$ | $0.9834$ | **PASS** |
| **8. FP32 Calibrated F1** | $F_1 \approx 0.7746$ | $0.7746$ | **PASS** |
| **9. PTQ Calibrated F1** | $F_1 \approx 0.7697$ | $0.7697$ | **PASS** |
| **10. QAT Calibrated F1** | $F_1 \approx 0.7762$ | $0.7762$ | **PASS** |
| **11. QAT Reused F1** | $F_1 == 0.0000$ | $0.0000$ | **PASS** |
| **12. QAT Probability Ceiling** | $p_{\max} \approx 0.733812$ | $0.733812$ | **PASS** |
| **13. QAT VPN Ceiling Count** | $N_{\text{vpn at ceiling}} == 2,642$ | $2,642$ ($98.11\%$) | **PASS** |
| **14. Wilcoxon Statistic W** | $W == 1506.0$ | $1506.0$ | **PASS** |
| **15. Wilcoxon p-value** | $p \approx 0.00045892$ | $0.00045892$ | **PASS** |
| **16. PTQ Mean Top-5 Jaccard** | $\text{IoU}_5 \approx 0.9786$ | $0.9786$ | **PASS** |
| **17. PTQ Mean Total Variation** | $\text{TV} \approx 0.0369$ | $0.0369$ | **PASS** |
| **18. QAT Mean Total Variation** | $\text{TV} \approx 0.2346$ | $0.2346$ | **PASS** |
| **19. Estimand A Point Estimate** | $\theta_A \approx -0.0778$ | $-0.0778$ | **PASS** |
| **20. Estimand B Point Estimate** | $\theta_B \approx +0.2270$ | $+0.2270$ | **PASS** |
| **21. Batched Throughput Speedup** | $\text{Speedup} \approx 11.94\times$ | $11.94\times$ | **PASS** |
| **22. Checkpoint Compression** | $\text{Ratio} \approx 3.56\times$ | $3.56\times$ | **PASS** |

---

## 4. Diagnostic & Resolution History During Assembly

During notebook development and automated execution via `ExecutePreprocessor`, three engineering issues were identified, diagnosed, and resolved:

1. **Attribution Masking for Truncated Flows**:
   - *Issue*: Unmasked Total Variation calculations included unobserved zero-padded sequence positions, skewing rank counts.
   - *Resolution*: Supplied the exact boolean sequence validity masks (`cohort_masks`) to `compute_pairwise_drift_metrics`, strictly evaluating packet positions $1 \dots L_i$. This perfectly reproduced the authoritative result: $69$ positive differences, $W=1506.0$, $p=0.000459$.
2. **Hardware Benchmark JSON Dictionary Keys**:
   - *Issue*: Checkpoint size keys in `results/metrics/hardware_benchmarks.json` are formatted as `fp32_bytes`, `ptq_bytes`, `qat_bytes`, rather than `fp32_checkpoint_bytes`.
   - *Resolution*: Corrected key indexing in `scripts/notebook_builder/part5_clustered_hardware.py`.
3. **Lazy Import of SHAP to Mitigate NumPy 2.x C-Extension Stderr Capture**:
   - *Issue*: `import shap` at module level triggered NumPy 1.x vs 2.x C-extension deprecation warnings from optional dependencies (`numexpr`, `bottleneck`, `pyarrow`), which `ipykernel` captured into the notebook traceback queue.
   - *Resolution*: Converted `import shap` in [`src/explainability/kernel_shap.py`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/src/explainability/kernel_shap.py) into a lazy import inside `explain_single_model()`. This allows all drift metric functions and utility modules to be imported without third-party C-extension warnings while preserving 100% functionality.
