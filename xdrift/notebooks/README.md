# XDrift: Complete A–Z Research Walkthrough Notebook

This directory contains the primary executable Jupyter research notebook and supporting assets for the research study:
> **Investigating Attribution Drift in INT8-Quantized 1D CNNs for Encrypted Traffic Classification (XDrift)**

---

## 1. Primary Deliverables

- **Master Research Notebook**: [`XDrift_A_to_Z_Research_Walkthrough.ipynb`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/notebooks/XDrift_A_to_Z_Research_Walkthrough.ipynb)  
  *A comprehensive, 78-cell executable Jupyter notebook spanning the entire research journey from network traffic fundamentals to clustered statistical inference and hardware benchmarking. Fully populated with executed outputs, tables, and 19 inline figures.*
- **Figures Directory**: [`figures/`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/notebooks/figures/)  
  *Contains all 19 figures generated during notebook execution, exported in dual formats: 300 DPI high-resolution PNG (`.png`) and vector publication-ready PDF (`.pdf`).*
- **Execution Summary**: [`EXECUTION_SUMMARY.md`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/notebooks/EXECUTION_SUMMARY.md)  
  *Reports cell counts, execution timings, output audit statistics, and detailed invariant test results.*
- **Integrity Summary**: [`INTEGRITY_SUMMARY.md`](file:///Volumes/LEXAR/Explainability%20Drift/xdrift/notebooks/INTEGRITY_SUMMARY.md)  
  *Cryptographic SHA-256 confirmation that all frozen baseline experimental artifacts remain pristine and unmodified.*

---

## 2. Notebook Structure & Content Map

The notebook is divided into six logical parts comprising 23 structured sections (Sections 0–22):

### Part 1: Foundations & Experimental Setup
- **Section 0: Executive Roadmap & Theoretical Grounding**: Overview of the research questions, dual-plane pipeline (data plane vs explanation plane), and quantitative milestones.
- **Section 1: Execution Environment & Cryptographic Hash Audit**: Hardware specification, PyTorch/QNNPACK backend details, and verification of dataset/model SHA-256 hashes.
- **Section 2: Network Traffic Fundamentals**: Packet dynamics, bidirectional 5-tuple canonicalization, inactivity timeout ($120\text{s}$), and transport-layer header stripping.
- **Section 3: Dataset Ingestion & Leakage-Free Session Group Isolation**: 140 PCAP captures, lexical normalization, Hangouts alias co-location, and strict canonical group disjunction across Train, Validation, and Test splits.

### Part 2: Models & Quantization Mechanics
- **Section 4: Flow Construction Pipeline**: Feature tensor representation $(N, 2, 64)$, directional lengths, inter-arrival times, log-scaling, and temporal padding. *(Generates Figure 1)*
- **Section 5: 1D-CNN Model Architecture**: Layer connectivity, receptive fields, 44,129 parameters, and class imbalance mitigation via loss weighting ($\text{pos\_weight} = 30.3756$). *(Generates Figure 2)*
- **Section 6: Quantization Fundamentals: FP32 vs PTQ vs QAT**: Uniform affine quantization equations, fake-quantization, Straight-Through Estimator (STE), and operator fusion. *(Generates Figure 3)*
- **Section 7: Checkpoint Loading & Zero-Residual Validation**: Instantiating deployed PyTorch graphs, loading frozen weights, and verifying exact bitwise logit reproduction ($0.00$ residual).

### Part 3: Predictive Performance & Probability Calibration
- **Section 8: Predictive Performance Evaluation**: Threshold-independent ranking (ROC-AUC, PR-AUC) and threshold-calibrated classification metrics on 49,251 held-out test flows. *(Generates Figures 4, 5, 6)*
- **Section 9: The Operational Cost of Quantization: Threshold Calibration**: Reusing FP32 threshold ($\tau=0.900$) on QAT produces total recall collapse ($F_1 = 0.0000$); validation re-calibration ($\tau=0.680$) fully restores accuracy ($F_1 = 0.7762$).
- **Section 10: Probability Calibration & Reliability Assessment**: Brier score Murphy decomposition (Uncertainty, Reliability, Resolution) and Expected Calibration Error (ECE). *(Generates Figure 7)*
- **Section 11: Forensic Derivation: Why QAT Saturates at $p_{\max} = 0.733812$**: Mathematical derivation tracing output linear layer scale, zero-point, unsigned integer clamping, and sigmoid activation saturation. *(Generates Figures 8, 9)*

### Part 4: Explainability, Attribution Drift & Subgroup Analyses
- **Section 12: Explainability Framework: KernelSHAP for Network Traffic**: Cooperative game theory, coalition sampling, 150-evaluation budget, and empirical verification of the efficiency axiom (additivity residual $< 7.15 \times 10^{-6}$).
- **Section 13: Measuring Explainer Noise**: Establishing the multi-seed baseline across Seeds 42, 43, 44 on identical FP32 models ($N=100$). *(Generates Figure 10)*
- **Section 14: Quantifying PTQ Attribution Drift: The Paired Wilcoxon Test**: Paired difference $\Delta \text{TV} = \text{TV}(\text{PTQ}) - \text{TV}(\text{Noise})$. Resolving the Wilcoxon direction ($W=1506.0$, $p=0.000459$, $R^+=3544.0 > R^-=1506.0$) and why mean ($-0.0185$) and median ($+0.0049$) have opposite signs. *(Generates Figure 11)*
- **Section 15: QAT Attribution Restructuring**: Signed cosine collapse to $0.4830$, Top-5 Jaccard drop to $0.9553$, and Total Variation surge to $0.2346$. *(Generates Figure 12)*
- **Section 16: Class-Conditional Attribution Drift**: Subgroup analysis demonstrating that Non-VPN traffic experiences significantly higher Total Variation drift than VPN traffic ($p = 1.13 \times 10^{-7}$). *(Generates Figure 13)*

### Part 5: Clustered Statistical Inference & Edge Hardware Benchmarks
- **Section 17: Clustered Statistical Inference Across Canonical Session Groups**: Distinguishing Estimand A (flow-weighted mean: $-0.0778$, CI spans zero) from Estimand B (group-unweighted mean: $+0.2270$, strictly positive). Leave-one-group-out sensitivity demonstrating the outsized leverage of `skype_audio3` (79.55% of test flows). *(Generates Figures 14, 15, 16)*
- **Section 18: Hardware Performance and Model Size**: Single-flow streaming latency (BS=1, modest $1.05\times$ speedup) versus batched throughput (BS=64, $11.94\times$ speedup), and $3.56\times$ checkpoint storage compression. *(Generates Figures 17, 18, 19)*
- **Section 19: Research Findings and Hypothesis Assessment Matrix**: Comprehensive evidence synthesis testing Hypotheses H1, H2, H3, and H4 against quantitative criteria.

### Part 6: Critical Limitations, Conclusions & Automated Verification
- **Section 20: Critical Limitations**: Systematic documentation of 12 critical scientific boundaries (class imbalance, small group count $K=16$, single capture dominance, QAT saturation, platform boundaries, etc.).
- **Section 21: Final Conclusions**: Core scientific findings and actionable deployment recommendations for practitioners.
- **Section 22: Reproducibility Appendix & Automated Verification Cell**: Complete repository layout and self-contained programmatic gate checking all 19 figures and 22 numerical invariants.

---

## 3. How to Run the Notebook

### Environment Prerequisites
- macOS Apple Silicon (ARM64 recommended)
- Python 3.12.7+ (e.g. via Anaconda / Miniconda)
- PyTorch 2.5.1 with QNNPACK quantization engine enabled

### Quickstart Execution

To launch and run the notebook interactively:
```bash
cd "/Volumes/LEXAR/Explainability Drift/xdrift"
jupyter lab notebooks/XDrift_A_to_Z_Research_Walkthrough.ipynb
```
From the Jupyter menu, select **Kernel -> Restart Kernel and Run All Cells**.

### Headless Programmatic Re-Execution

To re-execute the notebook headlessly from the command line and update all outputs and figures:
```bash
cd "/Volumes/LEXAR/Explainability Drift/xdrift"
python scripts/notebook_builder/assemble_and_run.py
```
This executes the entire pipeline via `nbconvert.preprocessors.ExecutePreprocessor` in approximately 5.7 minutes.
