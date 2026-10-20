# 📦 Historical Phase Notebooks Archive (October 17, 2026)

> [!IMPORTANT]
> **SUPERSEDED BY MASTER RESEARCH NOTEBOOK:**  
> The notebooks in this directory (`01_` through `04_`) represent early exploratory checkpoints from the initial development phases. They have been fully consolidated, audited, and superseded by the master research notebook at the repository root:
> 
> 👉 **[`Xdrift.ipynb`](../Xdrift.ipynb)**

---

### Why `Xdrift.ipynb` is the Single Source of Truth
1. **Methodological Rigor:**  
   Consolidates all 7 research phases into a unified 49-cell, self-contained, fully executed pipeline.
2. **True Fixed-Point INT8 Quantization:**  
   Implements true discrete integer inference and captures the authentic behavior of Quantization-Aware Training (QAT) fine-tuned for 5 epochs.
3. **Dual-Explainer Cross-Validation:**  
   Includes multi-seed KernelSHAP variance analysis (5 seeds) alongside Captum's Integrated Gradients via FakeQuantization with a Straight-Through Estimator (STE).
4. **Data Integrity & Metric Alignment:**  
   Fully aligned with `Comprehensive_Methodology_and_Drift_Defense_Report.md` and `Dataset/models/*.json` reports with 100% mathematical precision.

Please consult **[`Xdrift.ipynb`](../Xdrift.ipynb)** and **[`Comprehensive_Methodology_and_Drift_Defense_Report.md`](../Comprehensive_Methodology_and_Drift_Defense_Report.md)** for all current research findings, benchmarks, and defenses.
