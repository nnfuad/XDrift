# XDrift Final Research Summary: Qualified Empirical Findings

## 1. Dataset & Provenance
- **Total Flows Processed:** 55,498 (Train: 38,848, Val: 8,325, Test: 8,325).
- **Cryptographic Verification:** SHA-256 hashes for all 3 splits match Croissant metadata. Zero NaN, infinite, or missing values.
- **Dataset Partition Limitation:** Evaluation is restricted to stratified flow-level partitions. Raw 5-tuples and capture session IDs are unavailable, precluding verification of host- or PCAP-level session independence.

## 2. Predictive Performance & 10,000-Iteration Bootstrap Audit
- **FP32 Reference:** Accuracy 96.67%, VPN Recall 95.98%, Macro F1 0.8517, ROC-AUC 0.9921.
- **INT8 PTQ:** Accuracy 96.74%, VPN Recall 95.98%, Macro F1 0.8541, ROC-AUC 0.9914.
  - Paired diff vs FP32: ΔAcc = +0.07% (FDR q = 0.2955); ΔRecall = 0.00% (FDR q = 1.0000); ΔMacro F1 = +0.0024 (FDR q = 0.3213).
  - TOST Equivalence: Under an engineering tolerance margin of δ = ±1.0%, PTQ is statistically equivalent to FP32 (p_TOST < 0.0001).
- **INT8 QAT Deploy:** Accuracy 97.00%, VPN Recall 95.17%, Macro F1 0.8618, ROC-AUC 0.9910.
  - Paired diff vs FP32: ΔAcc = +0.32% (FDR q < 0.0001); ΔMacro F1 = +0.0102 (FDR q = 0.0014); ΔVPN Precision = +2.74% (FDR q < 0.0001).
  - VPN Recall change vs FP32: ΔRecall = -0.80% (95% CI: [-1.82%, 0.00%], FDR q = 0.0955, not statistically significant; touches 0.00%, reflecting 3 fewer detected flows: 355 vs 358 out of 373).
- **Prediction Agreement:** Models agree on 99.42% of test predictions across all three variants (48 / 8,325 disagreement flows: 44 Non-VPN, 4 VPN).

## 3. Explainability Drift: Unmatched vs. Length-Matched Cohorts
- **Natural Unmatched Cohort (N = 746):**
  - Non-VPN Traffic (Median 4 Packets): Top-5 IoU = 0.6959, Spearman ρ = 0.7341.
  - VPN Traffic (Median 24 Packets): Top-5 IoU = 0.3032, Spearman ρ = 0.3301.
  - QAT Deploy VPN: Top-5 IoU = 0.2944, Spearman ρ = 0.3287.
  - Confounder: Active sequence length explains R^2 = 0.448 of attribution variance (hypergeometric chance baseline drops from 0.4707 at L=4 to 0.0604 at L=24).
- **Controlled Sequence-Length-Matched Cohort (120 Matched Pairs):**
  - Top-5 IoU: Non-VPN 0.4652 vs VPN 0.5312 (Paired Diff +0.0660, 95% CI [+0.0254, +0.1065], Wilcoxon p = 0.0053).
  - Spearman ρ: Non-VPN 0.4918 vs VPN 0.5852 (Paired Diff +0.0934, 95% CI [+0.0557, +0.1323], Wilcoxon p < 0.0001, Permutation p < 0.0001).
  - Core Insight: Controlling for length refutes the notion that VPN flows are inherently more fragile; however, INT8 quantization still reorganizes ~50% of top features across ALL traffic types despite 99.4% classification agreement.

## 4. Hardware Benchmarks
- **Raw Weight Compression:** 3.55× (182.8 KB FP32 to 51.4 KB INT8).
- **Throughput:** Compiled TorchScript execution achieves 111,025 flows/sec at batch size 32 on host CPU.

## 5. Granular Research Audit Scorecard
- **Mechanical & Arithmetic Checks:** VERIFIED (Checksums, model-to-prediction provenance, confusion matrix arithmetic, tie-corrected bootstrap p-values).
- **Multiple Testing:** Adjusted via Benjamini-Hochberg (FDR) and Holm-Bonferroni across 21 paired tests.
- **Structural Limitations:** Explicitly documented (host/session grouping unverified; gradient explainer surrogate assumption; finite SHAP sampling sensitivity).
