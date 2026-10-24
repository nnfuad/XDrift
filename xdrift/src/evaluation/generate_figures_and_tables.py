"""Figure and Table Generation Script for XDrift Protocol.

Generates:
- Table 1: Predictive Performance Comparison (FP32 vs PTQ vs QAT)
- Table 2: Attribution Drift Metrics across Subgroups
- Table 3: Hardware & Efficiency Benchmarks
- Table 4: Primary Hypothesis Testing Matrix (H1-H4)
- Figure 1: Top-5 Jaccard overlap distribution vs Random Baseline & Explainer Noise
- Figure 2: Total Variation and Signed Cosine Similarity distributions
- Figure 3: Temporal attribution profiles across packet positions (masked)
- Figure 4: Edge Latency vs Artifact Size vs Accuracy Tradeoff
"""

import os
import json
from typing import Dict, List, Any

import numpy as np
import matplotlib.pyplot as plt


def generate_all_tables_and_figures(output_base_dir: str) -> None:
    results_dir = os.path.join(output_base_dir, "results")
    figures_dir = os.path.join(results_dir, "figures")
    tables_dir = os.path.join(results_dir, "tables")
    metrics_dir = os.path.join(results_dir, "metrics")
    attr_dir = os.path.join(results_dir, "attributions")

    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(tables_dir, exist_ok=True)

    # Load master summary if available
    master_file = os.path.join(metrics_dir, "master_pipeline_summary.json")
    if not os.path.exists(master_file):
        print(f"Warning: {master_file} not found yet.")
        return

    with open(master_file, 'r') as f:
        master = json.load(f)

    pred = master['predictive_evaluation']
    attr = master['attribution_drift']
    stats = master['statistical_analysis']
    hw = master['hardware_benchmarks']

    # -------------------------------------------------------------------------
    # TABLE 1: Predictive Performance Comparison
    # -------------------------------------------------------------------------
    t1_md = f"""# Table 1: Predictive Performance Comparison on Held-Out Test Set (N={pred['test_examples']})

| Metric | FP32 Baseline | INT8-PTQ | Delta (PTQ - FP32) | INT8-QAT | Delta (QAT - FP32) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | {pred['fp32_metrics']['accuracy']:.4f} | {pred['ptq_metrics']['accuracy']:.4f} | {pred['paired_ptq_diff']['delta_accuracy']:+.4f} | {pred.get('qat_metrics', {}).get('accuracy', 0.0):.4f} | {pred.get('paired_qat_diff', {}).get('delta_accuracy', 0.0):+.4f} |
| **Balanced Accuracy** | {pred['fp32_metrics']['balanced_accuracy']:.4f} | {pred['ptq_metrics']['balanced_accuracy']:.4f} | {pred['ptq_metrics']['balanced_accuracy'] - pred['fp32_metrics']['balanced_accuracy']:+.4f} | {pred.get('qat_metrics', {}).get('balanced_accuracy', 0.0):.4f} | {pred.get('qat_metrics', {}).get('balanced_accuracy', 0.0) - pred['fp32_metrics']['balanced_accuracy']:+.4f} |
| **Precision (VPN)** | {pred['fp32_metrics']['precision']:.4f} | {pred['ptq_metrics']['precision']:.4f} | {pred['ptq_metrics']['precision'] - pred['fp32_metrics']['precision']:+.4f} | {pred.get('qat_metrics', {}).get('precision', 0.0):.4f} | {pred.get('qat_metrics', {}).get('precision', 0.0) - pred['fp32_metrics']['precision']:+.4f} |
| **Recall (VPN)** | {pred['fp32_metrics']['recall']:.4f} | {pred['ptq_metrics']['recall']:.4f} | {pred['ptq_metrics']['recall'] - pred['fp32_metrics']['recall']:+.4f} | {pred.get('qat_metrics', {}).get('recall', 0.0):.4f} | {pred.get('qat_metrics', {}).get('recall', 0.0) - pred['fp32_metrics']['recall']:+.4f} |
| **F1-Score** | {pred['fp32_metrics']['f1']:.4f} | {pred['ptq_metrics']['f1']:.4f} | {pred['paired_ptq_diff']['delta_f1']:+.4f} | {pred.get('qat_metrics', {}).get('f1', 0.0):.4f} | {pred.get('paired_qat_diff', {}).get('delta_f1', 0.0):+.4f} |
| **ROC-AUC** | {pred['fp32_metrics']['roc_auc']:.4f} | {pred['ptq_metrics']['roc_auc']:.4f} | {pred['ptq_metrics']['roc_auc'] - pred['fp32_metrics']['roc_auc']:+.4f} | {pred.get('qat_metrics', {}).get('roc_auc', 0.0):.4f} | {pred.get('qat_metrics', {}).get('roc_auc', 0.0) - pred['fp32_metrics']['roc_auc']:+.4f} |
| **PR-AUC** | {pred['fp32_metrics']['pr_auc']:.4f} | {pred['ptq_metrics']['pr_auc']:.4f} | {pred['paired_ptq_diff']['delta_pr_auc']:+.4f} | {pred.get('qat_metrics', {}).get('pr_auc', 0.0):.4f} | {pred.get('paired_qat_diff', {}).get('delta_pr_auc', 0.0):+.4f} |
| **Brier Score** | {pred['fp32_metrics']['brier_score']:.4f} | {pred['ptq_metrics']['brier_score']:.4f} | {pred['ptq_metrics']['brier_score'] - pred['fp32_metrics']['brier_score']:+.4f} | {pred.get('qat_metrics', {}).get('brier_score', 0.0):.4f} | {pred.get('qat_metrics', {}).get('brier_score', 0.0) - pred['fp32_metrics']['brier_score']:+.4f} |
| **Prediction Flip Rate** | — | {pred['paired_ptq_diff']['prediction_flip_rate']*100:.2f}% | — | {pred.get('paired_qat_diff', {}).get('prediction_flip_rate', 0.0)*100:.2f}% | — |
"""
    with open(os.path.join(tables_dir, "table1_predictive_performance.md"), 'w') as f:
        f.write(t1_md)

    # -------------------------------------------------------------------------
    # TABLE 2: Attribution Drift Metrics Across Subgroups
    # -------------------------------------------------------------------------
    ptq_drift = attr['fp32_vs_ptq']
    qat_drift = attr['fp32_vs_qat']
    sub = attr.get('subgroups_ptq', {})

    t2_md = f"""# Table 2: Attribution Drift Metrics (FP32 vs INT8-PTQ & INT8-QAT)

| Subgroup / Condition | Top-5 Jaccard ($IoU_5$) | Top-10 Jaccard ($IoU_{{10}}$) | Spearman Rank ($r_s$) | Total Variation ($TV$) | Signed Cosine ($S_C$) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Baseline (Combinatorial)** | {ptq_drift['random_baseline_k5']:.4f} | {ptq_drift['random_baseline_k10']:.4f} | 0.0000 | 1.0000 | 0.0000 |
| **Explainer Noise Baseline** | {master['explainability_pilot']['within_model_explainer_noise']['top5_jaccard_mean']:.4f} | — | {master['explainability_pilot']['within_model_explainer_noise']['spearman_mean']:.4f} | {master['explainability_pilot']['within_model_explainer_noise']['tv_mean']:.4f} | {master['explainability_pilot']['within_model_explainer_noise']['cosine_mean']:.4f} |
| **Primary: All Flows (PTQ vs FP32)** | **{ptq_drift['top5_jaccard']['mean']:.4f}** | **{ptq_drift['top10_jaccard']['mean']:.4f}** | **{ptq_drift['spearman_correlation']['mean']:.4f}** | **{ptq_drift['total_variation_distance']['mean']:.4f}** | **{ptq_drift['signed_cosine_similarity']['mean']:.4f}** |
| **Primary: All Flows (QAT vs FP32)** | **{qat_drift['top5_jaccard']['mean']:.4f}** | **{qat_drift['top10_jaccard']['mean']:.4f}** | **{qat_drift['spearman_correlation']['mean']:.4f}** | **{qat_drift['total_variation_distance']['mean']:.4f}** | **{qat_drift['signed_cosine_similarity']['mean']:.4f}** |
| Subgroup: VPN Flows | {sub.get('vpn_only', {}).get('top5_jaccard', {}).get('mean', 0.0):.4f} | — | {sub.get('vpn_only', {}).get('spearman_correlation', {}).get('mean', 0.0):.4f} | {sub.get('vpn_only', {}).get('total_variation_distance', {}).get('mean', 0.0):.4f} | {sub.get('vpn_only', {}).get('signed_cosine_similarity', {}).get('mean', 0.0):.4f} |
| Subgroup: Non-VPN Flows | {sub.get('nonvpn_only', {}).get('top5_jaccard', {}).get('mean', 0.0):.4f} | — | {sub.get('nonvpn_only', {}).get('spearman_correlation', {}).get('mean', 0.0):.4f} | {sub.get('nonvpn_only', {}).get('total_variation_distance', {}).get('mean', 0.0):.4f} | {sub.get('nonvpn_only', {}).get('signed_cosine_similarity', {}).get('mean', 0.0):.4f} |
| Subgroup: Prediction Agree | {sub.get('prediction_agree', {}).get('top5_jaccard', {}).get('mean', 0.0):.4f} | — | {sub.get('prediction_agree', {}).get('spearman_correlation', {}).get('mean', 0.0):.4f} | {sub.get('prediction_agree', {}).get('total_variation_distance', {}).get('mean', 0.0):.4f} | {sub.get('prediction_agree', {}).get('signed_cosine_similarity', {}).get('mean', 0.0):.4f} |
| Subgroup: Prediction Disagree | {sub.get('prediction_disagree', {}).get('top5_jaccard', {}).get('mean', 0.0):.4f} | — | {sub.get('prediction_disagree', {}).get('spearman_correlation', {}).get('mean', 0.0):.4f} | {sub.get('prediction_disagree', {}).get('total_variation_distance', {}).get('mean', 0.0):.4f} | {sub.get('prediction_disagree', {}).get('signed_cosine_similarity', {}).get('mean', 0.0):.4f} |
"""
    with open(os.path.join(tables_dir, "table2_attribution_drift.md"), 'w') as f:
        f.write(t2_md)

    # -------------------------------------------------------------------------
    # TABLE 3: Hardware Benchmarks
    # -------------------------------------------------------------------------
    t3_md = f"""# Table 3: Hardware and Inference Benchmarks (macOS ARM64 QNNPACK)

| Model Condition | Artifact Size (KB) | Compression Ratio | Edge BS=1 Latency (Median ms) | BS=1 IQR (ms) | BS=1 p95 (ms) | Throughput (samples/s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **FP32 Baseline** | {hw['artifact_sizes']['fp32_kb']} KB | 1.00x | {hw['latency_edge_bs1']['fp32']['median_latency_ms']:.3f} ms | {hw['latency_edge_bs1']['fp32']['iqr_latency_ms']:.3f} ms | {hw['latency_edge_bs1']['fp32']['p95_latency_ms']:.3f} ms | {hw['latency_edge_bs1']['fp32']['throughput_samples_per_sec']:.1f} |
| **INT8-PTQ** | {hw['artifact_sizes']['ptq_kb']} KB | {hw['artifact_sizes']['ptq_compression_ratio']}x | {hw['latency_edge_bs1']['ptq']['median_latency_ms']:.3f} ms | {hw['latency_edge_bs1']['ptq']['iqr_latency_ms']:.3f} ms | {hw['latency_edge_bs1']['ptq']['p95_latency_ms']:.3f} ms | {hw['latency_edge_bs1']['ptq']['throughput_samples_per_sec']:.1f} |
| **INT8-QAT** | {hw['artifact_sizes']['qat_kb']} KB | {hw['artifact_sizes']['qat_compression_ratio']}x | {hw['latency_edge_bs1']['qat']['median_latency_ms']:.3f} ms | {hw['latency_edge_bs1']['qat']['iqr_latency_ms']:.3f} ms | {hw['latency_edge_bs1']['qat']['p95_latency_ms']:.3f} ms | {hw['latency_edge_bs1']['qat']['throughput_samples_per_sec']:.1f} |
"""
    with open(os.path.join(tables_dir, "table3_hardware_benchmarks.md"), 'w') as f:
        f.write(t3_md)

    # -------------------------------------------------------------------------
    # TABLE 4: Hypothesis Testing Matrix
    # -------------------------------------------------------------------------
    p_h1 = pred['paired_ptq_diff']['prediction_flip_rate']
    mean_jacc = stats['primary_metric_top5_jaccard']['observed_mean']
    noise_jacc = stats['primary_metric_top5_jaccard']['explainer_noise_baseline']
    delta_qat_ptq = stats['comparison_qat_vs_ptq']['delta_jaccard']

    h1_status = "SUPPORTED" if p_h1 > 0 else "NOT SUPPORTED"
    h2_status = "SUPPORTED" if mean_jacc < noise_jacc else "NOT RESOLVED (Within Explainer Noise)"
    h3_status = "SUPPORTED (QAT Differs)" if abs(delta_qat_ptq) > 0.01 else "NOT SUPPORTED (No Significant Difference)"

    t4_md = f"""# Table 4: Primary Hypothesis Testing Matrix

| Hypothesis | Registration & Question | Test Statistic / Evidence | Statistical Decision |
| :--- | :--- | :--- | :---: |
| **H1: Predictive Drift** | Does INT8 quantization change predictive outputs or accuracy relative to FP32? | Flip Rate = {p_h1*100:.2f}%, $\\Delta$Acc = {pred['paired_ptq_diff']['delta_accuracy']:+.4f}, $\\Delta$F1 = {pred['paired_ptq_diff']['delta_f1']:+.4f} | **{h1_status}** |
| **H2: Attribution Drift** | Does INT8-PTQ produce measurable changes in feature attribution beyond explainer noise? | Mean $IoU_5$ = {mean_jacc:.4f} (95% CI: [{stats['primary_metric_top5_jaccard']['bootstrap_95ci_flow_level'][0]:.4f}, {stats['primary_metric_top5_jaccard']['bootstrap_95ci_flow_level'][1]:.4f}]), Explainer Noise Baseline = {noise_jacc:.4f} | **{h2_status}** |
| **H3: QAT Comparison** | Does INT8-QAT exhibit different predictive and attribution drift than INT8-PTQ? | $\\Delta IoU_5$ (QAT - PTQ) = {delta_qat_ptq:+.4f}, Holm-corrected $p$ = {stats['comparison_qat_vs_ptq']['holm_corrected_pvalue']:.4e} | **{h3_status}** |
| **H4: Conditional Variation** | Does attribution drift vary across traffic classes and flow lengths? | Subgroup differences between VPN ($IoU_5$={sub.get('vpn_only',{}).get('top5_jaccard',{}).get('mean',0.0):.4f}) vs Non-VPN ($IoU_5$={sub.get('nonvpn_only',{}).get('top5_jaccard',{}).get('mean',0.0):.4f}) | **EVALUATED** |
"""
    with open(os.path.join(tables_dir, "table4_hypothesis_matrix.md"), 'w') as f:
        f.write(t4_md)

    # -------------------------------------------------------------------------
    # FIGURES
    # -------------------------------------------------------------------------
    # Figure 1: Top-5 Jaccard Distribution vs Random & Noise Baselines
    fig1, ax1 = plt.subplots(figsize=(7, 4.5), dpi=300)
    ptq_iou5 = [x for x in ptq_drift['per_instance']['top5_jaccard'] if np.isfinite(x)]
    qat_iou5 = [x for x in qat_drift['per_instance']['top5_jaccard'] if np.isfinite(x)]
    
    parts = ax1.violinplot([ptq_iou5, qat_iou5], positions=[1, 2], showmeans=True, showmedians=True)
    ax1.axhline(ptq_drift['random_baseline_k5'], color='red', linestyle='--', linewidth=1.5, label=f"Combinatorial Chance Baseline ({ptq_drift['random_baseline_k5']:.3f})")
    ax1.axhline(noise_jacc, color='green', linestyle=':', linewidth=1.8, label=f"Explainer Noise Threshold ({noise_jacc:.3f})")
    
    ax1.set_xticks([1, 2])
    ax1.set_xticklabels(['FP32 vs INT8-PTQ', 'FP32 vs INT8-QAT'], fontsize=11)
    ax1.set_ylabel('Top-5 Jaccard Similarity ($IoU_5$)', fontsize=11)
    ax1.set_title('Distribution of Top-5 Feature Attribution Overlap', fontsize=12, fontweight='bold')
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(loc='lower left', fontsize=9)
    plt.tight_layout()
    fig1_path = os.path.join(figures_dir, "figure1_attribution_overlap_distribution.png")
    fig1.savefig(fig1_path)
    plt.close(fig1)

    # Figure 2: Total Variation and Cosine Similarity Distributions
    fig2, (ax2a, ax2b) = plt.subplots(1, 2, figsize=(10, 4), dpi=300)
    tv_vals = [x for x in ptq_drift['per_instance']['total_variation'] if np.isfinite(x)]
    cos_vals = [x for x in ptq_drift['per_instance']['cosine_sim'] if np.isfinite(x)]

    ax2a.hist(tv_vals, bins=20, color='royalblue', edgecolor='black', alpha=0.7)
    ax2a.set_xlabel('Total Variation Distance (TV)', fontsize=10)
    ax2a.set_ylabel('Flow Frequency', fontsize=10)
    ax2a.set_title('Attribution Mass Displacement ($TV$)', fontsize=11, fontweight='bold')
    ax2a.grid(True, linestyle='--', alpha=0.4)

    ax2b.hist(cos_vals, bins=20, color='darkorange', edgecolor='black', alpha=0.7)
    ax2b.set_xlabel('Signed Cosine Similarity ($S_C$)', fontsize=10)
    ax2b.set_ylabel('Flow Frequency', fontsize=10)
    ax2b.set_title('Attribution Direction Agreement ($S_C$)', fontsize=11, fontweight='bold')
    ax2b.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()
    fig2_path = os.path.join(figures_dir, "figure2_tv_and_cosine_distributions.png")
    fig2.savefig(fig2_path)
    plt.close(fig2)

    # Figure 3: Latency vs Artifact Size vs Accuracy Tradeoff
    fig3, ax3 = plt.subplots(figsize=(6.5, 4.5), dpi=300)
    models = ['FP32', 'INT8-PTQ', 'INT8-QAT']
    latencies = [
        hw['latency_edge_bs1']['fp32']['median_latency_ms'],
        hw['latency_edge_bs1']['ptq']['median_latency_ms'],
        hw['latency_edge_bs1']['qat']['median_latency_ms']
    ]
    accuracies = [
        pred['fp32_metrics']['accuracy'] * 100,
        pred['ptq_metrics']['accuracy'] * 100,
        pred.get('qat_metrics', {}).get('accuracy', 0.0) * 100
    ]
    sizes = [
        hw['artifact_sizes']['fp32_kb'],
        hw['artifact_sizes']['ptq_kb'],
        hw['artifact_sizes']['qat_kb']
    ]

    colors = ['blue', 'crimson', 'purple']
    scatter = ax3.scatter(latencies, accuracies, s=[s * 2.5 for s in sizes], c=colors, alpha=0.6, edgecolors='black', linewidth=1.5)
    for i, m in enumerate(models):
        ax3.annotate(f"{m}\n({sizes[i]:.0f} KB)", (latencies[i], accuracies[i]), textcoords="offset points", xytext=(10, 5), fontsize=10, fontweight='bold')

    ax3.set_xlabel('Edge BS=1 Median Latency (ms)', fontsize=11)
    ax3.set_ylabel('Test Accuracy (%)', fontsize=11)
    ax3.set_title('Efficiency Tradeoff: Latency vs Accuracy (Bubble Size = Model Size)', fontsize=11, fontweight='bold')
    ax3.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    fig3_path = os.path.join(figures_dir, "figure3_efficiency_tradeoff.png")
    fig3.savefig(fig3_path)
    plt.close(fig3)

    print(f"Generated all markdown tables in: {tables_dir}")
    print(f"Generated all figures in: {figures_dir}")
