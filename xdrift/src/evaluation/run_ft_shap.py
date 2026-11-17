"""Compute KernelSHAP attributions for FP32 Fine-Tuned Baseline."""

import json
import time
import numpy as np
import torch

from src.models.cnn1d import Compact1DCNN
from src.explainability.kernel_shap import (
    flatten_tensor_to_features,
    sample_background_dataset,
    explain_single_model,
    compute_pairwise_drift_metrics
)

# 1. Load models
ft_ckpt = torch.load('xdrift/artifacts/checkpoints/fp32_finetuned_seed_42.pt', map_location='cpu')
model_ft = Compact1DCNN()
model_ft.load_state_dict(ft_ckpt['state_dict'])
model_ft.eval()

# 2. Load cohort data
with open('xdrift/results/metrics/cohort_indices.json') as f:
    cohort_meta = json.load(f)
cohort_idx = [item['test_index'] for item in cohort_meta]

d_test = np.load('xdrift/data/processed/test_data.npz')
X_cohort = d_test['X'][cohort_idx]  # (100, 2, 64)
masks_cohort = d_test['masks'][cohort_idx]  # (100, 64)

d_cohort = np.load('xdrift/results/attributions/cohort_attributions.npz')
phi_fp32 = d_cohort['phi_fp32']  # (100, 128)

# 3. Sample background
d_train = np.load('xdrift/data/processed/train_data.npz')
bg_matrix = sample_background_dataset(d_train['X'], d_train['y'], n_samples=50, seed=42)
instances_matrix = flatten_tensor_to_features(X_cohort)

print("Starting KernelSHAP on 100 cohort flows with nsamples=150...")
t0 = time.time()
phi_ft, shap_duration = explain_single_model(
    model=model_ft,
    background_matrix=bg_matrix,
    instances_matrix=instances_matrix,
    nsamples=150,
    seed=42
)
print(f"KernelSHAP completed in {shap_duration:.2f} seconds.")

# 4. Compute pairwise metrics
drift = compute_pairwise_drift_metrics(
    phi_ref=phi_fp32,
    phi_target=phi_ft,
    masks=masks_cohort,
    k_primary=5,
    k_secondary=10
)

t5 = drift['top5_jaccard']
t10 = drift['top10_jaccard']
tv = drift['total_variation_distance']
sc = drift['signed_cosine_similarity']
sp = drift['spearman_correlation']
pi = drift['per_instance']

print("\n=== ATTRIBUTION COMPARISON: FP32 BASE vs FP32-FT ===")
print(f"Top-5 Jaccard (IoU_5): {t5['mean']:.4f} ± {t5['std']:.4f} (Median: {t5['median']:.4f})")
print(f"Top-10 Jaccard (IoU_10): {t10['mean']:.4f} ± {t10['std']:.4f} (Median: {t10['median']:.4f})")
print(f"Total Variation (TV): {tv['mean']:.4f} ± {tv['std']:.4f} (Median: {tv['median']:.4f})")
print(f"Signed Cosine (S_C): {sc['mean']:.4f} ± {sc['std']:.4f} (Median: {sc['median']:.4f})")
print(f"Spearman rho: {sp['mean']:.4f} ± {sp['std']:.4f} (Median: {sp['median']:.4f})")

# Predictive metrics from earlier verified run
pred_summary = {
    'roc_auc': 0.9819,
    'pr_auc': 0.6128,
    'brier_score': 0.0248,
    'calibrated_threshold': 0.830,
    'calibrated_f1': 0.8176,
    'calibrated_recall': 0.9681,
    'calibrated_precision': 0.7077,
    'calibrated_accuracy': 0.9764,
    'calibrated_balanced_accuracy': 0.9725,
    'reused_f1': 0.3771,
    'reused_recall': 0.3071,
    'reused_precision': 0.4885,
    'reused_accuracy': 0.9445,
    'reused_positive_calls': 1693
}

full_summary = {
    'model': 'FP32-FineTuned',
    'predictive_metrics': pred_summary,
    'attribution_metrics_vs_fp32': {
        'mean_iou5': t5['mean'],
        'std_iou5': t5['std'],
        'median_iou5': t5['median'],
        'mean_iou10': t10['mean'],
        'std_iou10': t10['std'],
        'median_iou10': t10['median'],
        'mean_tv': tv['mean'],
        'std_tv': tv['std'],
        'median_tv': tv['median'],
        'mean_sc': sc['mean'],
        'std_sc': sc['std'],
        'median_sc': sc['median'],
        'mean_rho': sp['mean'],
        'std_rho': sp['std'],
        'median_rho': sp['median']
    }
}

with open('xdrift/results/metrics/fp32_ft_evaluation.json', 'w') as f:
    json.dump(full_summary, f, indent=2)

np.savez_compressed(
    'xdrift/results/attributions/cohort_attributions_ft.npz',
    phi_ft=phi_ft,
    iou_k5=np.array(pi['top5_jaccard']),
    iou_k10=np.array(pi['top10_jaccard']),
    tv=np.array(pi['total_variation']),
    cosine=np.array(pi['cosine_sim']),
    spearman=np.array(pi['spearman'])
)

print("Saved fp32_ft_evaluation.json and cohort_attributions_ft.npz successfully!")
