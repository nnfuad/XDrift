"""Evaluation and Attribution Script for Unquantized FP32 Fine-Tuning Baseline.

Computes:
1. Predictive performance on held-out test split (N=49,251)
2. Optimal decision threshold on validation split (N=69,564)
3. KernelSHAP feature attributions on 100-flow test cohort
4. Attribution alignment metrics (Top-5 Jaccard, Total Variation, Signed Cosine, Spearman)
   against the base FP32 model and against the noise floor.
"""

import os
import sys
# Prevent NumPy 2.x incompatible pyarrow/numexpr/bottleneck from breaking pandas/shap
sys.modules['pyarrow'] = None
sys.modules['numexpr'] = None
sys.modules['bottleneck'] = None

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


def evaluate_fp32_ft():
    print("=== Evaluating FP32 Fine-Tuned Baseline ===")
    
    # Paths
    ft_ckpt_path = "xdrift/artifacts/checkpoints/fp32_finetuned_seed_42.pt"
    base_ckpt_path = "xdrift/artifacts/checkpoints/fp32_seed_42.pt"
    val_path = "xdrift/data/processed/val_data.npz"
    test_path = "xdrift/data/processed/test_data.npz"
    train_path = "xdrift/data/processed/train_data.npz"
    
    # 1. Load models
    ft_ckpt = torch.load(ft_ckpt_path, map_location='cpu')
    model_ft = Compact1DCNN()
    model_ft.load_state_dict(ft_ckpt['state_dict'])
    model_ft.eval()

    base_ckpt = torch.load(base_ckpt_path, map_location='cpu')
    model_base = Compact1DCNN()
    model_base.load_state_dict(base_ckpt['state_dict'])
    model_base.eval()

    # Helper for memory-safe batched forward inference
    def batched_forward(model, X_arr, batch_size=2048):
        logits_list = []
        with torch.no_grad():
            for i in range(0, len(X_arr), batch_size):
                bx = torch.tensor(X_arr[i:i+batch_size], dtype=torch.float32)
                bl = model(bx).squeeze(-1).cpu().numpy()
                logits_list.append(bl)
        return np.concatenate(logits_list)

    # 2. Predictive Performance on Validation (Threshold search)
    d_val = np.load(val_path)
    X_val = d_val['X']
    y_val = d_val['y']

    val_logits = batched_forward(model_ft, X_val)
    val_probs = 1.0 / (1.0 + np.exp(-val_logits))

    thresholds = np.linspace(0.1, 0.9, 81)
    best_f1 = -1.0
    best_tau = 0.5
    for tau in thresholds:
        preds = (val_probs >= tau).astype(int)
        tp = np.sum((preds == 1) & (y_val == 1))
        fp = np.sum((preds == 1) & (y_val == 0))
        fn = np.sum((preds == 0) & (y_val == 1))
        p = tp / max(1, tp + fp)
        r = tp / max(1, tp + fn)
        f1 = 2 * p * r / max(1e-9, p + r)
        if f1 > best_f1:
            best_f1 = f1
            best_tau = float(tau)

    print(f"Validation Optimal Threshold: tau = {best_tau:.3f} (Val F1 = {best_f1:.4f})")

    # 3. Test Evaluation (N=49,251)
    d_test = np.load(test_path)
    X_test = d_test['X']
    y_test = d_test['y']

    test_logits = batched_forward(model_ft, X_test)
    test_probs = 1.0 / (1.0 + np.exp(-test_logits))

    # Fast manual ROC-AUC & PR-AUC calculation to avoid any numpy/sklearn dependency warnings
    # Sort by test_probs descending
    sort_idx = np.argsort(-test_probs)
    y_sorted = y_test[sort_idx]
    n_pos = np.sum(y_test == 1)
    n_neg = np.sum(y_test == 0)

    # ROC-AUC (Wilcoxon rank-sum equivalent)
    ranks = np.arange(len(y_test), 0, -1)
    pos_ranks = ranks[y_sorted == 1]
    roc_auc = float((np.sum(pos_ranks) - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))

    # PR-AUC
    tps = np.cumsum(y_sorted == 1)
    fps = np.cumsum(y_sorted == 0)
    precisions = tps / (tps + fps)
    recalls = tps / n_pos
    # Add point (0, 1) and calculate trapezoidal AUC
    rec_diff = np.diff(np.concatenate([[0.0], recalls]))
    pr_auc = float(np.sum(precisions * rec_diff))

    brier_score = float(np.mean((test_probs - y_test) ** 2))

    # Evaluated at calibrated threshold
    preds_cal = (test_probs >= best_tau).astype(int)
    tp_c = int(np.sum((preds_cal == 1) & (y_test == 1)))
    fp_c = int(np.sum((preds_cal == 1) & (y_test == 0)))
    tn_c = int(np.sum((preds_cal == 0) & (y_test == 0)))
    fn_c = int(np.sum((preds_cal == 0) & (y_test == 1)))
    prec_c = float(tp_c / max(1, tp_c + fp_c))
    rec_c = float(tp_c / max(1, tp_c + fn_c))
    f1_c = float(2 * prec_c * rec_c / max(1e-9, prec_c + rec_c))
    acc_c = float((tp_c + tn_c) / len(y_test))

    # Evaluated at reused FP32 threshold (0.900)
    preds_reused = (test_probs >= 0.900).astype(int)
    tp_r = int(np.sum((preds_reused == 1) & (y_test == 1)))
    fp_r = int(np.sum((preds_reused == 1) & (y_test == 0)))
    tn_r = int(np.sum((preds_reused == 0) & (y_test == 0)))
    fn_r = int(np.sum((preds_reused == 0) & (y_test == 1)))
    prec_r = float(tp_r / max(1, tp_r + fp_r))
    rec_r = float(tp_r / max(1, tp_r + fn_r))
    f1_r = float(2 * prec_r * rec_r / max(1e-9, prec_r + rec_r))
    acc_r = float((tp_r + tn_r) / len(y_test))

    print(f"Test ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f}, Brier: {brier_score:.4f}")
    print(f"Calibrated (tau={best_tau:.3f}): F1={f1_c:.4f}, Recall={rec_c:.4f}, Prec={prec_c:.4f}, Acc={acc_c*100:.2f}%")
    print(f"Reused (tau=0.900): F1={f1_r:.4f}, Recall={rec_r:.4f}, Prec={prec_r:.4f}, Acc={acc_r*100:.2f}%")

    # 4. KernelSHAP on 100-flow test cohort
    cohort_path = "xdrift/results/attributions/cohort_attributions.npz"
    d_cohort = np.load(cohort_path, allow_pickle=True)
    masks_cohort = d_cohort['masks']  # (100, 64)
    phi_fp32 = d_cohort['phi_fp32']  # (100, 128)

    # Reconstruct exact cohort input tensors from test_data using matched flow IDs from cohort_attributions
    d_test_raw = np.load(test_path, allow_pickle=True)
    test_fid_map = {fid: idx for idx, fid in enumerate(d_test_raw['flow_ids'])}
    cohort_idx = [test_fid_map[fid] for fid in d_cohort['flow_ids']]
    X_cohort = d_test_raw['X'][cohort_idx]  # (100, 2, 64)

    print("\nRunning KernelSHAP on 100-flow cohort for FP32-FT...")
    d_train = np.load(train_path)
    X_train_raw = d_train['X']
    y_train_raw = d_train['y']
    bg_matrix = sample_background_dataset(X_train_raw, y_train_raw, n_samples=50, seed=42)
    instances_matrix = flatten_tensor_to_features(X_cohort)

    phi_ft, shap_duration = explain_single_model(
        model=model_ft,
        background_matrix=bg_matrix,
        instances_matrix=instances_matrix,
        nsamples=150,
        seed=42
    )
    print(f"KernelSHAP completed in {shap_duration:.2f} seconds.")

    # 5. Compute attribution drift metrics between FP32 base and FP32-FT
    drift_metrics = compute_pairwise_drift_metrics(
        phi_ref=phi_fp32,
        phi_target=phi_ft,
        masks=masks_cohort,
        k_primary=5,
        k_secondary=10
    )

    t5_stats = drift_metrics['top5_jaccard']
    t10_stats = drift_metrics['top10_jaccard']
    tv_stats = drift_metrics['total_variation_distance']
    sc_stats = drift_metrics['signed_cosine_similarity']
    sp_stats = drift_metrics['spearman_correlation']
    per_inst = drift_metrics['per_instance']

    print(f"\n--- FP32 Base vs FP32-FT Attribution Metrics ---")
    print(f"Top-5 Jaccard (IoU_5): {t5_stats['mean']:.4f} ± {t5_stats['std']:.4f}")
    print(f"Top-10 Jaccard (IoU_10): {t10_stats['mean']:.4f} ± {t10_stats['std']:.4f}")
    print(f"Total Variation (TV): {tv_stats['mean']:.4f} ± {tv_stats['std']:.4f}")
    print(f"Signed Cosine (S_C): {sc_stats['mean']:.4f} ± {sc_stats['std']:.4f}")
    print(f"Spearman rho: {sp_stats['mean']:.4f} ± {sp_stats['std']:.4f}")

    # Output summary
    summary = {
        'model': 'FP32-FineTuned',
        'predictive_metrics': {
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'brier_score': brier_score,
            'calibrated_threshold': best_tau,
            'calibrated_f1': f1_c,
            'calibrated_recall': rec_c,
            'calibrated_precision': prec_c,
            'calibrated_accuracy': acc_c,
            'reused_f1': f1_r,
            'reused_recall': rec_r,
            'reused_precision': prec_r,
            'reused_accuracy': acc_r
        },
        'attribution_metrics_vs_fp32': {
            'mean_iou5': t5_stats['mean'],
            'std_iou5': t5_stats['std'],
            'mean_iou10': t10_stats['mean'],
            'std_iou10': t10_stats['std'],
            'mean_tv': tv_stats['mean'],
            'std_tv': tv_stats['std'],
            'median_tv': tv_stats['median'],
            'mean_sc': sc_stats['mean'],
            'std_sc': sc_stats['std'],
            'mean_rho': sp_stats['mean'],
            'std_rho': sp_stats['std']
        }
    }

    out_metrics_path = "xdrift/results/metrics/fp32_ft_evaluation.json"
    with open(out_metrics_path, 'w') as f:
        json.dump(summary, f, indent=2)

    out_npz_path = "xdrift/results/attributions/cohort_attributions_ft.npz"
    np.savez_compressed(
        out_npz_path,
        phi_ft=phi_ft,
        iou_k5=np.array(per_inst['top5_jaccard']),
        tv=np.array(per_inst['total_variation']),
        cosine=np.array(per_inst['cosine_sim']),
        spearman=np.array(per_inst['spearman'])
    )

    print(f"Saved evaluation metrics to {out_metrics_path} and attributions to {out_npz_path}")
    return summary


if __name__ == "__main__":
    evaluate_fp32_ft()
