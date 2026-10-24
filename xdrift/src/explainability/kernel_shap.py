"""KernelSHAP explanation and drift analysis module (Sections 11, 12, 13, 15).

Features:
- Uniform wrapper querying the deployed inference path of FP32, PTQ, and QAT
- Explains the exact same scalar target: pre-sigmoid logit for VPN class
- Background dataset sampled strictly from training data
- Paired random seeds and sample budgets
- Explainer stability estimation across multiple random seeds
- Full attribution drift metrics: Top-5/10 Jaccard, Spearman rank, TV distance,
  Signed Cosine similarity, Masked Temporal displacement, and Random Baseline
"""

import os
import time
import json
from typing import Dict, List, Tuple, Any, Optional, Callable

import numpy as np
import torch
import shap

from src.evaluation.attribution_metrics import (
    compute_topk_jaccard,
    compute_spearman_rank_correlation,
    compute_total_variation_distance,
    compute_signed_cosine_similarity,
    compute_temporal_attribution_displacement,
    compute_expected_random_jaccard
)


def flatten_tensor_to_features(X: np.ndarray) -> np.ndarray:
    """Convert (N, 2, 64) tensor into (N, 128) feature matrix.
    
    Cols 0-63: Directional Packet Length (Channel 1)
    Cols 64-127: Inter-arrival Time (Channel 2)
    """
    assert X.ndim == 3 and X.shape[1] == 2 and X.shape[2] == 64
    N = X.shape[0]
    flat = np.zeros((N, 128), dtype=np.float32)
    flat[:, :64] = X[:, 0, :]
    flat[:, 64:] = X[:, 1, :]
    return flat


def reconstruct_features_to_tensor(flat: np.ndarray) -> torch.Tensor:
    """Reconstruct (B, 128) feature matrix back to (B, 2, 64) PyTorch tensor."""
    B = flat.shape[0]
    t = np.zeros((B, 2, 64), dtype=np.float32)
    t[:, 0, :] = flat[:, :64]
    t[:, 1, :] = flat[:, 64:]
    return torch.from_numpy(t).float()


def make_predict_fn(model: torch.nn.Module, device: str = "cpu") -> Callable[[np.ndarray], np.ndarray]:
    """Create a black-box wrapper querying model pre-sigmoid logits for KernelSHAP.
    
    Per Section 11.2: Explains the exact scalar pre-sigmoid logit for the VPN class.
    """
    model.eval()
    dev = torch.device(device)

    def predict_fn(x_flat: np.ndarray) -> np.ndarray:
        bx = reconstruct_features_to_tensor(x_flat).to(dev)
        with torch.no_grad():
            out = model(bx)
        return out.cpu().numpy().flatten()

    return predict_fn


def sample_background_dataset(
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_samples: int = 50,
    seed: int = 42
) -> np.ndarray:
    """Sample representative background dataset strictly from training data (Section 11.1).
    
    Performs stratified sampling between VPN and Non-VPN classes.
    """
    rng = np.random.RandomState(seed)
    idx_vpn = np.where(y_train == 1)[0]
    idx_nonvpn = np.where(y_train == 0)[0]

    n_vpn = min(n_samples // 2, len(idx_vpn))
    n_nonvpn = n_samples - n_vpn

    sampled_vpn = rng.choice(idx_vpn, size=n_vpn, replace=False)
    sampled_nonvpn = rng.choice(idx_nonvpn, size=n_nonvpn, replace=False)

    sampled_idx = np.concatenate([sampled_vpn, sampled_nonvpn])
    rng.shuffle(sampled_idx)

    bg_tensors = X_train[sampled_idx]
    return flatten_tensor_to_features(bg_tensors)


def explain_single_model(
    model: torch.nn.Module,
    background_matrix: np.ndarray,
    instances_matrix: np.ndarray,
    nsamples: int = 200,
    seed: int = 42
) -> Tuple[np.ndarray, float]:
    """Compute KernelSHAP attribution values for a cohort of instances."""
    np.random.seed(seed)
    predict_fn = make_predict_fn(model)

    explainer = shap.KernelExplainer(predict_fn, background_matrix, seed=seed)
    t0 = time.time()
    shap_vals = explainer.shap_values(instances_matrix, nsamples=nsamples, silent=True)
    duration = time.time() - t0

    shap_matrix = np.array(shap_vals)
    if shap_matrix.ndim == 3 and shap_matrix.shape[0] == 1:
        shap_matrix = shap_matrix[0]

    return shap_matrix, duration


def compute_pairwise_drift_metrics(
    phi_ref: np.ndarray,
    phi_target: np.ndarray,
    masks: np.ndarray,
    k_primary: int = 5,
    k_secondary: int = 10
) -> Dict[str, Any]:
    """Compute per-instance and summary attribution drift metrics between two models (Section 12).
    
    phi_ref, phi_target: (N, 128) attribution matrices.
    masks: (N, 64) validity masks.
    """
    N = len(phi_ref)
    assert len(phi_target) == N

    iou_k5_list = []
    iou_k10_list = []
    spearman_list = []
    tv_list = []
    cosine_list = []
    early_ratio_ref_list = []
    early_ratio_target_list = []

    # Active feature mask per instance (128 features: 64 length + 64 IAT)
    active_masks = np.concatenate([masks, masks], axis=1)

    for i in range(N):
        p_ref = phi_ref[i]
        p_tgt = phi_target[i]
        act_m = active_masks[i]
        val_m = masks[i]

        # Top-k Jaccard
        iou5 = compute_topk_jaccard(p_ref, p_tgt, k=k_primary, active_mask=act_m)
        iou10 = compute_topk_jaccard(p_ref, p_tgt, k=k_secondary, active_mask=act_m)
        iou_k5_list.append(iou5)
        iou_k10_list.append(iou10)

        # Spearman rank correlation
        spearman = compute_spearman_rank_correlation(p_ref, p_tgt, active_mask=act_m)
        spearman_list.append(spearman)

        # Total Variation distance
        tv = compute_total_variation_distance(p_ref, p_tgt, active_mask=act_m)
        tv_list.append(tv)

        # Signed Cosine similarity
        cos_sim = compute_signed_cosine_similarity(p_ref, p_tgt, active_mask=act_m)
        cosine_list.append(cos_sim)

        # Temporal attribution displacement
        temp_ref = compute_temporal_attribution_displacement(p_ref, val_m)
        temp_tgt = compute_temporal_attribution_displacement(p_tgt, val_m)
        early_ratio_ref_list.append(temp_ref['early_ratio'])
        early_ratio_target_list.append(temp_tgt['early_ratio'])

    # Clean nan values for aggregate statistics
    def safe_stats(arr: List[float]) -> Dict[str, float]:
        v = np.array(arr)
        valid = v[np.isfinite(v)]
        if len(valid) == 0:
            return {'mean': float('nan'), 'median': float('nan'), 'std': float('nan'), 'valid_count': 0}
        return {
            'mean': float(np.mean(valid)),
            'median': float(np.median(valid)),
            'std': float(np.std(valid)),
            'valid_count': int(len(valid))
        }

    # Combinatorial Random Overlap Baselines (Section 15)
    random_k5 = compute_expected_random_jaccard(k=k_primary, universe_size=128)
    random_k10 = compute_expected_random_jaccard(k=k_secondary, universe_size=128)

    summary = {
        'count': N,
        'random_baseline_k5': random_k5,
        'random_baseline_k10': random_k10,
        'top5_jaccard': safe_stats(iou_k5_list),
        'top10_jaccard': safe_stats(iou_k10_list),
        'spearman_correlation': safe_stats(spearman_list),
        'total_variation_distance': safe_stats(tv_list),
        'signed_cosine_similarity': safe_stats(cosine_list),
        'temporal_early_ratio_ref': safe_stats(early_ratio_ref_list),
        'temporal_early_ratio_target': safe_stats(early_ratio_target_list),
        'per_instance': {
            'top5_jaccard': iou_k5_list,
            'top10_jaccard': iou_k10_list,
            'spearman': spearman_list,
            'total_variation': tv_list,
            'cosine_sim': cosine_list
        }
    }
    return summary
