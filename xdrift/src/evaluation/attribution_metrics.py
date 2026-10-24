"""Attribution drift metrics and Chance Baseline module (Sections 12 & 15).

Implements:
- Top-k Jaccard similarity (IoU) with deterministic tie-breaking
- Spearman rank correlation with NaN handling for zero-variance vectors
- Total Variation (TV) distance on L1-normalized attribution mass
- Signed Cosine similarity
- Masked temporal attribution mass displacement (early [0-5] vs late [6-63])
- Exact Hypergeometric combinatorial expected random Jaccard baseline
"""

import math
from typing import Dict, Tuple, Optional, Any
import numpy as np
from scipy.stats import spearmanr


def compute_topk_jaccard(
    phi_a: np.ndarray,
    phi_b: np.ndarray,
    k: int = 5,
    active_mask: Optional[np.ndarray] = None
) -> float:
    """Calculate Jaccard similarity IoU_k = |A n B| / |A u B| of top-k features.
    
    phi_a, phi_b: (128,) attribution vectors.
    active_mask: (128,) boolean or 0/1 mask indicating eligible features.
    """
    abs_a = np.abs(phi_a)
    abs_b = np.abs(phi_b)

    if active_mask is not None:
        abs_a = np.where(active_mask > 0, abs_a, -1.0)
        abs_b = np.where(active_mask > 0, abs_b, -1.0)

    # Deterministic top-k selection (highest attribution first, ties broken by feature index)
    # Using negative values so lexsort / argsort works cleanly
    indices_a = np.argsort(-abs_a, kind='stable')[:k]
    indices_b = np.argsort(-abs_b, kind='stable')[:k]

    set_a = set(indices_a)
    set_b = set(indices_b)

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return np.nan
    return float(intersection) / float(union)


def compute_spearman_rank_correlation(
    phi_a: np.ndarray,
    phi_b: np.ndarray,
    active_mask: Optional[np.ndarray] = None
) -> float:
    """Compute Spearman rank correlation of absolute attribution vectors.
    
    Per Section 12.2: If correlation is undefined (constant/zero vector), return NaN.
    """
    abs_a = np.abs(phi_a)
    abs_b = np.abs(phi_b)

    if active_mask is not None:
        valid_idx = np.where(active_mask > 0)[0]
        if len(valid_idx) < 2:
            return np.nan
        abs_a = abs_a[valid_idx]
        abs_b = abs_b[valid_idx]

    if np.all(abs_a == abs_a[0]) or np.all(abs_b == abs_b[0]):
        return np.nan

    res = spearmanr(abs_a, abs_b)
    return float(res.correlation) if np.isfinite(res.correlation) else np.nan


def compute_total_variation_distance(
    phi_a: np.ndarray,
    phi_b: np.ndarray,
    active_mask: Optional[np.ndarray] = None
) -> float:
    """Compute Total Variation distance on L1-normalized attribution vectors.
    
    p_i = |phi_i| / sum |phi_j|
    TV = 0.5 * sum |p_a,i - p_b,i|
    Per Section 12.3: Returns NaN if either vector has zero total mass.
    """
    abs_a = np.abs(phi_a)
    abs_b = np.abs(phi_b)

    if active_mask is not None:
        abs_a = np.where(active_mask > 0, abs_a, 0.0)
        abs_b = np.where(active_mask > 0, abs_b, 0.0)

    sum_a = np.sum(abs_a)
    sum_b = np.sum(abs_b)

    if sum_a <= 1e-12 or sum_b <= 1e-12:
        return np.nan

    p_a = abs_a / sum_a
    p_b = abs_b / sum_b

    tv = 0.5 * np.sum(np.abs(p_a - p_b))
    return float(np.clip(tv, 0.0, 1.0))


def compute_signed_cosine_similarity(
    phi_a: np.ndarray,
    phi_b: np.ndarray,
    active_mask: Optional[np.ndarray] = None
) -> float:
    """Compute Cosine Similarity between signed attribution vectors.
    
    Returns NaN if either vector has zero L2 norm.
    """
    a = phi_a.copy()
    b = phi_b.copy()

    if active_mask is not None:
        a = np.where(active_mask > 0, a, 0.0)
        b = np.where(active_mask > 0, b, 0.0)

    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)

    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return np.nan

    cos_sim = np.dot(a, b) / (norm_a * norm_b)
    return float(np.clip(cos_sim, -1.0, 1.0))


def compute_temporal_attribution_displacement(
    phi: np.ndarray,
    validity_mask: np.ndarray,
    early_bound: int = 5,
    max_packets: int = 64
) -> Dict[str, float]:
    """Compute masked temporal attribution displacement (Section 12.5).
    
    phi: (128,) attribution vector (first 64 are length, next 64 are IAT).
    validity_mask: (64,) mask of observed packet positions.
    """
    phi_len = phi[:max_packets]
    phi_iat = phi[max_packets:]

    psi = np.abs(phi_len) + np.abs(phi_iat)
    # Mask unobserved packet positions
    masked_psi = psi * validity_mask

    total_mass = np.sum(masked_psi)
    if total_mass <= 1e-12:
        return {
            'early_mass': 0.0,
            'late_mass': 0.0,
            'total_mass': 0.0,
            'early_ratio': np.nan
        }

    early_mass = np.sum(masked_psi[:early_bound + 1])
    late_mass = np.sum(masked_psi[early_bound + 1:])
    early_ratio = early_mass / total_mass

    return {
        'early_mass': float(early_mass),
        'late_mass': float(late_mass),
        'total_mass': float(total_mass),
        'early_ratio': float(early_ratio)
    }


def compute_expected_random_jaccard(k: int, universe_size: int) -> float:
    """Exact combinatorial expected random Jaccard overlap (Section 15).
    
    Calculates E[|A n B| / |A u B|] under hypergeometric distribution
    for two independent random k-subsets from M features.
    """
    M = universe_size
    assert 1 <= k <= M

    # Support of hypergeometric distribution: max(0, 2k - M) <= c <= k
    c_min = max(0, 2 * k - M)
    c_max = k

    expected_jaccard = 0.0
    total_ways = math.comb(M, k)

    for c in range(c_min, c_max + 1):
        # Ways to choose c common features from A, and k - c from M - k
        ways_c = math.comb(k, c) * math.comb(M - k, k - c)
        prob_c = ways_c / total_ways
        union_size = 2 * k - c
        jaccard_c = c / union_size
        expected_jaccard += prob_c * jaccard_c

    return float(expected_jaccard)
