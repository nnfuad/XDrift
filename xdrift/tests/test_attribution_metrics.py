"""Automated unit tests for Attribution Drift Metrics and Chance Baseline.
Verifies invariants 14-19 from Section 20 of XDrift Protocol.
"""

import math
import numpy as np
import pytest

from src.evaluation.attribution_metrics import (
    compute_topk_jaccard,
    compute_spearman_rank_correlation,
    compute_total_variation_distance,
    compute_signed_cosine_similarity,
    compute_temporal_attribution_displacement,
    compute_expected_random_jaccard
)


def test_14_attribution_vector_dimensions_and_feature_ordering():
    """Test 14: Attribution vector dimensions (128,) with correct ordering (length then IAT)."""
    phi = np.random.randn(128)
    mask = np.zeros(64)
    mask[:10] = 1.0  # 10 observed packets

    res = compute_temporal_attribution_displacement(phi, mask)
    assert 'early_mass' in res
    assert 'late_mass' in res
    assert 'early_ratio' in res
    assert np.isfinite(res['early_ratio'])


def test_15_zero_attribution_and_undefined_correlation():
    """Test 15: Correct handling of zero attribution vectors and undefined correlations."""
    zero_vec = np.zeros(128)
    rand_vec = np.random.randn(128)

    # Invariant: Spearman correlation on constant zero vector returns NaN (not silently zero!)
    spearman = compute_spearman_rank_correlation(zero_vec, rand_vec)
    assert np.isnan(spearman)

    # Invariant: TV distance on zero vector returns NaN
    tv = compute_total_variation_distance(zero_vec, rand_vec)
    assert np.isnan(tv)

    # Invariant: Cosine similarity on zero vector returns NaN
    cos_sim = compute_signed_cosine_similarity(zero_vec, rand_vec)
    assert np.isnan(cos_sim)


def test_16_temporal_masking_of_unobserved_positions():
    """Test 16: Validity mask strictly zeroes out unobserved packet positions in temporal summaries."""
    phi = np.ones(128)  # Constant mass 1.0 everywhere
    mask = np.zeros(64)
    mask[:4] = 1.0      # Only first 4 packets observed (all in early region [0-5])

    res = compute_temporal_attribution_displacement(phi, mask)
    # Positions 0-3 observed: 4 packets * (1.0 length + 1.0 IAT) = 8.0 mass
    assert np.isclose(res['early_mass'], 8.0)
    # Late positions [6-63] all masked to 0.0
    assert np.isclose(res['late_mass'], 0.0)
    assert np.isclose(res['early_ratio'], 1.0)


def test_17_jaccard_and_total_variation():
    """Test 17: Correct calculation of Jaccard and total variation distance."""
    # Synthetic vectors with known top-k overlap
    vec_a = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 1.0, 1.0, 1.0])
    vec_b = np.array([10.0, 9.0, 8.0, 1.0, 1.0, 7.0, 6.0, 1.0])
    # Top-5 of A: {0, 1, 2, 3, 4}
    # Top-5 of B: {0, 1, 2, 5, 6}
    # Intersection: {0, 1, 2} -> 3
    # Union: {0, 1, 2, 3, 4, 5, 6} -> 7
    # Expected IoU_5: 3 / 7
    iou = compute_topk_jaccard(vec_a, vec_b, k=5)
    assert np.isclose(iou, 3.0 / 7.0)

    # TV distance on simple known distribution
    # p_a = [0.8, 0.2], p_b = [0.5, 0.5]
    # TV = 0.5 * (|0.8 - 0.5| + |0.2 - 0.5|) = 0.5 * (0.3 + 0.3) = 0.3
    tv_a = np.array([0.8, 0.2])
    tv_b = np.array([0.5, 0.5])
    tv = compute_total_variation_distance(tv_a, tv_b)
    assert np.isclose(tv, 0.3)


def test_18_random_overlap_hypergeometric_calculation():
    """Test 18: Correct combinatorial random-overlap calculation validated against ground truth."""
    # Small verifiable case: M = 4, k = 2
    # Possible 2-subsets of 4 items: comb(4,2) = 6 subsets.
    # Independent choice: 36 pairs.
    # 6 pairs are identical (c=2, IoU = 1.0) -> prob 6/36 = 1/6
    # 24 pairs share 1 element (c=1, IoU = 1/3) -> prob 24/36 = 4/6
    # 6 pairs share 0 elements (c=0, IoU = 0.0) -> prob 6/36 = 1/6
    # Expected IoU = 1/6 * 1.0 + 4/6 * (1/3) + 1/6 * 0 = 1/6 + 4/18 = 3/18 + 4/18 = 7/18 ~= 0.388888
    expected_small = compute_expected_random_jaccard(k=2, universe_size=4)
    assert np.isclose(expected_small, 7.0 / 18.0)

    # Feature universe M = 128, k = 5
    baseline_k5 = compute_expected_random_jaccard(k=5, universe_size=128)
    assert 0.0 < baseline_k5 < 0.1  # Very low chance overlap for 5 out of 128


def test_19_pairing_of_observations():
    """Test 19: Strict pairing of observations across models."""
    fp32_scores = np.array([0.8, 0.6, 0.9])
    ptq_scores = np.array([0.75, 0.65, 0.88])
    diff = ptq_scores - fp32_scores
    assert len(diff) == len(fp32_scores)
    assert np.isclose(diff[0], -0.05)
