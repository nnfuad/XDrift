"""Statistical Inference and Hypothesis Testing for XDrift Protocol (Section 14).

Implements:
- Paired Wilcoxon signed-rank tests and paired t-tests
- Nonparametric paired bootstrap for mean/median differences (95% CIs)
- Group-clustered bootstrap accounting for capture/session dependence
- Effect size calculations (Cohen's d and Wilcoxon r)
- Holm multiple-testing correction
- Subgroup stratification (VPN vs Non-VPN, short vs long flows, agreement vs disagreement)
"""

import math
from typing import Dict, List, Tuple, Any, Optional

import numpy as np
from scipy.stats import wilcoxon, ttest_rel


def paired_bootstrap_ci(
    diffs: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42
) -> Dict[str, float]:
    """Calculate 95% bootstrap confidence interval for paired differences."""
    rng = np.random.RandomState(seed)
    valid_diffs = diffs[np.isfinite(diffs)]
    N = len(valid_diffs)
    if N == 0:
        return {'mean': float('nan'), 'ci_lower': float('nan'), 'ci_upper': float('nan')}

    boot_means = []
    for _ in range(n_boot):
        sample = rng.choice(valid_diffs, size=N, replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))

    return {
        'mean': float(np.mean(valid_diffs)),
        'median': float(np.median(valid_diffs)),
        'std': float(np.std(valid_diffs)),
        'ci_lower': lower,
        'ci_upper': upper,
        'sample_size': N
    }


def group_cluster_bootstrap_ci(
    diffs: np.ndarray,
    capture_ids: np.ndarray,
    n_boot: int = 1000,
    ci: float = 0.95,
    seed: int = 42,
    map_canonical: bool = True
) -> Dict[str, float]:
    """Calculate cluster-resampled bootstrap confidence interval by capture/session group.
    
    Per Section 14.1: Resamples capture groups with replacement rather than individual flows.
    Supports both:
      - Estimand A: flow-weighted mean across clusters (pooling flows from sampled clusters)
      - Estimand B: unweighted mean of cluster means (equal weight per session group)
    """
    rng = np.random.RandomState(seed)
    valid_mask = np.isfinite(diffs)
    valid_diffs = diffs[valid_mask]
    raw_caps = capture_ids[valid_mask]

    if map_canonical:
        from src.data.leakage_free_split import get_session_group
        valid_caps = np.array([get_session_group(str(c)) for c in raw_caps])
    else:
        valid_caps = raw_caps

    unique_groups = np.unique(valid_caps)
    G = len(unique_groups)

    if G < 2:
        pt = float(np.mean(valid_diffs))
        return {
            'mean': pt, 'ci_lower': float('nan'), 'ci_upper': float('nan'),
            'mean_flow_weighted': pt, 'ci_lower_flow_weighted': float('nan'), 'ci_upper_flow_weighted': float('nan'),
            'mean_group_unweighted': pt, 'ci_lower_group_unweighted': float('nan'), 'ci_upper_group_unweighted': float('nan'),
            'groups': G, 'sample_size': int(len(valid_diffs))
        }

    # Group index mapping and precomputed group means/counts
    group_map = {g: np.where(valid_caps == g)[0] for g in unique_groups}
    group_means = {g: np.mean(valid_diffs[group_map[g]]) for g in unique_groups}
    group_counts = {g: len(group_map[g]) for g in unique_groups}

    boot_means_flow = []
    boot_means_unw = []

    for _ in range(n_boot):
        resampled_groups = rng.choice(unique_groups, size=G, replace=True)
        # Estimand A: pooled flow-weighted
        tot_sum = sum(group_means[g] * group_counts[g] for g in resampled_groups)
        tot_cnt = sum(group_counts[g] for g in resampled_groups)
        if tot_cnt > 0:
            boot_means_flow.append(tot_sum / tot_cnt)
        # Estimand B: unweighted group means
        boot_means_unw.append(np.mean([group_means[g] for g in resampled_groups]))

    alpha = (1.0 - ci) / 2.0
    lower_flow = float(np.percentile(boot_means_flow, 100.0 * alpha))
    upper_flow = float(np.percentile(boot_means_flow, 100.0 * (1.0 - alpha)))
    lower_unw = float(np.percentile(boot_means_unw, 100.0 * alpha))
    upper_unw = float(np.percentile(boot_means_unw, 100.0 * (1.0 - alpha)))

    pt_flow = float(np.mean(valid_diffs))
    pt_unw = float(np.mean([group_means[g] for g in unique_groups]))

    return {
        'mean': pt_flow,
        'ci_lower': lower_flow,
        'ci_upper': upper_flow,
        'mean_flow_weighted': pt_flow,
        'ci_lower_flow_weighted': lower_flow,
        'ci_upper_flow_weighted': upper_flow,
        'mean_group_unweighted': pt_unw,
        'ci_lower_group_unweighted': lower_unw,
        'ci_upper_group_unweighted': upper_unw,
        'groups': int(G),
        'sample_size': int(len(valid_diffs))
    }


def compute_paired_effect_sizes(diffs: np.ndarray) -> Dict[str, float]:
    """Compute Cohen's d and Wilcoxon effect size r."""
    valid = diffs[np.isfinite(diffs)]
    N = len(valid)
    if N < 2:
        return {'cohens_d': float('nan'), 'wilcoxon_r': float('nan')}

    # Cohen's d for paired samples
    mean_d = np.mean(valid)
    std_d = np.std(valid, ddof=1)
    cohens_d = float(mean_d / std_d) if std_d > 1e-12 else 0.0

    # Wilcoxon signed-rank test
    try:
        w_res = wilcoxon(valid, alternative='two-sided')
        w_greater = wilcoxon(valid, alternative='greater')
        w_less = wilcoxon(valid, alternative='less')
        p_val = float(w_res.pvalue)
        w_stat = float(w_res.statistic)
        p_greater = float(w_greater.pvalue)
        p_less = float(w_less.pvalue)

        # Signed rank sums
        nonzero = valid[valid != 0]
        if len(nonzero) > 0:
            from scipy.stats import rankdata
            ranks = rankdata(np.abs(nonzero))
            r_plus = float(np.sum(ranks[nonzero > 0]))
            r_minus = float(np.sum(ranks[nonzero < 0]))
        else:
            r_plus = 0.0
            r_minus = 0.0

        # z-approximation: r = z / sqrt(N)
        from scipy.stats import norm
        z_stat = norm.ppf(1.0 - p_val / 2.0) if p_val > 0 else 10.0
        r_eff = float(abs(z_stat) / math.sqrt(N))
    except Exception:
        p_val = float('nan')
        w_stat = float('nan')
        r_plus = float('nan')
        r_minus = float('nan')
        p_greater = float('nan')
        p_less = float('nan')
        r_eff = float('nan')

    return {
        'cohens_d': cohens_d,
        'wilcoxon_r': r_eff,
        'wilcoxon_pvalue': p_val,
        'wilcoxon_stat': w_stat,
        'wilcoxon_r_plus': r_plus,
        'wilcoxon_r_minus': r_minus,
        'wilcoxon_greater_p': p_greater,
        'wilcoxon_less_p': p_less
    }


def apply_holm_correction(p_values: List[float]) -> List[float]:
    """Apply Holm-Bonferroni step-down correction for multiple testing."""
    m = len(p_values)
    indexed_p = sorted(enumerate(p_values), key=lambda x: x[1])
    corrected = [0.0] * m

    cum_max = 0.0
    for rank, (orig_idx, p) in enumerate(indexed_p):
        k = rank + 1
        adj_p = min(1.0, (m - k + 1) * p)
        cum_max = max(cum_max, adj_p)
        corrected[orig_idx] = float(min(1.0, cum_max))

    return corrected
