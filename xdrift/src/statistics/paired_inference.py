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
    seed: int = 42
) -> Dict[str, float]:
    """Calculate cluster-resampled bootstrap confidence interval by capture/session group.
    
    Per Section 14.1: Resamples capture groups with replacement rather than individual flows.
    """
    rng = np.random.RandomState(seed)
    valid_mask = np.isfinite(diffs)
    valid_diffs = diffs[valid_mask]
    valid_caps = capture_ids[valid_mask]

    unique_groups = np.unique(valid_caps)
    G = len(unique_groups)

    if G < 2:
        return {'mean': float(np.mean(valid_diffs)), 'ci_lower': float('nan'), 'ci_upper': float('nan'), 'groups': G}

    # Group index mapping
    group_map = {g: np.where(valid_caps == g)[0] for g in unique_groups}

    boot_means = []
    for _ in range(n_boot):
        resampled_groups = rng.choice(unique_groups, size=G, replace=True)
        sample_indices = []
        for g in resampled_groups:
            sample_indices.extend(group_map[g])
        if len(sample_indices) > 0:
            boot_means.append(np.mean(valid_diffs[sample_indices]))

    alpha = (1.0 - ci) / 2.0
    lower = float(np.percentile(boot_means, 100.0 * alpha))
    upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha)))

    return {
        'mean': float(np.mean(valid_diffs)),
        'ci_lower': lower,
        'ci_upper': upper,
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
        # z-approximation: r = z / sqrt(N)
        # scipy wilcoxon returns statistic and pvalue
        p_val = float(w_res.pvalue)
        # Approximate z from p-value using normal inverse
        from scipy.stats import norm
        z_stat = norm.ppf(1.0 - p_val / 2.0) if p_val > 0 else 10.0
        r_eff = float(abs(z_stat) / math.sqrt(N))
    except Exception:
        p_val = float('nan')
        r_eff = float('nan')

    return {
        'cohens_d': cohens_d,
        'wilcoxon_r': r_eff,
        'wilcoxon_pvalue': p_val
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
