"""
Fidelity Assessor
-----------------
Computes statistical similarity metrics between real and synthetic dataframes:
1. Wasserstein Distance for continuous columns.
2. Jensen-Shannon Divergence for categorical columns.
3. Pairwise Cross-Correlation Difference Matrix:
   - Pearson correlation for continuous-continuous pairs.
   - Cramer's V for categorical-categorical pairs.
   - Correlation Ratio for continuous-categorical pairs.
"""

import logging
import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.spatial.distance import jensenshannon
from typing import List, Dict, Tuple, Optional, Any

logger = logging.getLogger(__name__)


def compute_wasserstein_distance(real_series: pd.Series, synth_series: pd.Series) -> float:
    """Compute Wasserstein-1 distance for a continuous column.

    Both series are MinMax-normalized to [0, 1] using the real data range
    before computing the distance. This ensures scale-invariance so that
    fixed thresholds (e.g. 0.05, 0.15) remain meaningful across columns
    with different value ranges.

    Returns:
        float: Wasserstein-1 distance in [0, 1] normalized space.
    """
    r_vals = real_series.dropna().values.astype(np.float64)
    s_vals = synth_series.dropna().values.astype(np.float64)
    if len(r_vals) == 0 or len(s_vals) == 0:
        return 0.0

    # Normalize both series to [0, 1] using real data range for scale-invariance
    r_min, r_max = r_vals.min(), r_vals.max()
    denom = r_max - r_min
    if denom == 0:
        # Constant column — distance is 0 if synth is also constant at same value
        return 0.0
    r_normed = (r_vals - r_min) / denom
    s_normed = (s_vals - r_min) / denom  # Use real range to normalize synthetic
    return float(ss.wasserstein_distance(r_normed, s_normed))


def compute_js_divergence(real_series: pd.Series, synth_series: pd.Series) -> float:
    """Compute Jensen-Shannon Divergence (JSD) for a categorical column.
    
    Returns the squared JS distance (JS divergence), which lies in [0, 1].
    """
    r_counts = real_series.dropna().astype(str).value_counts()
    s_counts = synth_series.dropna().astype(str).value_counts()
    
    # Get union of all unique categories
    union_cats = list(set(r_counts.index) | set(s_counts.index))
    try:
        union_cats.sort()
    except TypeError:
        union_cats.sort(key=str)
    if not union_cats:
        return 0.0
        
    # Construct probability distributions
    p = np.array([r_counts.get(cat, 0) for cat in union_cats], dtype=np.float64)
    q = np.array([s_counts.get(cat, 0) for cat in union_cats], dtype=np.float64)
    
    # Normalize
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        return 1.0 if (p_sum > 0 or q_sum > 0) else 0.0
        
    p = p / p_sum
    q = q / q_sum
    
    # jensenshannon returns JS distance = sqrt(JSD), so we square it
    js_dist = jensenshannon(p, q, base=2.0)
    return float(js_dist ** 2)


def compute_cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Compute Cramer's V for two categorical columns.
    
    Uses bias-corrected Cramér's V (Bergsma & Wicher, 2013) to obtain an
    unbiased estimate that is robust to small sample sizes and high-dimensional
    contingency tables. This correction aligns with the implementation in
    eda_framework/utils/statistics.py for consistency between EDA and
    evaluation reports.
    """
    conf_matrix = pd.crosstab(x, y)
    if conf_matrix.empty:
        return 0.0
    try:
        chi2 = ss.chi2_contingency(conf_matrix, correction=False)[0]
    except Exception:
        return 0.0
        
    n = conf_matrix.sum().sum()
    r, k = conf_matrix.shape
    if n <= 1 or min(r - 1, k - 1) == 0:
        return 0.0
    
    phi2 = chi2 / n
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    divisor = min((kcorr - 1), (rcorr - 1))
    if divisor <= 0:
        return 0.0
        
    return float(np.sqrt(phi2corr / divisor))


def compute_correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    """Compute Correlation Ratio (η) between categorical and continuous columns.

    Returns η = sqrt(SS_between / SS_total), which lies in [0, 1].
    This is the square root of η² (eta-squared, the proportion of variance
    explained). Returning η (not η²) is intentional so that it is on the
    same scale as |Pearson r| and Cramér's V in the mixed correlation matrix.
    """
    cat_series = pd.Series(categories).dropna()
    meas_series = pd.Series(measurements).dropna()
    
    # Align indices
    common_idx = cat_series.index.intersection(meas_series.index)
    if len(common_idx) == 0:
        return 0.0
        
    cat_series = cat_series.loc[common_idx]
    meas_series = meas_series.loc[common_idx]
    
    ss_total = ((meas_series - meas_series.mean()) ** 2).sum()
    if ss_total == 0:
        return 0.0
        
    ss_between = 0.0
    overall_mean = meas_series.mean()
    groups = meas_series.groupby(cat_series)
    for _, group in groups:
        n_i = len(group)
        mean_i = group.mean()
        ss_between += n_i * ((mean_i - overall_mean) ** 2)
        
    return float(np.sqrt(ss_between / ss_total))


class FidelityAssessor:
    """Assesses statistical fidelity between real and synthetic dataframes."""
    
    def __init__(
        self,
        continuous_cols: List[str],
        categorical_cols: List[str],
    ) -> None:
        self.continuous_cols = continuous_cols
        self.categorical_cols = categorical_cols
        
    def evaluate(self, real_df: pd.DataFrame, synth_df: pd.DataFrame) -> Dict[str, Any]:
        """
        Evaluate statistical distance and correlation difference.
        
        Returns a dict containing:
            'wasserstein': {col_name: distance}
            'js_divergence': {col_name: divergence}
            'correlation_difference': float (average absolute difference)
            'real_corr': correlation matrix dataframe
            'synth_corr': correlation matrix dataframe
        """
        results: Dict[str, Any] = {
            "wasserstein": {},
            "js_divergence": {},
            "correlation_difference": 0.0,
        }
        
        # 1. Wasserstein Distance for continuous columns
        for col in self.continuous_cols:
            if col in real_df.columns and col in synth_df.columns:
                results["wasserstein"][col] = compute_wasserstein_distance(
                    real_df[col], synth_df[col]
                )
                
        # 2. JS Divergence for categorical columns
        for col in self.categorical_cols:
            if col in real_df.columns and col in synth_df.columns:
                results["js_divergence"][col] = compute_js_divergence(
                    real_df[col], synth_df[col]
                )
                
        # 3. Correlation Matrices
        all_cols = self.continuous_cols + self.categorical_cols
        real_corr = self._compute_mixed_correlation_matrix(real_df, all_cols)
        synth_corr = self._compute_mixed_correlation_matrix(synth_df, all_cols)
        
        # Calculate average absolute difference in correlations
        diff_matrix = (real_corr - synth_corr).abs()
        
        # Mask out diagonal since it's always 1.0 (difference 0.0)
        diff_values = diff_matrix.values.copy()
        np.fill_diagonal(diff_values, np.nan)
        avg_diff = float(np.nanmean(diff_values)) if not diff_matrix.empty else 0.0
        
        results["correlation_difference"] = avg_diff
        results["real_corr"] = real_corr
        results["synth_corr"] = synth_corr
        
        return results

    def _compute_mixed_correlation_matrix(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Helper to build a mixed correlation matrix for the specified columns."""
        n = len(columns)
        corr_matrix = pd.DataFrame(np.eye(n), index=columns, columns=columns)
        
        # Process each pair (upper triangle only)
        for i in range(n):
            for j in range(i + 1, n):
                col_i = columns[i]
                col_j = columns[j]
                
                if col_i not in df.columns or col_j not in df.columns:
                    continue
                    
                val = 0.0
                is_i_cont = col_i in self.continuous_cols
                is_j_cont = col_j in self.continuous_cols
                
                try:
                    if is_i_cont and is_j_cont:
                        # Continuous - Continuous: Pearson correlation
                        series_i = df[col_i].dropna()
                        series_j = df[col_j].dropna()
                        common_idx = series_i.index.intersection(series_j.index)
                        if len(common_idx) > 1:
                            val = float(ss.pearsonr(series_i.loc[common_idx], series_j.loc[common_idx])[0])
                    elif not is_i_cont and not is_j_cont:
                        # Categorical - Categorical: Cramer's V
                        val = compute_cramers_v(df[col_i], df[col_j])
                    else:
                        # Continuous - Categorical: Correlation Ratio
                        cat_col = col_i if not is_i_cont else col_j
                        cont_col = col_j if is_j_cont else col_i
                        val = compute_correlation_ratio(df[cat_col], df[cont_col])
                except Exception as exc:
                    logger.debug("Failed to compute correlation for %s & %s: %s", col_i, col_j, exc)
                    val = 0.0
                    
                if np.isnan(val):
                    val = 0.0
                    
                corr_matrix.loc[col_i, col_j] = val
                corr_matrix.loc[col_j, col_i] = val
                
        return corr_matrix
