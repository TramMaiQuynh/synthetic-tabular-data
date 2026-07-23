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
import os
import re
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import scipy.stats as ss
from scipy.spatial.distance import jensenshannon
import yaml

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
        # Return NaN to mark the computation as invalid. Returning 0.0 would
        # incorrectly indicate perfect alignment between two distributions
        # when one or both are entirely empty/NaN. NaN propagates correctly
        # through downstream aggregation (e.g., np.nanmean) and ensures
        # failed computations are not silently treated as perfect matches.
        return float("nan")

    # Normalize both series to [0, 1] using real data range for scale-invariance
    r_min, r_max = r_vals.min(), r_vals.max()
    denom = r_max - r_min
    if denom == 0:
        # Constant column: distance is the absolute difference between the
        # constant values (normalized to [0,1] by treating the constant as
        # the only value). If both are the same constant, distance is 0.
        s_const = s_vals[0] if len(s_vals) > 0 else 0.0
        r_const = r_vals[0] if len(r_vals) > 0 else 0.0
        return float(abs(s_const - r_const))
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
        # Return NaN when no categories exist in either distribution.
        # Returning 0.0 would incorrectly indicate perfect alignment between
        # two non-existent distributions. NaN ensures downstream aggregation
        # (e.g., np.nanmean) properly excludes this invalid computation.
        return float("nan")
        
    # Construct probability distributions
    p = np.array([r_counts.get(cat, 0) for cat in union_cats], dtype=np.float64)
    q = np.array([s_counts.get(cat, 0) for cat in union_cats], dtype=np.float64)
    
    # Normalize
    p_sum = p.sum()
    q_sum = q.sum()
    if p_sum == 0 or q_sum == 0:
        # Return NaN when both distributions are empty (p_sum == 0 and q_sum == 0).
        # Returning 0.0 would incorrectly indicate perfect alignment between
        # two non-existent distributions. NaN ensures downstream aggregation
        # (e.g., np.nanmean) properly excludes this invalid computation.
        # When only one side is empty (but the other has mass), return 1.0
        # (maximally divergent) which is mathematically correct.
        return 1.0 if (p_sum > 0 or q_sum > 0) else float("nan")
        
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

    def evaluate_constraints(
        self,
        real_df: pd.DataFrame,
        synth_df: pd.DataFrame,
        constraint_formulas_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate business logic constraint fidelity.

        Loads constraint formulas from a YAML file (e.g.,
        config/<dataset>/constraint_formulas.yaml) and checks whether the
        synthetic data satisfies each constraint.

        For each constraint of the form "result_col = expr", the method:
          1. Computes the expected value (expr) from the synthetic data.
          2. Computes the Mean Absolute Percentage Error (MAPE) between
             expected and actual values.
          3. Reports the fraction of rows where the constraint is violated
             beyond a configurable tolerance.

        This is a critical fidelity check: generative models that only match
        marginal distributions but fail to learn cross-column business logic
        (e.g., TotalCharges = tenure * MonthlyCharges) are not truly faithful
        to the data-generating process.

        Args:
            real_df: Real DataFrame (used for reference, e.g., to compute
                     the real-data MAPE for comparison).
            synth_df: Synthetic DataFrame to evaluate.
            constraint_formulas_path: Path to YAML file with constraint
                                      definitions. If None, the method
                                      returns an empty result.

        Returns:
            dict with keys:
              'constraint_results': list of per-constraint dicts, each with:
                  - 'expression': the constraint expression string.
                  - 'description': human-readable description.
                  - 'mape': Mean Absolute Percentage Error on synthetic data.
                  - 'real_mape': MAPE on real data (for comparison).
                  - 'violation_rate': fraction of rows with |error| > tolerance.
                  - 'tolerance': the tolerance threshold used.
              'avg_constraint_mape': average MAPE across all constraints.
        """
        if constraint_formulas_path is None or not os.path.exists(constraint_formulas_path):
            return {
                "constraint_results": [],
                "avg_constraint_mape": float("nan"),
            }

        with open(constraint_formulas_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

        constraints = config.get("constraints", [])
        if not constraints:
            return {
                "constraint_results": [],
                "avg_constraint_mape": float("nan"),
            }

        results = []
        for c in constraints:
            expression = c.get("expression", "")
            metric = c.get("metric", "mape")
            tolerance = c.get("tolerance", 0.15)
            description = c.get("description", "")

            # Parse "result_col = expr" pattern
            # e.g., "TotalCharges = tenure * MonthlyCharges"
            match = re.match(r"^\s*(\w+)\s*=\s*(.+)\s*$", expression)
            if not match:
                logger.warning("Cannot parse constraint expression: %s", expression)
                continue

            result_col = match.group(1)
            expr_str = match.group(2).strip()

            if result_col not in synth_df.columns:
                logger.warning(
                    "Result column '%s' not found in synthetic data. "
                    "Skipping constraint: %s",
                    result_col, expression,
                )
                continue

            # Evaluate the expression on synthetic data
            try:
                expected = synth_df.eval(expr_str)
            except Exception as exc:
                logger.warning(
                    "Failed to evaluate expression '%s' on synthetic data: %s",
                    expr_str, exc,
                )
                continue

            actual = pd.to_numeric(synth_df[result_col], errors="coerce")

            # Compute per-row absolute percentage error
            # Avoid division by zero: skip rows where expected == 0
            valid_mask = (expected != 0) & expected.notna() & actual.notna()
            n_valid = valid_mask.sum()
            n_total = len(synth_df)

            if n_valid == 0:
                mape = float("nan")
                violation_rate = float("nan")
            else:
                abs_pct_error = (
                    (actual[valid_mask] - expected[valid_mask]).abs()
                    / expected[valid_mask].abs()
                )
                mape = float(abs_pct_error.mean())
                violation_rate = float((abs_pct_error > tolerance).mean())

            # Compute real-data MAPE for comparison (same expression)
            real_mape = float("nan")
            if result_col in real_df.columns:
                try:
                    real_expected = real_df.eval(expr_str)
                    real_actual = pd.to_numeric(
                        real_df[result_col], errors="coerce"
                    )
                    real_valid = (
                        (real_expected != 0)
                        & real_expected.notna()
                        & real_actual.notna()
                    )
                    if real_valid.sum() > 0:
                        real_ape = (
                            (real_actual[real_valid] - real_expected[real_valid]).abs()
                            / real_expected[real_valid].abs()
                        )
                        real_mape = float(real_ape.mean())
                except Exception:
                    pass

            results.append({
                "expression": expression,
                "description": description,
                "mape": mape,
                "real_mape": real_mape,
                "violation_rate": violation_rate,
                "tolerance": tolerance,
                "n_valid": int(n_valid),
                "n_total": int(n_total),
            })

        avg_mape = float(
            np.mean([r["mape"] for r in results if not np.isnan(r["mape"])])
        ) if results else float("nan")

        return {
            "constraint_results": results,
            "avg_constraint_mape": avg_mape,
        }

    def _compute_mixed_correlation_matrix(self, df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
        """Helper to build a mixed correlation matrix for the specified columns."""
        n = len(columns)
        # Initialize with NaN (not 0.0) so that uncomputed pairs are explicitly
        # marked as invalid. Using np.eye(n) would set off-diagonal elements
        # to 0.0, which would inject false "zero correlation" signals when
        # columns are missing or computation fails (continue is hit). NaN
        # ensures these invalid entries are excluded by np.nanmean in the
        # correlation_difference aggregation. The diagonal is set to 1.0
        # (self-correlation) after initialization.
        corr_matrix = pd.DataFrame(
            np.full((n, n), np.nan, dtype=np.float64), index=columns, columns=columns
        )
        # Set diagonal to 1.0 (self-correlation). Use .copy() to ensure
        # the underlying array is writable, as DataFrame.values may be
        # read-only in certain pandas/numpy versions.
        corr_values = corr_matrix.values.copy()
        np.fill_diagonal(corr_values, 1.0)
        corr_matrix = pd.DataFrame(corr_values, index=columns, columns=columns)
        
        # Process each pair (upper triangle only)
        for i in range(n):
            for j in range(i + 1, n):
                col_i = columns[i]
                col_j = columns[j]
                
                if col_i not in df.columns or col_j not in df.columns:
                    # Leave as NaN — column missing from data, cannot compute.
                    continue
                    
                val = float("nan")
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
                    # Keep val as NaN to mark computation failure. Setting to
                    # 0.0 would inject a false "no correlation" signal that
                    # skews the correlation_difference metric. NaN is properly
                    # excluded by np.nanmean in the aggregation step.
                    val = float("nan")
                    
                # Preserve NaN values: do NOT coerce to 0.0. NaN indicates
                # a failed/invalid computation that should be excluded from
                # downstream aggregation (np.nanmean), not treated as a
                # valid "zero correlation" result.
                    
                corr_matrix.loc[col_i, col_j] = val
                corr_matrix.loc[col_j, col_i] = val
                
        return corr_matrix
