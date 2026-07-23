import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, pearsonr, spearmanr, kruskal, ks_2samp
from scipy.spatial.distance import jensenshannon
from typing import List, Optional, Tuple


def wasserstein_distance(real_series: pd.Series, synth_series: pd.Series) -> float:
    """
    Wasserstein-1 distance between two continuous distributions.
    Normalizes both to [0,1] using real data range for scale-invariance.
    """
    r = real_series.dropna().values.astype(np.float64)
    s = synth_series.dropna().values.astype(np.float64)
    if len(r) == 0 or len(s) == 0:
        return 0.0
    r_min, r_max = r.min(), r.max()
    denom = r_max - r_min
    if denom == 0:
        return float(abs(s[0] - r[0])) if len(s) > 0 and len(r) > 0 else 0.0
    r_norm = (r - r_min) / denom
    s_norm = (s - r_min) / denom
    from scipy.stats import wasserstein_distance as wd
    return float(wd(r_norm, s_norm))


def js_divergence(real_series: pd.Series, synth_series: pd.Series, base: float = 2.0) -> float:
    """
    Jensen-Shannon Divergence (squared) between two categorical distributions.
    Returns value in [0, 1] when base=2.
    Aligns categories via union before computing.
    """
    r_counts = real_series.dropna().astype(str).value_counts()
    s_counts = synth_series.dropna().astype(str).value_counts()
    union = sorted(set(r_counts.index) | set(s_counts.index), key=str)
    if not union:
        return 0.0
    p = np.array([r_counts.get(c, 0) for c in union], dtype=np.float64)
    q = np.array([s_counts.get(c, 0) for c in union], dtype=np.float64)
    p_sum, q_sum = p.sum(), q.sum()
    if p_sum == 0 or q_sum == 0:
        return 1.0 if (p_sum > 0 or q_sum > 0) else 0.0
    p, q = p / p_sum, q / q_sum
    return float(jensenshannon(p, q, base=base) ** 2)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Bias-corrected Cramér's V (Bergsma & Wicher, 2013).
    Measures association between two categorical variables.
    Returns value in [0, 1].
    """
    mask = x.notna() & y.notna()
    if not mask.any():
        return 0.0
    cx, cy = x[mask], y[mask]
    if cx.nunique() <= 1 or cy.nunique() <= 1:
        return 0.0
    ct = pd.crosstab(cx, cy)
    try:
        chi2 = chi2_contingency(ct)[0]
    except Exception:
        return 0.0
    n = ct.sum().sum()
    if n <= 1:
        return 0.0
    phi2 = chi2 / n
    r, k = ct.shape
    phi2_corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    r_corr = r - ((r - 1) ** 2) / (n - 1)
    k_corr = k - ((k - 1) ** 2) / (n - 1)
    divisor = min((k_corr - 1), (r_corr - 1))
    if divisor <= 0:
        return 0.0
    return float(np.sqrt(phi2_corr / divisor))


def correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    """
    Correlation ratio η between categorical and continuous variables.
    η = sqrt(SS_between / SS_total). Returns value in [0, 1].
    """
    mask = categories.notna() & measurements.notna()
    if not mask.any():
        return 0.0
    cat, meas = categories[mask], measurements[mask]
    if cat.nunique() <= 1 or meas.nunique() <= 1:
        return 0.0
    ss_total = ((meas - meas.mean()) ** 2).sum()
    if ss_total == 0:
        return 0.0
    ss_between = sum(
        len(g) * (g.mean() - meas.mean()) ** 2
        for _, g in meas.groupby(cat)
    )
    return float(np.sqrt(ss_between / ss_total))


def theils_u(x: pd.Series, y: pd.Series) -> float:
    """
    Theil's U (Uncertainty Coefficient): symmetric measure of association
    between two categorical variables. Based on mutual information.
    Returns value in [0, 1].
    """
    mask = x.notna() & y.notna()
    if not mask.any():
        return 0.0
    cx, cy = x[mask], y[mask]
    ct = pd.crosstab(cx, cy)
    n = ct.sum().sum()
    if n == 0:
        return 0.0
    p_joint = ct.values / n
    p_x = p_joint.sum(axis=1)
    p_y = p_joint.sum(axis=0)
    p_joint = np.maximum(p_joint, 1e-12)
    p_x = np.maximum(p_x, 1e-12)
    p_y = np.maximum(p_y, 1e-12)
    mi = np.sum(p_joint * np.log(p_joint / (p_x[:, None] * p_y[None, :])))
    h_x = -np.sum(p_x * np.log(p_x))
    h_y = -np.sum(p_y * np.log(p_y))
    if h_x == 0 or h_y == 0:
        return 0.0
    return float(2.0 * mi / (h_x + h_y))


def shannon_entropy(series: pd.Series, base: float = 2.0) -> float:
    """Shannon entropy of a categorical variable in bits (base=2)."""
    counts = series.dropna().value_counts()
    if len(counts) <= 1:
        return 0.0
    p = counts / counts.sum()
    return float(-np.sum(p * np.log(p) / np.log(base)))


def describe_continuous(series: pd.Series) -> dict:
    """
    Full descriptive statistics for a continuous variable.
    Returns dict with: count, mean, std, min, Q25, median, Q75, max,
                       skewness, kurtosis, iqr, range, zero_count, zero_ratio.
    """
    s = series.dropna()
    if len(s) == 0:
        return {k: None for k in ["count", "mean", "std", "min", "q25", "median", "q75", "max",
                                   "skewness", "kurtosis", "iqr", "range", "zero_count", "zero_ratio"]}
    q25, q75 = s.quantile(0.25), s.quantile(0.75)
    return {
        "count": int(len(s)),
        "mean": float(s.mean()),
        "std": float(s.std()),
        "min": float(s.min()),
        "q25": float(q25),
        "median": float(s.median()),
        "q75": float(q75),
        "max": float(s.max()),
        "skewness": float(s.skew()),
        "kurtosis": float(s.kurtosis()),
        "iqr": float(q75 - q25),
        "range": float(s.max() - s.min()),
        "zero_count": int((s == 0).sum()),
        "zero_ratio": float((s == 0).mean()),
    }


def describe_categorical(series: pd.Series) -> dict:
    """
    Full descriptive statistics for a categorical variable.
    Returns dict with: count, cardinality, mode, mode_freq, mode_pct,
                       entropy, unique_values (list of top 20).
    """
    s = series.dropna()
    if len(s) == 0:
        return {k: None for k in ["count", "cardinality", "mode", "mode_freq",
                                   "mode_pct", "entropy", "unique_values"]}
    vc = s.value_counts()
    return {
        "count": int(len(s)),
        "cardinality": int(s.nunique()),
        "mode": str(vc.index[0]),
        "mode_freq": int(vc.iloc[0]),
        "mode_pct": float(vc.iloc[0] / len(s)),
        "entropy": shannon_entropy(s),
        "unique_values": [str(v) for v in vc.index[:20]],
    }


def normality_test(series: pd.Series, method: str = "auto") -> dict:
    """
    Test normality of a continuous variable.
    method='auto': Shapiro-Wilk if n < 5000, D'Agostino-Pearson otherwise.
    Returns dict with: statistic, p_value, test_name, is_normal (p < 0.05).
    """
    s = series.dropna()
    n = len(s)
    if n < 3:
        return {"statistic": None, "p_value": None, "test_name": "insufficient_data", "is_normal": None}

    if method == "auto":
        method = "shapiro" if n < 5000 else "normaltest"

    if method == "shapiro":
        from scipy.stats import shapiro
        stat, p = shapiro(s)
        return {"statistic": float(stat), "p_value": float(p), "test_name": "Shapiro-Wilk", "is_normal": p >= 0.05}
    else:
        from scipy.stats import normaltest
        stat, p = normaltest(s)
        return {"statistic": float(stat), "p_value": float(p), "test_name": "D'Agostino-Pearson", "is_normal": p >= 0.05}


def detect_outliers_iqr(series: pd.Series, multiplier: float = 1.5) -> dict:
    """
    Detect outliers using IQR method.
    Returns dict with: lower_bound, upper_bound, n_outliers, pct_outliers, outlier_mask.
    multiplier=1.5 là convention (Tukey), không phải threshold cứng.
    """
    s = series.dropna()
    if len(s) == 0:
        return {"lower_bound": None, "upper_bound": None, "n_outliers": 0, "pct_outliers": 0.0, "outlier_mask": pd.Series(False, index=series.index)}
    q1, q3 = s.quantile(0.25), s.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - multiplier * iqr
    upper = q3 + multiplier * iqr
    mask = (series < lower) | (series > upper)
    return {
        "lower_bound": float(lower),
        "upper_bound": float(upper),
        "n_outliers": int(mask.sum()),
        "pct_outliers": float(mask.mean() * 100),
        "outlier_mask": mask,
    }


def detect_outliers_zscore(series: pd.Series, threshold: float = 3.0) -> dict:
    """
    Detect outliers using Z-score method.
    threshold=3.0 là convention (3-sigma), không phải threshold cứng.
    """
    s = series.dropna()
    if len(s) == 0 or s.std() == 0:
        return {"n_outliers": 0, "pct_outliers": 0.0, "outlier_mask": pd.Series(False, index=series.index)}
    z = (series - s.mean()) / s.std()
    mask = z.abs() > threshold
    return {
        "n_outliers": int(mask.sum()),
        "pct_outliers": float(mask.mean() * 100),
        "outlier_mask": mask,
    }


def detect_outliers_mad(series: pd.Series, threshold: float = 3.0) -> dict:
    """
    Detect outliers using Median Absolute Deviation (MAD) method.
    More robust than Z-score for skewed distributions.
    threshold=3.0 là convention.
    """
    s = series.dropna()
    if len(s) == 0:
        return {"n_outliers": 0, "pct_outliers": 0.0, "outlier_mask": pd.Series(False, index=series.index)}
    median = s.median()
    mad = np.median(np.abs(s - median))
    if mad == 0:
        return {"n_outliers": 0, "pct_outliers": 0.0, "outlier_mask": pd.Series(False, index=series.index)}
    modified_z = 0.6745 * (series - median) / mad
    mask = modified_z.abs() > threshold
    return {
        "n_outliers": int(mask.sum()),
        "pct_outliers": float(mask.mean() * 100),
        "outlier_mask": mask,
    }


def multimodality_kde_peaks(series: pd.Series) -> dict:
    """
    Detect number of modes (peaks) in a continuous distribution using KDE.
    Returns dict with: n_peaks, peak_locations, bandwidth.
    """
    from scipy.signal import find_peaks
    from sklearn.neighbors import KernelDensity

    s = series.dropna()
    if len(s) < 10 or s.nunique() <= 1:
        return {"n_peaks": 1, "peak_locations": [], "bandwidth": None}

    # Scott's rule for bandwidth
    bw = 1.06 * s.std() * len(s) ** (-0.2)
    if bw == 0:
        return {"n_peaks": 1, "peak_locations": [], "bandwidth": 0.0}

    x_grid = np.linspace(s.min(), s.max(), 500)
    kde = KernelDensity(bandwidth=bw, kernel="gaussian")
    kde.fit(s.values.reshape(-1, 1))
    log_dens = kde.score_samples(x_grid.reshape(-1, 1))
    density = np.exp(log_dens)

    peaks, properties = find_peaks(density, prominence=0.01 * density.max())
    return {
        "n_peaks": int(len(peaks)),
        "peak_locations": [float(x_grid[p]) for p in peaks],
        "bandwidth": float(bw),
    }


def missing_analysis(df: pd.DataFrame) -> dict:
    """
    Analyze missing values in a DataFrame.
    Returns dict with: per_column (missing_count, missing_pct),
                       total_missing, total_cells, overall_pct,
                       nullity_correlation_matrix.
    """
    n_rows = len(df)
    n_cells = n_rows * len(df.columns)
    null_counts = df.isnull().sum()
    null_pcts = (null_counts / n_rows * 100).round(4)

    per_column = {}
    for col in df.columns:
        per_column[col] = {
            "missing_count": int(null_counts[col]),
            "missing_pct": float(null_pcts[col]),
        }

    # Nullity correlation (pairwise missing pattern correlation)
    missing_cols = [c for c in df.columns if null_counts[c] > 0]
    nullity_corr = {}
    if len(missing_cols) >= 2:
        null_df = df[missing_cols].isnull().astype(int)
        corr_mat = null_df.corr()
        nullity_corr = corr_mat.to_dict()

    return {
        "per_column": per_column,
        "total_missing": int(null_counts.sum()),
        "total_cells": int(n_cells),
        "overall_missing_pct": float(null_counts.sum() / n_cells * 100),
        "nullity_correlation": nullity_corr,
    }


def duplicate_analysis(df: pd.DataFrame, subset: Optional[List[str]] = None) -> dict:
    """
    Analyze duplicate rows.
    Returns dict with: n_exact_duplicates, pct_exact_duplicates,
                       n_duplicates_by_subset (if subset given).
    """
    n_exact = int(df.duplicated().sum())
    result = {
        "n_exact_duplicates": n_exact,
        "pct_exact_duplicates": float(n_exact / len(df) * 100),
    }
    if subset:
        n_subset = int(df.duplicated(subset=subset).sum())
        result["n_duplicates_by_subset"] = n_subset
        result["pct_duplicates_by_subset"] = float(n_subset / len(df) * 100)
    return result


def cardinality_analysis(series: pd.Series) -> dict:
    """
    Analyze cardinality of a categorical variable.
    Returns dict with: cardinality, n_rare (frequency < 1%),
                       rare_categories (list), frequency_table.
    """
    s = series.dropna()
    if len(s) == 0:
        return {"cardinality": 0, "n_rare": 0, "rare_categories": [], "frequency_table": {}}
    vc = s.value_counts(normalize=True)
    rare = vc[vc < 0.01]
    return {
        "cardinality": int(s.nunique()),
        "n_rare": int(len(rare)),
        "rare_categories": [str(v) for v in rare.index[:20]],
        "frequency_table": {str(k): float(v) for k, v in vc.items()},
    }


def vif_analysis(df: pd.DataFrame, continuous_cols: List[str]) -> dict:
    """
    Variance Inflation Factor for continuous features.
    Uses LinearRegression R². Returns dict of {col: vif}.
    VIF = 1 / (1 - R²_i) where R²_i is from regressing col on all others.
    """
    from sklearn.linear_model import LinearRegression

    clean = df[continuous_cols].dropna()
    if len(clean) < 10 or len(continuous_cols) < 2:
        return {col: 1.0 for col in continuous_cols}

    vifs = {}
    for col in continuous_cols:
        y = clean[col]
        X = clean.drop(columns=[col])
        lr = LinearRegression()
        try:
            lr.fit(X, y)
            r2 = lr.score(X, y)
            vifs[col] = float(1.0 / (1.0 - r2)) if r2 < 1.0 else float("inf")
        except Exception:
            vifs[col] = 1.0
    return vifs


def mutual_information(x: pd.Series, y: pd.Series, discrete_x: bool = False, discrete_y: bool = False) -> float:
    """
    Mutual Information between two variables in bits.
    Handles mixed types (continuous/categorical).
    """
    from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
    from sklearn.preprocessing import LabelEncoder

    mask = x.notna() & y.notna()
    if not mask.any():
        return 0.0
    xc, yc = x[mask], y[mask]
    if xc.nunique() <= 1 or yc.nunique() <= 1:
        return 0.0

    # Encode x
    if discrete_x:
        x_enc = LabelEncoder().fit_transform(xc.astype(str)).reshape(-1, 1)
    else:
        x_enc = xc.values.reshape(-1, 1).astype(np.float64)

    # Encode y
    if discrete_y:
        y_enc = LabelEncoder().fit_transform(yc.astype(str))
        mi = mutual_info_classif(x_enc, y_enc, discrete_features=[discrete_x], random_state=42)
    else:
        mi = mutual_info_regression(x_enc, yc.values, discrete_features=[discrete_x], random_state=42)

    # Convert nats → bits
    return float(mi[0] / np.log(2.0)) if len(mi) > 0 else 0.0


def partial_correlation(df: pd.DataFrame, col1: str, col2: str, covariates: List[str]) -> float:
    """
    Partial correlation between col1 and col2 controlling for covariates.
    Uses precision matrix approach (inverse of correlation matrix).
    """
    from sklearn.linear_model import LinearRegression

    cols = [col1, col2] + covariates
    clean = df[cols].dropna()
    if len(clean) < 10:
        return 0.0

    # Regress col1 on covariates, get residuals
    lr1 = LinearRegression().fit(clean[covariates], clean[col1])
    res1 = clean[col1] - lr1.predict(clean[covariates])

    # Regress col2 on covariates, get residuals
    lr2 = LinearRegression().fit(clean[covariates], clean[col2])
    res2 = clean[col2] - lr2.predict(clean[covariates])

    return float(pearsonr(res1, res2)[0])


def conditional_mutual_information(df: pd.DataFrame, x: str, y: str, z: List[str]) -> float:
    """
    Conditional Mutual Information I(X; Y | Z).
    Uses k-NN based estimation (continuous) or plug-in (discrete).
    """
    # Simple approximation: I(X;Y|Z) = I(X;Y,Z) - I(X;Z)
    # This is exact by chain rule
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.preprocessing import LabelEncoder

    clean = df[[x, y] + z].dropna()
    if len(clean) < 10:
        return 0.0

    # Determine if variables are discrete
    def is_discrete(col):
        return clean[col].dtype == object or clean[col].nunique() <= 10

    discrete_x = is_discrete(x)
    discrete_y = is_discrete(y)
    discrete_z = [is_discrete(c) for c in z]

    # Encode
    def encode(col, discrete):
        if discrete:
            return LabelEncoder().fit_transform(clean[col].astype(str))
        return clean[col].values.astype(np.float64)

    x_vals = encode(x, discrete_x)
    y_vals = encode(y, discrete_y)
    z_vals = np.column_stack([encode(c, dz) for c, dz in zip(z, discrete_z)])

    # I(X; Y, Z)
    yz = np.column_stack([y_vals] + [z_vals]) if z_vals.ndim > 1 else np.column_stack([y_vals, z_vals])
    if discrete_x:
        mi_xyz = mutual_info_regression(x_vals.reshape(-1, 1), yz, discrete_features=[True], random_state=42)[0]
    else:
        mi_xyz = mutual_info_regression(x_vals.reshape(-1, 1), yz, discrete_features=[False], random_state=42)[0]

    # I(X; Z)
    if discrete_x:
        mi_xz = mutual_info_regression(x_vals.reshape(-1, 1), z_vals, discrete_features=[True], random_state=42)[0]
    else:
        mi_xz = mutual_info_regression(x_vals.reshape(-1, 1), z_vals, discrete_features=[False], random_state=42)[0]

    result = max(0.0, mi_xyz - mi_xz) / np.log(2.0)  # convert to bits
    return float(result)


def covariate_shift_detection(
    df_train: pd.DataFrame, df_test: pd.DataFrame,
    continuous_cols: List[str], categorical_cols: List[str]
) -> list:
    """
    Detect covariate shift between train and test sets.
    Uses KS test for continuous, Chi-square for categorical.
    Returns list of dicts with: feature, method, statistic, p_value.
    """
    results = []
    for col in continuous_cols:
        if col not in df_train.columns or col not in df_test.columns:
            continue
        t = df_train[col].dropna()
        e = df_test[col].dropna()
        if len(t) == 0 or len(e) == 0:
            continue
        stat, p = ks_2samp(t, e)
        results.append({
            "feature": col,
            "type": "continuous",
            "method": "Kolmogorov-Smirnov",
            "statistic": float(stat),
            "p_value": float(p),
        })

    for col in categorical_cols:
        if col not in df_train.columns or col not in df_test.columns:
            continue
        t = df_train[col].value_counts()
        e = df_test[col].value_counts()
        merged = pd.DataFrame({"train": t, "test": e}).fillna(0)
        if merged.empty or merged.sum().sum() == 0:
            continue
        try:
            stat, p, _, _ = chi2_contingency(merged)
        except Exception:
            stat, p = 0.0, 1.0
        results.append({
            "feature": col,
            "type": "categorical",
            "method": "Chi-square homogeneity",
            "statistic": float(stat),
            "p_value": float(p),
        })
    return results


def class_balance(series: pd.Series) -> dict:
    """
    Analyze class balance of a categorical target.
    Returns dict with: n_classes, class_counts, class_pcts,
                       imbalance_ratio (majority/minority), entropy.
    """
    s = series.dropna()
    if len(s) == 0:
        return {}
    vc = s.value_counts()
    return {
        "n_classes": int(s.nunique()),
        "class_counts": {str(k): int(v) for k, v in vc.items()},
        "class_pcts": {str(k): float(v / len(s)) for k, v in vc.items()},
        "imbalance_ratio": float(vc.iloc[0] / vc.iloc[-1]),
        "entropy": shannon_entropy(s),
    }


def multiple_testing_correction(p_values: List[float], method: str = "fdr_bh") -> dict:
    """
    Correct p-values for multiple testing.
    method: 'bonferroni', 'fdr_bh' (Benjamini-Hochberg), 'fdr_by' (Benjamini-Yekutieli).
    Returns dict with: corrected_p_values, method, n_tests, n_significant (alpha=0.05).
    """
    from scipy.stats import false_discovery_control
    import numpy as np

    n = len(p_values)
    if n == 0:
        return {"corrected_p_values": [], "method": method, "n_tests": 0, "n_significant": 0}

    p = np.array(p_values)

    if method == "bonferroni":
        corrected = np.minimum(p * n, 1.0)
    elif method == "fdr_bh":
        sorted_idx = np.argsort(p)
        sorted_p = p[sorted_idx]
        ranks = np.arange(1, n + 1)
        bh = sorted_p * n / ranks
        # Monotonicity constraint
        bh = np.minimum.accumulate(bh[::-1])[::-1]
        corrected = np.empty(n)
        corrected[sorted_idx] = bh
        corrected = np.minimum(corrected, 1.0)
    elif method == "fdr_by":
        sorted_idx = np.argsort(p)
        sorted_p = p[sorted_idx]
        ranks = np.arange(1, n + 1)
        c_n = np.sum(1.0 / np.arange(1, n + 1))
        by = sorted_p * n / ranks * c_n
        by = np.minimum.accumulate(by[::-1])[::-1]
        corrected = np.empty(n)
        corrected[sorted_idx] = by
        corrected = np.minimum(corrected, 1.0)
    else:
        raise ValueError(f"Unknown method: {method}")

    return {
        "corrected_p_values": [float(v) for v in corrected],
        "method": method,
        "n_tests": n,
        "n_significant_005": int((corrected < 0.05).sum()),
        "n_significant_001": int((corrected < 0.01).sum()),
    }