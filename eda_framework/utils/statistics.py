"""
Pure Statistical Utilities for EDA Framework
---------------------------------------------
Provides standardized mathematical and statistical calculations for tabular data.
This module is stateless, pure-python/pandas/numpy, and has zero plotting dependencies.
"""

import logging
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from typing import List, Dict, Optional, Tuple


logger = logging.getLogger(__name__)

def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """
    Computes Cramér's V statistic for two categorical columns.
    Cramér's V ranges from 0 (no association) to 1 (perfect association).
    """
    mask = x.notnull() & y.notnull()
    if not mask.any():
        return 0.0
    
    cx = x[mask]
    cy = y[mask]
    
    if cx.nunique() <= 1 or cy.nunique() <= 1:
        return 0.0
        
    confusion_matrix = pd.crosstab(cx, cy)
    try:
        chi2 = chi2_contingency(confusion_matrix)[0]
    except Exception:
        return 0.0
        
    n = confusion_matrix.sum().sum()
    if n <= 1:
        return 0.0
        
    phi2 = chi2 / n
    r, k = confusion_matrix.shape
    
    # Bias correction
    phi2corr = max(0.0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    
    divisor = min((kcorr - 1), (rcorr - 1))
    if divisor <= 0:
        return 0.0
        
    return float(np.sqrt(phi2corr / divisor))

def correlation_ratio(categories: pd.Series, measurements: pd.Series) -> float:
    """
    Computes the correlation ratio (eta) between a categorical and a continuous variable.
    Represents the strength of the association. Ranges from 0 to 1.
    """
    categories = pd.Series(categories)
    measurements = pd.Series(measurements)
    
    mask = categories.notnull() & measurements.notnull()
    if not mask.any():
        return 0.0
        
    cat = categories[mask]
    meas = measurements[mask]
    
    if cat.nunique() <= 1 or meas.nunique() <= 1:
        return 0.0
        
    # Total sum of squares
    mean_total = meas.mean()
    ss_total = np.sum((meas - mean_total) ** 2)
    if ss_total == 0:
        return 0.0
        
    # Between-group sum of squares
    ss_between = 0.0
    for group in cat.unique():
        group_meas = meas[cat == group]
        ss_between += len(group_meas) * (group_meas.mean() - mean_total) ** 2
        
    return float(np.sqrt(ss_between / ss_total))

def calculate_nullity_correlation(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes correlation of missingness (correlation matrix of missing status indicator variables).
    """
    missing_cols = df.columns[df.isnull().any()].tolist()
    if len(missing_cols) < 2:
        return pd.DataFrame()
        
    null_df = df[missing_cols].isnull().astype(int)
    return null_df.corr()

def detect_outliers_iqr(df: pd.DataFrame, col: str) -> Tuple[float, float, int, float, pd.Series]:
    """
    Identifies outliers using the IQR (Interquartile Range) method.
    Returns: lower_bound, upper_bound, number of outliers, percentage of outliers, and mask of outliers.
    """
    series = df[col].dropna()
    if len(series) == 0:
        return 0.0, 0.0, 0, 0.0, pd.Series(False, index=df.index)
        
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    num_outliers = outlier_mask.sum()
    pct_outliers = (num_outliers / len(df)) * 100
    
    return float(lower_bound), float(upper_bound), int(num_outliers), float(pct_outliers), outlier_mask

def detect_outliers_zscore(df: pd.DataFrame, col: str, threshold: float = 3.0) -> Tuple[int, float, pd.Series]:
    """
    Identifies outliers using the Z-score method.
    Returns: number of outliers, percentage of outliers, and mask of outliers.
    """
    series = df[col].dropna()
    if len(series) == 0:
        return 0, 0.0, pd.Series(False, index=df.index)
        
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return 0, 0.0, pd.Series(False, index=df.index)
        
    z_scores = (df[col] - mean) / std
    outlier_mask = z_scores.abs() > threshold
    num_outliers = outlier_mask.sum()
    pct_outliers = (num_outliers / len(df)) * 100
    
    return int(num_outliers), float(pct_outliers), outlier_mask

def detect_rare_categories(df: pd.DataFrame, col: str, threshold: float = 0.05) -> pd.DataFrame:
    """
    Checks if a categorical column has rare categories (frequency below threshold).
    Returns a DataFrame containing frequencies and percentages for rare categories.
    """
    counts = df[col].value_counts(dropna=False)
    pcts = df[col].value_counts(normalize=True, dropna=False) * 100
    
    summary = pd.DataFrame({'Count': counts, 'Percentage': pcts})
    rare = summary[summary['Percentage'] < (threshold * 100)]
    return rare

def association_matrix(df: pd.DataFrame, continuous_cols: Optional[List[str]] = None, categorical_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Generates a unified association matrix for mixed-type datasets.
    Computes:
    - Pearson correlation for continuous-continuous pairs.
    - Cramér's V for categorical-categorical pairs.
    - Correlation Ratio for continuous-categorical pairs.
    """
    if continuous_cols is None:
        continuous_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
    all_cols = list(continuous_cols) + list(categorical_cols)
    all_cols = [c for c in all_cols if c in df.columns]
    
    n = len(all_cols)
    matrix = pd.DataFrame(np.zeros((n, n)), index=all_cols, columns=all_cols)
    
    for i in range(n):
        for j in range(i, n):
            col1 = all_cols[i]
            col2 = all_cols[j]
            
            if col1 == col2:
                val = 1.0
            elif col1 in continuous_cols and col2 in continuous_cols:
                val = df[col1].corr(df[col2], method='pearson')
                if np.isnan(val):
                    val = 0.0
            elif col1 in categorical_cols and col2 in categorical_cols:
                val = cramers_v(df[col1], df[col2])
            else:
                cat_col = col1 if col1 in categorical_cols else col2
                cont_col = col1 if col1 in continuous_cols else col2
                val = correlation_ratio(df[cat_col], df[cont_col])
                if np.isnan(val):
                    val = 0.0
            
            matrix.loc[col1, col2] = val
            matrix.loc[col2, col1] = val
            
    return matrix
