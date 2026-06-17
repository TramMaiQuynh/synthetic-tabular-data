"""
Shared EDA Utilities
--------------------
Enterprise-grade utility functions for Exploratory Data Analysis (EDA).
Provides standardized statistical metrics and plotting helpers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

def setup_plot_style():
    """Sets up a modern, high-quality, professional plotting aesthetic."""
    sns.set_theme(style="whitegrid")
    plt.rcParams.update({
        'figure.figsize': (10, 6),
        'figure.dpi': 120,
        'axes.labelsize': 11,
        'axes.titlesize': 13,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 10,
        'figure.titlesize': 15,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans', 'sans-serif']
    })

def cramers_v(x, y):
    """
    Computes Cramér's V statistic for two categorical columns.
    Cramér's V ranges from 0 (no association) to 1 (perfect association).
    """
    # Drop rows with NaN in either column
    mask = x.notnull() & y.notnull()
    if not mask.any():
        return 0.0
    
    cx = x[mask]
    cy = y[mask]
    
    # Avoid zero variance or single unique values
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
    phi2corr = max(0, phi2 - ((k - 1) * (r - 1)) / (n - 1))
    rcorr = r - ((r - 1) ** 2) / (n - 1)
    kcorr = k - ((k - 1) ** 2) / (n - 1)
    
    divisor = min((kcorr - 1), (rcorr - 1))
    if divisor <= 0:
        return 0.0
        
    return np.sqrt(phi2corr / divisor)

def correlation_ratio(categories, measurements):
    """
    Computes the correlation ratio (eta) between a categorical and a continuous variable.
    Represents the strength of the association. Ranges from 0 to 1.
    """
    # Align and drop missing values
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
        
    return np.sqrt(ss_between / ss_total)

def calculate_nullity_correlation(df):
    """
    Computes correlation of missingness (correlation matrix of missing status indicator variables).
    Helpful to identify MAR / MNAR patterns.
    """
    missing_cols = df.columns[df.isnull().any()].tolist()
    if len(missing_cols) < 2:
        return pd.DataFrame()
        
    null_df = df[missing_cols].isnull().astype(int)
    return null_df.corr()

def detect_outliers_iqr(df, col):
    """
    Identifies outliers using the IQR (Interquartile Range) method.
    Returns: lower_bound, upper_bound, number of outliers, percentage of outliers, and mask of outliers.
    """
    series = df[col].dropna()
    if len(series) == 0:
        return 0, 0, 0, 0.0, pd.Series(dtype=bool)
        
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    
    outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
    num_outliers = outlier_mask.sum()
    pct_outliers = (num_outliers / len(df)) * 100
    
    return lower_bound, upper_bound, num_outliers, pct_outliers, outlier_mask

def detect_outliers_zscore(df, col, threshold=3.0):
    """
    Identifies outliers using the Z-score method.
    Returns: number of outliers, percentage of outliers, and mask of outliers.
    """
    series = df[col].dropna()
    if len(series) == 0:
        return 0, 0.0, pd.Series(dtype=bool)
        
    mean = series.mean()
    std = series.std()
    
    if std == 0:
        return 0, 0.0, pd.Series(dtype=bool)
        
    z_scores = (df[col] - mean) / std
    outlier_mask = z_scores.abs() > threshold
    num_outliers = outlier_mask.sum()
    pct_outliers = (num_outliers / len(df)) * 100
    
    return num_outliers, pct_outliers, outlier_mask

def detect_rare_categories(df, col, threshold=0.05):
    """
    Checks if a categorical column has rare categories (frequency below threshold).
    Returns a DataFrame containing frequencies and percentages for rare categories.
    """
    counts = df[col].value_counts(dropna=False)
    pcts = df[col].value_counts(normalize=True, dropna=False) * 100
    
    summary = pd.DataFrame({'Count': counts, 'Percentage': pcts})
    rare = summary[summary['Percentage'] < (threshold * 100)]
    return rare

def association_matrix(df, continuous_cols=None, categorical_cols=None):
    """
    Generates a unified association matrix for mixed-type datasets.
    Computes:
    - Pearson correlation for continuous-continuous pairs.
    - Cramér's V for categorical-categorical pairs.
    - Correlation Ratio for continuous-categorical pairs.
    """
    # Auto-detect column types if not provided
    if continuous_cols is None:
        continuous_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()
        
    all_cols = list(continuous_cols) + list(categorical_cols)
    # Filter columns that are actually in df
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
                # Pearson correlation
                val = df[col1].corr(df[col2], method='pearson')
                if np.isnan(val):
                    val = 0.0
            elif col1 in categorical_cols and col2 in categorical_cols:
                # Cramér's V
                val = cramers_v(df[col1], df[col2])
            else:
                # One continuous, one categorical
                cat_col = col1 if col1 in categorical_cols else col2
                cont_col = col1 if col1 in continuous_cols else col2
                val = correlation_ratio(df[cat_col], df[cont_col])
                if np.isnan(val):
                    val = 0.0
            
            matrix.loc[col1, col2] = val
            matrix.loc[col2, col1] = val
            
    return matrix
