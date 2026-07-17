"""
EDA Framework - Stage 4: Feature Profiling
------------------------------------------
Profiles numerical and categorical features to capture their univariate distributions, cardinality, outliers, and skewness.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from eda_framework.utils.statistics import detect_outliers_iqr, detect_outliers_zscore, detect_rare_categories

logger = logging.getLogger(__name__)

class FeatureProfiler:
    def __init__(self, config: Dict[str, Any]):
        self.rare_pct_threshold = config.get("profiling", {}).get("rare_category_pct_threshold", 5.0) / 100.0
        self.zscore_threshold = config.get("profiling", {}).get("outlier_zscore_threshold", 3.0)
        self.max_onehot = config.get("profiling", {}).get("max_onehot_cardinality", 10)

    def profile(self, df: pd.DataFrame, continuous_cols: List[str], categorical_cols: List[str]) -> Dict[str, Any]:
        """Profile continuous and categorical features independently."""
        continuous_profile = {}
        categorical_profile = {}

        # 1. Profile continuous features
        for col in continuous_cols:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            # Basic stats
            desc = series.describe()
            skew = series.skew()
            kurt = series.kurtosis()
            
            # Outlier detection
            iqr_lower, iqr_upper, iqr_count, iqr_pct, _ = detect_outliers_iqr(df, col)
            z_count, z_pct, _ = detect_outliers_zscore(df, col, self.zscore_threshold)
            
            # Dist recommendations
            recomm = "minmax"
            recomm_reason = "Standard minmax normalization"
            skew_val = float(skew) if not pd.isna(skew) else 0.0
            if abs(skew_val) > 1.5:
                # Highly skewed
                recomm = "log1p"
                recomm_reason = f"High skewness ({skew_val:.2f}). Log-transform recommended."

            continuous_profile[col] = {
                "count": int(desc["count"]),
                "mean": float(desc["mean"]),
                "std": float(desc["std"]) if not pd.isna(desc["std"]) else 0.0,
                "min": float(desc["min"]),
                "max": float(desc["max"]),
                "median": float(desc["50%"]),
                "skewness": float(skew) if not pd.isna(skew) else 0.0,
                "kurtosis": float(kurt) if not pd.isna(kurt) else 0.0,
                "iqr_outliers_pct": float(iqr_pct),
                "zscore_outliers_pct": float(z_pct),
                "scaling_recommendation": recomm,
                "scaling_reason": recomm_reason
            }

        # 2. Profile categorical features
        for col in categorical_cols:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            cardinality = int(series.nunique())
            
            # Detect rare categories
            rare_df = detect_rare_categories(df, col, self.rare_pct_threshold)
            rare_cats = rare_df.index.tolist() if not rare_df.empty else []
            
            # Imbalance check (entropy or max ratio)
            val_counts = series.value_counts(normalize=True)
            imbalanced = False
            imbalance_ratio = 1.0
            if not val_counts.empty:
                imbalance_ratio = float(val_counts.iloc[0] / val_counts.iloc[-1])
                # If majority class occupies > 80% or ratio is very high
                if val_counts.iloc[0] > 0.8:
                    imbalanced = True
            
            # Encoding recommendation
            encoding_recomm = "onehot"
            if cardinality > self.max_onehot:
                encoding_recomm = "label"

            categorical_profile[col] = {
                "cardinality": cardinality,
                "rare_categories": rare_cats,
                "is_highly_imbalanced": imbalanced,
                "imbalance_ratio": imbalance_ratio,
                "encoding_recommendation": encoding_recomm
            }

        return {
            "continuous_features": continuous_profile,
            "categorical_features": categorical_profile
        }
