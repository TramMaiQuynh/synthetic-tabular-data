"""
EDA Framework - Stage 3: Missing Value Analysis
-----------------------------------------------
Measures missing value counts and ratios, computes nullity correlations, and suggests imputation strategies.
"""

import logging
import pandas as pd
from typing import Dict, Any, List
from eda_framework.utils.statistics import calculate_nullity_correlation

logger = logging.getLogger(__name__)

class MissingValueAnalyzer:
    def __init__(self, config: Dict[str, Any]):
        self.drop_threshold = config.get("missing", {}).get("drop_missing_pct_threshold", 50.0)

    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform missingness profiling on the DataFrame."""
        n_rows = len(df)
        missing_report = {}
        columns_to_drop = []
        imputation_recommendations = {}

        null_counts = df.isnull().sum()
        null_pcts = (null_counts / n_rows) * 100

        for col in df.columns:
            count = int(null_counts[col])
            pct = float(null_pcts[col])
            
            missing_report[col] = {
                "missing_count": count,
                "missing_pct": pct
            }
            
            if pct > self.drop_threshold:
                columns_to_drop.append(col)
            elif count > 0:
                # Recommend strategy
                if pd.api.types.is_numeric_dtype(df[col]):
                    # Check skewness to recommend mean vs median
                    skew = df[col].skew(skipna=True)
                    if abs(skew) > 1.0 if not pd.isna(skew) else False:
                        imputation_recommendations[col] = "median"
                    else:
                        imputation_recommendations[col] = "mean"
                else:
                    imputation_recommendations[col] = "mode"

        # Calculate nullity correlation
        null_corr_df = calculate_nullity_correlation(df)
        null_corr_matrix = null_corr_df.to_dict() if not null_corr_df.empty else {}

        return {
            "missing_report": missing_report,
            "columns_to_drop_high_missing": columns_to_drop,
            "imputation_recommendations": imputation_recommendations,
            "nullity_correlation_matrix": null_corr_matrix
        }
