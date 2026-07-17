"""
EDA Framework - Stage 5: Relationship Analysis
----------------------------------------------
Computes mixed-type pairwise association matrices and identifies potential feature collinearity and redundancies.
"""

import logging
import pandas as pd
from typing import Dict, Any, List, Optional
from eda_framework.utils.statistics import association_matrix

logger = logging.getLogger(__name__)

class RelationshipAnalyzer:
    def __init__(self, correlation_threshold: float = 0.85):
        self.correlation_threshold = correlation_threshold

    def analyze(self, df: pd.DataFrame, continuous_cols: List[str], categorical_cols: List[str]) -> Dict[str, Any]:
        """Compute association matrix and identify redundant/collinear column pairs."""
        all_cols = list(continuous_cols) + list(categorical_cols)
        all_cols = [c for c in all_cols if c in df.columns]
        
        if not all_cols:
            return {"association_matrix": {}, "high_correlations": []}

        assoc_df = association_matrix(df, continuous_cols, categorical_cols)
        
        # Search for high correlations (excluding self-correlation)
        high_correlations = []
        n = len(all_cols)
        for i in range(n):
            for j in range(i + 1, n):
                col1 = all_cols[i]
                col2 = all_cols[j]
                val = assoc_df.loc[col1, col2]
                
                if abs(val) >= self.correlation_threshold:
                    high_correlations.append({
                        "feature_1": col1,
                        "feature_2": col2,
                        "coefficient": float(val)
                    })

        return {
            "association_matrix": assoc_df.to_dict(),
            "high_correlations": high_correlations
        }
