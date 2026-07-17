"""
Tabular Feature Scaler
----------------------
Normalizes continuous features using:
1. MinMax Scaling (scaling to [0, 1]).
2. Standard Scaling (Z-score normalization).
Only scales continuous columns, leaving one-hot columns, label indices, and missing indicators untouched.
Saves scaling parameters for inverse_transform.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Any

__all__ = ["TabularScaler"]

logger = logging.getLogger(__name__)

VALID_SCALING_STRATEGIES = {"minmax", "standard", "log1p"}

class TabularScaler:
    def __init__(self, strategy: str = "minmax", feature_range: Tuple[float, float] = (0.0, 1.0), column_strategies: Optional[Dict[str, Dict[str, Any]]] = None):
        """
        Initialize the TabularScaler.
        
        Args:
            strategy: Scaling strategy ('minmax', 'standard', or 'log1p').
            feature_range: Targeted range for MinMax/log1p scaling (default is (0.0, 1.0)).
            column_strategies: Optional mapping of column names to specific config dicts (e.g. {'col_name': {'strategy': 'standard'}})
        
        Raises:
            ValueError: If strategy is not one of the supported values.
        """
        if strategy not in VALID_SCALING_STRATEGIES:
            raise ValueError(
                f"Invalid scaling strategy '{strategy}'. "
                f"Must be one of {sorted(VALID_SCALING_STRATEGIES)}"
            )
        self.strategy = strategy
        self.feature_range = feature_range
        self.column_strategies = column_strategies or {}
        
        for col, col_config in self.column_strategies.items():
            if "strategy" in col_config and col_config["strategy"] not in VALID_SCALING_STRATEGIES:
                raise ValueError(
                    f"Invalid scaling strategy '{col_config['strategy']}' for column '{col}'. "
                    f"Must be one of {sorted(VALID_SCALING_STRATEGIES)}"
                )
        
        # Fitted states
        self.continuous_cols_: List[str] = []
        self.means_: Dict[str, float] = {}
        self.stds_: Dict[str, float] = {}
        self.mins_: Dict[str, float] = {}
        self.maxs_: Dict[str, float] = {}
        self.is_fitted_ = False

    def fit(self, df: pd.DataFrame, continuous_cols: List[str]) -> "TabularScaler":
        """
        Learn scaling parameters (min, max, mean, std) from continuous columns.
        
        Args:
            df: Input DataFrame.
            continuous_cols: List of continuous column names to scale.
        
        Returns:
            Self reference for method chaining.
        """
        self.continuous_cols_ = list(continuous_cols)
        self.means_ = {}
        self.stds_ = {}
        self.mins_ = {}
        self.maxs_ = {}
        
        for col in self.continuous_cols_:
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            col_strat = self.column_strategies.get(col, {}).get("strategy", self.strategy)
            
            if len(series) == 0:
                self.means_[col] = 0.0
                self.stds_[col] = 1.0
                self.mins_[col] = 0.0
                self.maxs_[col] = 1.0
            else:
                self.means_[col] = float(series.mean())
                # Avoid std = 0
                std_val = float(series.std())
                self.stds_[col] = std_val if std_val > 0 else 1.0
                
                if col_strat == "log1p":
                    series_log = np.log1p(np.maximum(series, 0.0))
                    self.mins_[col] = float(series_log.min())
                    self.maxs_[col] = float(series_log.max())
                else:
                    self.mins_[col] = float(series.min())
                    self.maxs_[col] = float(series.max())
                
        self.is_fitted_ = True
        return self

    def fit_transform(self, df: pd.DataFrame, continuous_cols: List[str]) -> pd.DataFrame:
        """
        Fit to data, then transform it.
        
        Args:
            df: Input DataFrame.
            continuous_cols: List of continuous column names to scale.
        
        Returns:
            pd.DataFrame: Scaled DataFrame.
        """
        return self.fit(df, continuous_cols).transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale the continuous columns of the dataframe.
        
        Args:
            df: Input DataFrame with continuous columns to scale.
        
        Returns:
            pd.DataFrame: DataFrame with continuous columns scaled.
        """
        if not self.is_fitted_:
            raise ValueError("Scaler is not fitted yet. Call fit() first.")
            
        res_df = df.copy()
        
        for col in self.continuous_cols_:
            if col not in res_df.columns:
                continue
                
            col_strat = self.column_strategies.get(col, {}).get("strategy", self.strategy)
            
            if col_strat == "minmax":
                col_min = self.mins_[col]
                col_max = self.maxs_[col]
                rng = col_max - col_min
                if rng == 0:
                    rng = 1.0
                
                # Apply MinMax Formula
                scaled_col = (res_df[col] - col_min) / rng
                # Rescale to target feature range
                col_feat_range = self.column_strategies.get(col, {}).get("feature_range", self.feature_range)
                target_min, target_max = col_feat_range
                res_df[col] = scaled_col * (target_max - target_min) + target_min
                
            elif col_strat == "log1p":
                # Apply log1p first, clamping to 0.0 to prevent negative values in log
                x_log = np.log1p(np.maximum(res_df[col], 0.0))
                col_min = self.mins_[col]
                col_max = self.maxs_[col]
                rng = col_max - col_min
                if rng == 0:
                    rng = 1.0
                
                # Apply MinMax on log scale
                scaled_col = (x_log - col_min) / rng
                col_feat_range = self.column_strategies.get(col, {}).get("feature_range", self.feature_range)
                target_min, target_max = col_feat_range
                res_df[col] = scaled_col * (target_max - target_min) + target_min
                
            elif col_strat == "standard":
                col_mean = self.means_[col]
                col_std = self.stds_[col]
                # Apply Z-score
                res_df[col] = (res_df[col] - col_mean) / col_std
                
        return res_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert scaled numerical values back to their original range.
        
        Args:
            df: Input DataFrame with scaled values.
        
        Returns:
            pd.DataFrame: DataFrame with values restored to original scale.
        """
        if not self.is_fitted_:
            raise ValueError("Scaler is not fitted yet. Call fit() first.")
            
        res_df = df.copy()
        
        for col in self.continuous_cols_:
            if col not in res_df.columns:
                continue
                
            col_strat = self.column_strategies.get(col, {}).get("strategy", self.strategy)
            
            if col_strat == "minmax":
                col_min = self.mins_[col]
                col_max = self.maxs_[col]
                rng = col_max - col_min
                
                # Inverse feature range scaling first
                col_feat_range = self.column_strategies.get(col, {}).get("feature_range", self.feature_range)
                target_min, target_max = col_feat_range
                target_rng = target_max - target_min
                if target_rng == 0:
                    target_rng = 1.0
                    
                unscaled_val = (res_df[col] - target_min) / target_rng
                # Reconstruct original values
                res_df[col] = unscaled_val * rng + col_min
                
            elif col_strat == "log1p":
                # Inverse MinMax scaling on log scale
                col_min = self.mins_[col]
                col_max = self.maxs_[col]
                rng = col_max - col_min
                
                col_feat_range = self.column_strategies.get(col, {}).get("feature_range", self.feature_range)
                target_min, target_max = col_feat_range
                target_rng = target_max - target_min
                if target_rng == 0:
                    target_rng = 1.0
                    
                unscaled_val = (res_df[col] - target_min) / target_rng
                x_log = unscaled_val * rng + col_min
                
                # Reconstruct original values using expm1
                res_df[col] = np.expm1(x_log)
                
            elif col_strat == "standard":
                col_mean = self.means_[col]
                col_std = self.stds_[col]
                # Inverse Z-score
                res_df[col] = res_df[col] * col_std + col_mean
                
        return res_df
