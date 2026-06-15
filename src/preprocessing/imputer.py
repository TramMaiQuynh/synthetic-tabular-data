"""
Tabular Imputer
---------------
Handles missing values by:
1. Creating binary indicator columns (`<col>_is_missing`) for columns with missing values.
2. Imputing numerical columns with Mean or Median.
3. Imputing categorical columns with Mode or a placeholder 'missing'.
Saves fitted statistics for replication on test data or during inference.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Union

__all__ = ["TabularImputer"]

logger = logging.getLogger(__name__)

VALID_NUMERIC_STRATEGIES = {"mean", "median"}
VALID_CATEGORICAL_STRATEGIES = {"mode", "missing"}

class TabularImputer:
    def __init__(self, numeric_strategy: str = "median", categorical_strategy: str = "mode"):
        """
        Initialize the TabularImputer.
        
        Args:
            numeric_strategy: Strategy for numerical columns ('mean' or 'median').
            categorical_strategy: Strategy for categorical columns ('mode' or 'missing').
        
        Raises:
            ValueError: If strategy is not one of the supported values.
        """
        if numeric_strategy not in VALID_NUMERIC_STRATEGIES:
            raise ValueError(
                f"Invalid numeric_strategy '{numeric_strategy}'. "
                f"Must be one of {sorted(VALID_NUMERIC_STRATEGIES)}"
            )
        if categorical_strategy not in VALID_CATEGORICAL_STRATEGIES:
            raise ValueError(
                f"Invalid categorical_strategy '{categorical_strategy}'. "
                f"Must be one of {sorted(VALID_CATEGORICAL_STRATEGIES)}"
            )
        self.numeric_strategy = numeric_strategy
        self.categorical_strategy = categorical_strategy
        
        # Fitted states
        self.impute_values_: Dict[str, Union[float, str]] = {}
        self.indicator_cols_: List[str] = []  # Columns that had missing values during fit
        self.continuous_cols_: List[str] = []
        self.categorical_cols_: List[str] = []
        self.is_fitted_ = False

    def fit(self, df: pd.DataFrame, continuous_cols: List[str], categorical_cols: List[str]) -> "TabularImputer":
        """
        Learn the imputation values and identify columns with missing data.
        
        Args:
            df: Input DataFrame.
            continuous_cols: List of continuous column names.
            categorical_cols: List of categorical column names.
        
        Returns:
            Self reference for method chaining.
        """
        self.continuous_cols_ = list(continuous_cols)
        self.categorical_cols_ = list(categorical_cols)
        self.impute_values_ = {}
        self.indicator_cols_ = []
        
        # 1. Process continuous columns
        for col in self.continuous_cols_:
            if col not in df.columns:
                continue
            
            # Check if column has missing values
            if df[col].isnull().any():
                self.indicator_cols_.append(col)
                
            # Calculate imputation value
            non_null_series = df[col].dropna()
            if len(non_null_series) == 0:
                # Fallback if the whole column is NaN
                val = 0.0
            elif self.numeric_strategy == "mean":
                val = float(non_null_series.mean())
            else:
                val = float(non_null_series.median())
            self.impute_values_[col] = val
            
        # 2. Process categorical columns
        for col in self.categorical_cols_:
            if col not in df.columns:
                continue
                
            # Check if column has missing values
            if df[col].isnull().any():
                self.indicator_cols_.append(col)
                
            # Calculate imputation value
            non_null_series = df[col].dropna()
            if len(non_null_series) == 0:
                val = "missing"
            elif self.categorical_strategy == "mode":
                # Mode might return multiple values; take the first one
                mode_res = non_null_series.mode()
                val = str(mode_res.iloc[0]) if not mode_res.empty else "missing"
            else:
                val = "missing"
            self.impute_values_[col] = val
            
        self.is_fitted_ = True
        return self

    def fit_transform(self, df: pd.DataFrame, continuous_cols: List[str], categorical_cols: List[str]) -> pd.DataFrame:
        """
        Fit to data, then transform it.
        
        Args:
            df: Input DataFrame.
            continuous_cols: List of continuous column names.
            categorical_cols: List of categorical column names.
        
        Returns:
            pd.DataFrame: Imputed DataFrame with indicator columns.
        """
        return self.fit(df, continuous_cols, categorical_cols).transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply imputation and append binary indicator columns.
        
        Args:
            df: Input DataFrame.
        
        Returns:
            pd.DataFrame: DataFrame with missing values filled and indicator columns appended.
        """
        if not self.is_fitted_:
            raise ValueError("Imputer is not fitted yet. Call fit() first.")
            
        res_df = df.copy()
        
        # 1. Create binary indicator columns first (using fitted indicator list)
        for col in self.continuous_cols_ + self.categorical_cols_:
            # Only create indicator if the column was flagged during fit
            if col in self.indicator_cols_:
                indicator_col_name = f"{col}_is_missing"
                # If column is missing in input df, it is treated as fully missing
                if col in res_df.columns:
                    res_df[indicator_col_name] = res_df[col].isnull().astype(np.float32)
                else:
                    res_df[indicator_col_name] = np.ones(len(res_df), dtype=np.float32)
                    
        # 2. Perform imputation in-place on continuous and categorical columns
        for col, val in self.impute_values_.items():
            if col in res_df.columns:
                res_df[col] = res_df[col].fillna(val)
                
        return res_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Restore NaNs to the original columns based on the binary indicator columns,
        and drop the indicator columns.
        
        Args:
            df: Input DataFrame containing indicator columns.
        
        Returns:
            pd.DataFrame: DataFrame with NaN values restored and indicator columns removed.
        """
        if not self.is_fitted_:
            raise ValueError("Imputer is not fitted yet. Call fit() first.")
            
        res_df = df.copy()
        
        # Restore NaNs
        for col in self.indicator_cols_:
            indicator_col_name = f"{col}_is_missing"
            if indicator_col_name in res_df.columns:
                # If indicator is > 0.5 (handles continuous generation noise), set to NaN
                # We also need to be careful with type conversions (e.g. integer types can't hold NaN, pandas converts to float or Object)
                is_missing_mask = res_df[indicator_col_name] > 0.5
                
                # Check column type to avoid pandas dtype issues
                if col in res_df.columns:
                    res_df.loc[is_missing_mask, col] = np.nan
                    
                # Drop indicator column
                res_df = res_df.drop(columns=[indicator_col_name])
                
        return res_df
