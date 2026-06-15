"""
Tabular Categorical Encoder
----------------------------
Encodes categorical features using:
1. One-Hot Encoding if cardinality <= max_onehot_cardinality.
2. Label Encoding if cardinality > max_onehot_cardinality.
Handles unseen categories gracefully.
Provides inverse_transform to reconstruct categorical strings from numeric outputs.
"""

import logging
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional

__all__ = ["TabularEncoder"]

logger = logging.getLogger(__name__)

VALID_HANDLE_UNKNOWN = {"ignore", "error"}

class TabularEncoder:
    def __init__(self, max_onehot_cardinality: int = 10, handle_unknown: str = "ignore"):
        """
        Initialize the TabularEncoder.
        
        Args:
            max_onehot_cardinality: Maximum number of categories to apply One-Hot Encoding.
                                    Above this, Label Encoding is used.
            handle_unknown: Strategy for handling unseen categories ('ignore' or 'error').
        
        Raises:
            ValueError: If handle_unknown is not one of the supported values.
        """
        if handle_unknown not in VALID_HANDLE_UNKNOWN:
            raise ValueError(
                f"Invalid handle_unknown '{handle_unknown}'. "
                f"Must be one of {sorted(VALID_HANDLE_UNKNOWN)}"
            )
        self.max_onehot_cardinality = max_onehot_cardinality
        self.handle_unknown = handle_unknown
        
        # Fitted states
        self.categorical_cols_: List[str] = []
        self.encoding_types_: Dict[str, str] = {}      # col -> 'onehot' or 'label'
        self.categories_: Dict[str, List[str]] = {}      # col -> list of categories
        self.label_maps_: Dict[str, Dict[str, int]] = {} # col -> {cat_name: int_val}
        self.inverse_label_maps_: Dict[str, Dict[int, str]] = {} # col -> {int_val: cat_name}
        self.onehot_cols_: Dict[str, List[str]] = {}    # col -> list of onehot column names
        self.is_fitted_ = False

    def fit(self, df: pd.DataFrame, categorical_cols: List[str]) -> "TabularEncoder":
        """
        Fit the encoder on the categorical columns.
        
        Args:
            df: Input DataFrame.
            categorical_cols: List of categorical column names to encode.
        
        Returns:
            Self reference for method chaining.
        """
        self.original_cols_ = list(df.columns)
        self.categorical_cols_ = list(categorical_cols)
        self.encoding_types_ = {}
        self.categories_ = {}
        self.label_maps_ = {}
        self.inverse_label_maps_ = {}
        self.onehot_cols_ = {}
        
        for col in self.categorical_cols_:
            if col not in df.columns:
                continue
                
            # Get unique values, sorting them for consistency
            unique_vals = sorted(list(df[col].dropna().unique().astype(str)))
            self.categories_[col] = unique_vals
            
            # Determine encoding type
            cardinality = len(unique_vals)
            if cardinality <= self.max_onehot_cardinality:
                self.encoding_types_[col] = "onehot"
                # Store expected one-hot column names
                self.onehot_cols_[col] = [f"{col}_{val}" for val in unique_vals]
            else:
                self.encoding_types_[col] = "label"
                # Build label mapping dictionary
                # If handle_unknown is 'ignore', append an 'Unknown' category to the end
                cats = list(unique_vals)
                if self.handle_unknown == "ignore":
                    cats.append("Unknown")
                    
                self.label_maps_[col] = {val: idx for idx, val in enumerate(cats)}
                self.inverse_label_maps_[col] = {idx: val for idx, val in enumerate(cats)}
                
        self.is_fitted_ = True
        return self

    def fit_transform(self, df: pd.DataFrame, categorical_cols: List[str]) -> pd.DataFrame:
        """
        Fit to data, then transform it.
        
        Args:
            df: Input DataFrame.
            categorical_cols: List of categorical column names to encode.
        
        Returns:
            pd.DataFrame: Encoded DataFrame.
        """
        return self.fit(df, categorical_cols).transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform categorical columns to numeric representation.
        
        Args:
            df: Input DataFrame with categorical columns.
        
        Returns:
            pd.DataFrame: DataFrame with categorical columns encoded as numeric values.
        """
        if not self.is_fitted_:
            raise ValueError("Encoder is not fitted yet. Call fit() first.")
            
        res_df = df.copy()
        
        for col in self.categorical_cols_:
            if col not in res_df.columns:
                continue
                
            enc_type = self.encoding_types_[col]
            categories = self.categories_[col]
            
            if enc_type == "onehot":
                # Apply One-Hot Encoding
                # Convert column to string to ensure matching
                col_series = res_df[col].astype(str)
                
                # Generate binary columns for each category
                for val in categories:
                    onehot_col_name = f"{col}_{val}"
                    res_df[onehot_col_name] = (col_series == val).astype(np.float32)
                    
                # Drop original column
                res_df = res_df.drop(columns=[col])
                
            elif enc_type == "label":
                # Apply Label Encoding
                label_map = self.label_maps_[col]
                unknown_idx = label_map.get("Unknown", 0)  # default fallback
                
                def map_val(val, _label_map=label_map, _unknown_idx=unknown_idx, _col=col):
                    if pd.isnull(val):
                        return _unknown_idx
                    val_str = str(val)
                    if val_str in _label_map:
                        return _label_map[val_str]
                    else:
                        if self.handle_unknown == "error":
                            raise ValueError(f"Unseen category '{val_str}' found in column '{_col}'")
                        return _unknown_idx
                        
                res_df[col] = res_df[col].map(map_val).astype(np.float32)
                
        return res_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Decode numeric columns back to original categories.
        
        Args:
            df: Input DataFrame with encoded numeric values.
        
        Returns:
            pd.DataFrame: DataFrame with decoded categorical string values.
        """
        if not self.is_fitted_:
            raise ValueError("Encoder is not fitted yet. Call fit() first.")
            
        res_df = df.copy()
        
        for col in self.categorical_cols_:
            enc_type = self.encoding_types_[col]
            
            if enc_type == "onehot":
                onehot_cols = self.onehot_cols_[col]
                # Filter to onehot columns that actually exist in the dataframe
                existing_oh_cols = [c for c in onehot_cols if c in res_df.columns]
                
                if not existing_oh_cols:
                    continue
                    
                # Find argmax over the one-hot columns for each row
                # We extract the categories from the columns
                val_mapping = {c: c[len(col)+1:] for c in existing_oh_cols}
                
                # Perform argmax
                oh_data = res_df[existing_oh_cols]
                argmax_cols = oh_data.idxmax(axis=1)
                
                # Handle unseen categories where all one-hot columns are 0
                max_vals = oh_data.max(axis=1)
                decoded = argmax_cols.map(val_mapping)
                decoded = decoded.where(max_vals >= 0.5, None)
                
                # Map back to category values
                res_df[col] = decoded
                
                # Drop the one-hot columns
                res_df = res_df.drop(columns=existing_oh_cols)
                
            elif enc_type == "label":
                if col not in res_df.columns:
                    continue
                    
                inverse_map = self.inverse_label_maps_[col]
                max_idx = len(inverse_map) - 1
                
                # Round continuous inputs and clip to valid index ranges
                indices = np.clip(np.round(res_df[col].values), 0, max_idx).astype(int)
                
                # Map back
                res_df[col] = [inverse_map[idx] for idx in indices]
                
        # Reorder columns to match fit dataframe if original columns order is available
        if hasattr(self, "original_cols_") and self.original_cols_:
            target_order = [c for c in self.original_cols_ if c in res_df.columns]
            extra_cols = [c for c in res_df.columns if c not in target_order]
            res_df = res_df[target_order + extra_cols]
            
        return res_df
