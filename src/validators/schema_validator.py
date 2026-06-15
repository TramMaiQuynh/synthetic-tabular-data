"""
Schema Validator
----------------
1. Fail-Fast Input Validation: Validates incoming raw datasets against structures
   and data types defined in data_schema.yaml.
2. Output Post-processing Audit & Correction: Validates that generated/decoded synthetic
   data complies with min/max bounds and category sets, correcting/clamping values if necessary.
"""

import os
import logging
import yaml
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Tuple
from src.config.config_loader import ConfigLoader

__all__ = ["SchemaValidator"]

logger = logging.getLogger(__name__)

class SchemaValidator:
    def __init__(self, dataset_name: str):
        """
        Initialize the SchemaValidator.
        
        Args:
            dataset_name: Name of the dataset folder under configs/
        """
        self.dataset_name = dataset_name
        self.schema = self._load_schema()
        
        self.categorical_features = self.schema.get("categorical_features", {})
        self.continuous_features = self.schema.get("continuous_features", {})
        self.target_column = self.schema.get("target_column", "")
        self.pii_columns = self.schema.get("PII_columns_to_drop", [])
        
    def _load_schema(self) -> dict:
        """Loads data_schema.yaml for this dataset."""
        return ConfigLoader.load_schema(self.dataset_name)

    def validate_input(self, df: pd.DataFrame, raise_error: bool = True) -> bool:
        """
        Validates the structure of the input raw dataframe against the schema.
        Ensures all non-PII features exist and are of correct datatypes.
        
        Args:
            df: Input DataFrame to validate.
            raise_error: If True, raises ValueError on validation failure.
        
        Returns:
            bool: True if validation passes, False otherwise.
        """
        errors = []
        
        # 1. Check required columns (excluding PII which should be dropped anyway)
        expected_cols = (
            list(self.continuous_features.keys()) + 
            list(self.categorical_features.keys())
        )
        if self.target_column and self.target_column not in expected_cols:
            expected_cols.append(self.target_column)
            
        # Filter out columns that are explicitly marked to drop as PII
        expected_cols = [c for c in expected_cols if c not in self.pii_columns]
        
        for col in expected_cols:
            if col not in df.columns:
                errors.append(f"Missing required column: '{col}'")
                
        # 2. Check continuous columns are numeric or coercible
        for col in self.continuous_features.keys():
            if col in df.columns and col not in self.pii_columns:
                if not pd.api.types.is_numeric_dtype(df[col]):
                    # Check if it can be coerced to numeric (not completely non-numeric)
                    coerced = pd.to_numeric(df[col], errors='coerce')
                    if coerced.isnull().all() and not df[col].isnull().all():
                        errors.append(f"Column '{col}' is continuous but contains non-numeric data type ({df[col].dtype}) and cannot be coerced to numbers.")
                    
        # Raise or return status
        if errors:
            msg = f"Fail-Fast Input validation failed for dataset '{self.dataset_name}':\n" + "\n".join(errors)
            if raise_error:
                raise ValueError(msg)
            else:
                logger.warning(msg)
                return False
                
        logger.info("Input data validated successfully against schema for '%s'", self.dataset_name)
        return True

    def audit_output(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Audit a synthetic dataframe against schema bounds and categories.
        Returns detailed statistics of any violations.
        
        Args:
            df: Synthetic DataFrame to audit.
        
        Returns:
            Dict[str, Any]: Audit report with violation statistics.
        """
        audit_report = {
            "dataset_name": self.dataset_name,
            "total_rows": len(df),
            "continuous_violations": {},
            "categorical_violations": {},
            "is_valid": True
        }
        
        # 1. Audit continuous features
        for col, bounds in self.continuous_features.items():
            if col not in df.columns:
                continue
                
            min_val = bounds.get("min")
            max_val = bounds.get("max")
            
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            under_min_mask = series < min_val
            over_max_mask = series > max_val
            
            under_min_count = int(under_min_mask.sum())
            over_max_count = int(over_max_mask.sum())
            
            if under_min_count > 0 or over_max_count > 0:
                audit_report["is_valid"] = False
                audit_report["continuous_violations"][col] = {
                    "under_min_count": under_min_count,
                    "under_min_pct": float(under_min_count / len(df) * 100),
                    "over_max_count": over_max_count,
                    "over_max_pct": float(over_max_count / len(df) * 100),
                    "bounds": [min_val, max_val]
                }
                
        # 2. Audit categorical features
        for col, allowed_cats in self.categorical_features.items():
            if col not in df.columns:
                continue
                
            series = df[col].dropna()
            if len(series) == 0:
                continue
                
            # Check for categories not in allowed list
            # Convert both to string to be safe
            allowed_set = set(str(c) for c in allowed_cats)
            # Add 'Unknown' if pipeline handles unknowns
            allowed_set.add("Unknown")
            allowed_set.add("missing")
            
            invalid_mask = ~series.astype(str).isin(allowed_set)
            invalid_count = int(invalid_mask.sum())
            
            if invalid_count > 0:
                audit_report["is_valid"] = False
                # Find sample invalid values
                invalid_samples = list(series[invalid_mask].astype(str).unique()[:5])
                audit_report["categorical_violations"][col] = {
                    "invalid_count": invalid_count,
                    "invalid_pct": float(invalid_count / len(df) * 100),
                    "invalid_samples": invalid_samples
                }
                
        return audit_report

    def audit_and_correct(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Audit the synthetic data, and automatically apply corrections (clamping bounds,
        and fallback mappings) to guarantee 100% compliance.
        Returns the corrected dataframe along with the audit report.
        
        Args:
            df: Synthetic DataFrame to audit and correct.
        
        Returns:
            Tuple[pd.DataFrame, Dict[str, Any]]: Corrected DataFrame and the audit report.
        """
        corrected_df = df.copy()
        report = self.audit_output(corrected_df)
        
        # 1. Clamp continuous bounds
        for col, bounds in self.continuous_features.items():
            if col not in corrected_df.columns:
                continue
            min_val = bounds.get("min")
            max_val = bounds.get("max")
            # In-place clipping
            corrected_df[col] = corrected_df[col].clip(lower=min_val, upper=max_val)
            
        # 2. Correct invalid categories
        for col, allowed_cats in self.categorical_features.items():
            if col not in corrected_df.columns:
                continue
            
            allowed_set = set(str(c) for c in allowed_cats)
            allowed_set.add("Unknown")
            allowed_set.add("missing")
            
            series = corrected_df[col]
            # Find first allowed category as fallback mode
            fallback = allowed_cats[0] if allowed_cats else "Unknown"
            
            # Map invalid items to the fallback
            def correct_cat(val, _allowed_set=allowed_set, _fallback=fallback):
                if pd.isnull(val):
                    return val
                val_str = str(val)
                return val_str if val_str in _allowed_set else _fallback
                
            corrected_df[col] = series.map(correct_cat)
            
        return corrected_df, report
