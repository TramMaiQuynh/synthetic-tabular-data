"""
EDA Framework - Stage 1: Validation
-----------------------------------
Performs raw file structure checks, schema audits, data type checks, duplicate detection, and memory profiling.
"""

import os
import logging
import pandas as pd
from typing import Dict, Any, Tuple, List
from eda_framework.utils.helpers import detect_file_properties

logger = logging.getLogger(__name__)

class RawDataValidator:
    def __init__(self, file_path: str):
        self.file_path = file_path
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

    def inspect_raw_structure(self) -> Dict[str, Any]:
        """Detect file sep, encoding, and check if it has a header."""
        sep, encoding = detect_file_properties(self.file_path)
        
        # Read first few lines to detect header using csv.Sniffer
        try:
            with open(self.file_path, "r", encoding=encoding) as f:
                sample_lines = []
                for _ in range(10):
                    line = f.readline()
                    if not line:
                        break
                    sample_lines.append(line)
            
            if sample_lines:
                import csv
                sample = "".join(sample_lines)
                has_header = csv.Sniffer().has_header(sample)
            else:
                has_header = True
        except Exception as e:
            logger.warning("Error during header checking using csv.Sniffer, assuming True: %s", e)
            has_header = True

        return {
            "delimiter": sep,
            "encoding": encoding,
            "has_header": has_header
        }

    def load_and_profile(self, sep: str, encoding: str, **kwargs) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Load full dataframe and perform memory and duplicate checks."""
        df = pd.read_csv(self.file_path, sep=sep, encoding=encoding, **kwargs)
        
        # Profile memory usage
        mem_usage = df.memory_usage(deep=True)
        total_mem_bytes = mem_usage.sum()
        total_mem_mb = total_mem_bytes / (1024 * 1024)

        # Count duplicates
        duplicate_rows = int(df.duplicated().sum())
        duplicate_cols = 0
        # Check constant features
        constant_features = [col for col in df.columns if df[col].nunique(dropna=True) <= 1]
        
        # Near constant features (e.g. 99% values are same)
        near_constant_features = []
        for col in df.columns:
            if col not in constant_features:
                val_counts = df[col].value_counts(normalize=True)
                if not val_counts.empty and val_counts.iloc[0] >= 0.99:
                    near_constant_features.append(col)

        profile_report = {
            "shape": list(df.shape),
            "columns": list(df.columns),
            "memory_usage_mb": float(total_mem_mb),
            "duplicate_rows": duplicate_rows,
            "constant_features": constant_features,
            "near_constant_features": near_constant_features,
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        }

        return df, profile_report
