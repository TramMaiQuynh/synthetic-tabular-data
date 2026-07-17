"""
Tabular Preprocessing Pipeline
------------------------------
Coordinates the ingestion, PII removal, missing value imputation,
categorical encoding, and feature scaling workflows.
Supports memory-based chunk size calculation and outputs artifacts to artifacts/<dataset_name>/.
"""

import os
import hashlib
import logging
import joblib
import pandas as pd
from typing import Dict, Optional
from src.config.config_loader import ConfigLoader

__all__ = ["PreprocessingPipeline"]

logger = logging.getLogger(__name__)

class PreprocessingPipeline:
    def __init__(self, dataset_name: str):
        """
        Initialize the preprocessing pipeline.
        
        Args:
            dataset_name: Name of the dataset folder under configs/
        """
        self.dataset_name = dataset_name
        self.config = ConfigLoader.load_config(dataset_name)
        
        # Load pipeline config
        self.pipeline_config = ConfigLoader.load_pipeline_config(dataset_name)
        
        # Load data schema
        self.schema = self._load_schema()
        
        # Extract features from schema and pipeline config
        self.pii_columns = list(set(self.schema.get("PII_columns_to_drop", []) + self.pipeline_config.get("columns_to_drop", [])))
        
        dropped_set = set(self.pii_columns)
        self.categorical_cols = [c for c in self.schema.get("categorical_features", {}).keys() if c not in dropped_set]
        self.continuous_cols = [c for c in self.schema.get("continuous_features", {}).keys() if c not in dropped_set]
        self.target_col = self.schema.get("target_column", "")
        
        # If target column is categorical, ensure it is in categorical list
        if self.target_col and self.target_col not in self.categorical_cols and self.target_col not in self.continuous_cols and self.target_col not in dropped_set:
            # Let's inspect the target column. Usually, it's categorical or continuous.
            # We assume it is treated as categorical by default unless listed in continuous.
            self.categorical_cols.append(self.target_col)
            
        # Initialize sub-modules
        # Read scaling strategy from config if available, default to "minmax"
        self.imputer = None
        self.encoder = None
        self.scaler = None
        self.is_fitted_ = False

        # Tracks original dtype per continuous column (e.g. 'int64' or 'float64').
        # Populated during fit_transform; used in inverse_transform for type-casting.
        self._original_dtypes: Dict[str, str] = {}
        
        # Set up artifact paths
        self.artifacts_dir = os.path.abspath(os.path.join(
            os.path.dirname(__file__), "..", "..", "artifacts", dataset_name
        ))

    def _load_schema(self) -> dict:
        """Loads data_schema.yaml for this dataset."""
        return ConfigLoader.load_schema(self.dataset_name)

    def estimate_chunk_rows(self, file_path: str) -> int:
        """
        Estimate chunk size (row count) to limit memory consumption to 10% of max_ram_gb.
        """
        # Determine file type
        _, ext = os.path.splitext(file_path.lower())
        
        # 1. Read first 1000 rows as a sample
        if ext == '.parquet':
            import pyarrow.parquet as pq
            parquet_file = pq.ParquetFile(file_path)
            try:
                first_batch = next(parquet_file.iter_batches(batch_size=1000))
                sample_df = first_batch.to_pandas()
            except StopIteration:
                sample_df = pd.DataFrame()
        elif ext in ['.xls', '.xlsx']:
            sample_df = pd.read_excel(file_path, nrows=1000)
        else:  # default to CSV
            # Respect ingestion configuration parameters
            read_kwargs = {"nrows": 1000}
            if hasattr(self.config, "ingestion"):
                sep = getattr(self.config.ingestion, "separator", ",")
                read_kwargs["sep"] = sep
                if len(sep) > 1:
                    read_kwargs["engine"] = "python"
                if hasattr(self.config.ingestion, "has_header") and not self.config.ingestion.has_header:
                    read_kwargs["header"] = None
                    if hasattr(self.config.ingestion, "columns") and self.config.ingestion.columns:
                        read_kwargs["names"] = self.config.ingestion.columns
                if hasattr(self.config.ingestion, "na_values") and self.config.ingestion.na_values:
                    read_kwargs["na_values"] = self.config.ingestion.na_values
            
            sample_df = pd.read_csv(file_path, **read_kwargs)
            
        if sample_df.empty:
            return 10000 # fallback default
            
        # 2. Measure memory footprint of the sample in Pandas
        mem_bytes = sample_df.memory_usage(deep=True).sum()
        bytes_per_row = max(1.0, mem_bytes / len(sample_df))
        
        # 3. Target memory is 10% of max_ram_gb (converted to bytes)
        max_ram_gb = self.config.model.max_ram_gb
        target_memory_bytes = 0.1 * max_ram_gb * 1e9
        
        # 4. Calculate rows per chunk
        chunk_rows = int(target_memory_bytes / bytes_per_row)
        return max(1000, chunk_rows) # ensure at least 1000 rows

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Load data from CSV or Parquet, checking if memory chunking is required.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Data file not found at {file_path}")
            
        # Get dynamic chunk rows
        chunk_rows = self.estimate_chunk_rows(file_path)
        
        # Let's check size of file (rough estimate)
        file_size_bytes = os.path.getsize(file_path)
        max_ram_bytes = self.config.model.max_ram_gb * 1e9
        
        _, ext = os.path.splitext(file_path.lower())
        
        # Prepare read options for CSV/Text files
        read_kwargs = {}
        if ext not in ['.parquet', '.xls', '.xlsx']:
            if hasattr(self.config, "ingestion"):
                sep = getattr(self.config.ingestion, "separator", ",")
                read_kwargs["sep"] = sep
                if len(sep) > 1:
                    read_kwargs["engine"] = "python"
                if hasattr(self.config.ingestion, "has_header") and not self.config.ingestion.has_header:
                    read_kwargs["header"] = None
                    if hasattr(self.config.ingestion, "columns") and self.config.ingestion.columns:
                        read_kwargs["names"] = self.config.ingestion.columns
                if hasattr(self.config.ingestion, "na_values") and self.config.ingestion.na_values:
                    read_kwargs["na_values"] = self.config.ingestion.na_values
        
        if file_size_bytes > 0.2 * max_ram_bytes:
            logger.info("File size is large (%.1f MB). Reading in chunks of %d rows.", file_size_bytes / 1e6, chunk_rows)
            chunks = []
            if ext == '.parquet':
                import pyarrow.parquet as pq
                parquet_file = pq.ParquetFile(file_path)
                for batch in parquet_file.iter_batches(batch_size=chunk_rows):
                    chunks.append(batch.to_pandas())
            elif ext in ['.xls', '.xlsx']:
                return pd.read_excel(file_path)
            else:
                read_kwargs["chunksize"] = chunk_rows
                for chunk in pd.read_csv(file_path, **read_kwargs):
                    chunks.append(chunk)
            if not chunks:
                return pd.DataFrame()
            return pd.concat(chunks, ignore_index=True)
        else:
            if ext == '.parquet':
                return pd.read_parquet(file_path)
            elif ext in ['.xls', '.xlsx']:
                return pd.read_excel(file_path)
            else:
                return pd.read_csv(file_path, **read_kwargs)

    def fit_transform(self, df: pd.DataFrame, model_type: Optional[str] = None) -> pd.DataFrame:
        """
        Execute full preprocessing: PII removal -> Imputer -> Encoder -> Scaler.

        Also records the original dtype of each continuous column so that
        inverse_transform can round integer-typed columns after decoding.
        """
        from src.preprocessing.imputer import TabularImputer
        from src.preprocessing.encoder import TabularEncoder
        from src.preprocessing.scaler import TabularScaler

        # 1. Record original dtypes BEFORE any coercion (for type-casting on inverse)
        self._original_dtypes = {}
        for col in self.continuous_cols:
            if col in df.columns:
                self._original_dtypes[col] = str(df[col].dtype)

        # 2. Type coercion of continuous columns to numeric and PII Removal
        res_df = df.copy()
        for col in self.continuous_cols:
            if col in res_df.columns:
                res_df[col] = pd.to_numeric(res_df[col], errors='coerce')

        if self.pii_columns:
            existing_pii = [col for col in self.pii_columns if col in res_df.columns]
            if existing_pii:
                res_df = res_df.drop(columns=existing_pii)

        # 3. Determine global feature range based on model_type
        resolved_model_type = model_type or self.config.model.model_type
        if resolved_model_type.lower() in ["ctgan", "ctvae"]:
            global_feature_range = (-1.0, 1.0)
        else:
            global_feature_range = (0.0, 1.0)

        # Map scaling strategies to options for TabularScaler
        scaling_opts = {}
        for col, strat in self.pipeline_config.get("scaling_strategy", {}).items():
            scaling_opts[col] = {"strategy": strat}

        # 4. Initialize sub-modules with column strategies
        self.imputer = TabularImputer(
            numeric_strategy="median",
            categorical_strategy="mode",
            column_strategies=self.pipeline_config.get("imputation_strategy", {})
        )
        self.encoder = TabularEncoder(
            max_onehot_cardinality=self.config.ingestion.max_onehot_cardinality,
            handle_unknown="ignore",
            scale_labels=True,
            column_strategies=self.pipeline_config.get("encoding_strategy", {})
        )
        self.scaler = TabularScaler(
            strategy="minmax",
            feature_range=global_feature_range,
            column_strategies=scaling_opts
        )

        # Adjust lists of features based on what remains in the dataframe
        current_cols = set(res_df.columns)
        active_continuous = [c for c in self.continuous_cols if c in current_cols]
        active_categorical = [c for c in self.categorical_cols if c in current_cols]

        # 5. Missing Value Imputation
        self.imputer.fit(res_df, active_continuous, active_categorical)
        res_df = self.imputer.transform(res_df)

        # 6. Categorical Encoding
        self.encoder.fit(res_df, active_categorical)
        res_df = self.encoder.transform(res_df)

        # 7. Scale continuous columns only (not one-hot / label / missing indicators)
        self.scaler.fit(res_df, active_continuous)
        res_df = self.scaler.transform(res_df)

        self.is_fitted_ = True
        return res_df

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply already-fitted transformations to a new dataset."""
        if not self.is_fitted_:
            raise ValueError("Pipeline is not fitted yet. Call fit_transform() first.")
            
        res_df = df.copy()
        for col in self.continuous_cols:
            if col in res_df.columns:
                res_df[col] = pd.to_numeric(res_df[col], errors='coerce')
                
        if self.pii_columns:
            existing_pii = [col for col in self.pii_columns if col in res_df.columns]
            if existing_pii:
                res_df = res_df.drop(columns=existing_pii)
                
        res_df = self.imputer.transform(res_df)
        res_df = self.encoder.transform(res_df)
        res_df = self.scaler.transform(res_df)
        return res_df

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        De-normalize and decode a preprocessed dataset back to raw format.

        Ordering (mandatory — see draft.md §2.7 technical note):
          1. Scaler inverse  — bring all values (including sentinels) to original scale.
          2. Encoder inverse — decode one-hot / label columns to string labels.
          3. Imputer inverse — restore NaN via is_missing flags; drop indicator cols.
          4. Type-casting    — round integer columns; clamp to schema min/max bounds.
        """
        if not self.is_fitted_:
            raise ValueError("Pipeline is not fitted yet. Call fit_transform() first.")

        res_df = df.copy()

        # Step 1: Unscale continuous columns (sentinel values decoded here too)
        res_df = self.scaler.inverse_transform(res_df)

        # Step 2: Decode categorical columns
        res_df = self.encoder.inverse_transform(res_df)

        # Step 3: Restore NaN and drop is_missing indicator columns
        res_df = self.imputer.inverse_transform(res_df)

        # Step 4: Type-casting — enforce original dtypes on continuous columns
        schema_continuous = self.schema.get("continuous_features", {})
        for col, orig_dtype in self._original_dtypes.items():
            if col not in res_df.columns:
                continue

            col_vals = pd.to_numeric(res_df[col], errors="coerce")

            # Clamp to schema-defined min/max bounds if present
            col_schema = schema_continuous.get(col, {})
            if isinstance(col_schema, dict):
                if "min" in col_schema:
                    col_vals = col_vals.clip(lower=float(col_schema["min"]))
                if "max" in col_schema:
                    col_vals = col_vals.clip(upper=float(col_schema["max"]))

            # Round if original dtype was integer (avoids float artifacts on e.g. 'tenure')
            if "int" in orig_dtype:
                col_vals = col_vals.round(0)
                # Attempt safe cast back to nullable integer to preserve NaN compatibility
                try:
                    res_df[col] = col_vals.astype("Int64")
                except (TypeError, ValueError):
                    res_df[col] = col_vals
            else:
                res_df[col] = col_vals

        return res_df

    def save_artifacts(self, path: Optional[str] = None) -> None:
        """Save fitted states of Imputer, Encoder, and Scaler to artifacts directory."""
        if not self.is_fitted_:
            raise ValueError("Pipeline is not fitted. Cannot save artifacts.")
            
        if path is not None:
            state_path = os.path.abspath(path)
        else:
            state_path = os.path.join(self.artifacts_dir, "preprocessing_pipeline.joblib")
            
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        
        pipeline_state = {
            "imputer": self.imputer,
            "encoder": self.encoder,
            "scaler": self.scaler,
            "categorical_cols": self.categorical_cols,
            "continuous_cols": self.continuous_cols,
            "pii_columns": self.pii_columns,
            "target_col": self.target_col,
            "original_dtypes": self._original_dtypes,
        }
        
        joblib.dump(pipeline_state, state_path)
        
        # Generate SHA256 checksum for integrity verification
        sha256_hash = hashlib.sha256()
        with open(state_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        checksum_path = state_path + ".sha256"
        with open(checksum_path, "w", encoding="utf-8") as f:
            f.write(sha256_hash.hexdigest())
            
        logger.info("Preprocessing artifacts saved to %s (checksum: %s)", state_path, checksum_path)

    def load_artifacts(self, path: Optional[str] = None) -> None:
        """Load fitted states of Imputer, Encoder, and Scaler from artifacts directory."""
        if path is not None:
            state_path = os.path.abspath(path)
        else:
            state_path = os.path.join(self.artifacts_dir, "preprocessing_pipeline.joblib")
            
        if not os.path.exists(state_path):
            raise FileNotFoundError(f"No fitted pipeline artifact found at {state_path}")
        
        # Verify SHA256 checksum before deserialization
        checksum_path = state_path + ".sha256"
        if os.path.exists(checksum_path):
            with open(checksum_path, "r", encoding="utf-8") as f:
                expected_hash = f.read().strip()
            sha256_hash = hashlib.sha256()
            with open(state_path, "rb") as f:
                for chunk in iter(lambda: f.read(8192), b""):
                    sha256_hash.update(chunk)
            actual_hash = sha256_hash.hexdigest()
            if actual_hash != expected_hash:
                raise RuntimeError(
                    f"Artifact integrity check failed for {state_path}. "
                    f"Expected SHA256: {expected_hash}, Got: {actual_hash}. "
                    f"The artifact file may have been tampered with."
                )
        else:
            logger.warning(
                "No SHA256 checksum file found at %s. "
                "Loading artifact without integrity verification.",
                checksum_path
            )
            
        state = joblib.load(state_path)
            
        self.imputer = state["imputer"]
        self.encoder = state["encoder"]
        self.scaler = state["scaler"]
        self.categorical_cols = state["categorical_cols"]
        self.continuous_cols = state["continuous_cols"]
        self.pii_columns = state["pii_columns"]
        self.target_col = state["target_col"]
        self._original_dtypes = state.get("original_dtypes", {})
        self.is_fitted_ = True
        
        logger.info("Preprocessing artifacts loaded from %s", state_path)
