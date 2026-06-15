"""
Inference Sampler
------------------
Loads a trained generative model and produces a post-processed synthetic
DataFrame in a single call, executing the full decoding pipeline:

  1. Load model checkpoint from artifacts/<dataset_name>/checkpoints/.
  2. Load preprocessing pipeline from artifacts/<dataset_name>/.
  3. Generate data_tensor in batches (OOM-safe).
  4. Apply ConstraintsEngine.post_correction() on the decoded DataFrame.
  5. Call PreprocessingPipeline.inverse_transform() to recover original space.
  6. Apply type-casting: round integer columns, clamp to schema bounds.
  7. Return a clean pd.DataFrame ready for downstream consumption.

Conditional generation is supported by passing condition_col + condition_val.

Design principles:
  - Sampler is read-only: it never modifies checkpoints or configs.
  - Chunked generation prevents OOM for large n_rows requests.
  - All decode / type-cast logic is delegated to pipeline.inverse_transform()
    (single source of truth). Sampler does not duplicate that logic.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

__all__ = ["SyntheticSampler"]

logger = logging.getLogger(__name__)

SUPPORTED_MODELS = {"ctgan", "ctvae", "diffusion"}


class SyntheticSampler:
    """
    Loads a trained model and generates decoded synthetic data.

    Parameters
    ----------
    model_type : str
        'ctgan', 'ctvae', or 'diffusion'.
    dataset_name : str
        Dataset identifier — used to resolve artifact and checkpoint paths.
    artifacts_root : str
        Root path for artifact storage.
    device : str or None
        Compute device for sampling. Auto-detected if None.
    """

    def __init__(
        self,
        model_type: str,
        dataset_name: str,
        artifacts_root: str,
        device: Optional[str] = None,
    ) -> None:
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Must be one of {sorted(SUPPORTED_MODELS)}."
            )
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.artifacts_root = artifacts_root
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self._model = None
        self._pipeline = None
        self._col_meta = None

        logger.info(
            "SyntheticSampler init: model='%s', dataset='%s', device='%s'.",
            model_type, dataset_name, self.device,
        )

    # ------------------------------------------------------------------
    # Load artefacts
    # ------------------------------------------------------------------

    def load(
        self,
        checkpoint_path: Optional[str] = None,
        pipeline_path: Optional[str] = None,
    ) -> "SyntheticSampler":
        """
        Load model checkpoint and preprocessing pipeline.

        Args:
            checkpoint_path : Override path to model .pt file.
            pipeline_path   : Override path to pipeline .joblib file.

        Returns:
            self (for method chaining).
        """
        # Resolve checkpoint path
        if checkpoint_path is None:
            checkpoint_path = os.path.join(
                self.artifacts_root,
                self.dataset_name,
                "checkpoints",
                f"{self.model_type}_model.pt",
            )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Model checkpoint not found at '{checkpoint_path}'. "
                "Run ModelTrainer.train() first."
            )

        # Load model
        if self.model_type == "ctgan":
            from src.models.ctgan import TabularCTGAN
            self._model = TabularCTGAN.load(checkpoint_path, device=self.device)
        elif self.model_type == "ctvae":
            from src.models.ctvae import TabularCTVAE
            self._model = TabularCTVAE.load(checkpoint_path, device=self.device)
        elif self.model_type == "diffusion":
            from src.models.diffusion import TabularDiffusion
            self._model = TabularDiffusion.load(checkpoint_path, device=self.device)

        # Recover col_meta from model
        self._col_meta = self._model.col_meta
        logger.info("Model loaded from '%s'.", checkpoint_path)

        # Resolve pipeline path
        if pipeline_path is None:
            pipeline_path = os.path.join(
                self.artifacts_root,
                self.dataset_name,
                "preprocessing_pipeline.joblib",
            )

        if os.path.exists(pipeline_path):
            from src.preprocessing.pipeline import PreprocessingPipeline
            self._pipeline = PreprocessingPipeline(dataset_name=self.dataset_name)
            self._pipeline.load_artifacts()
            logger.info("Preprocessing pipeline loaded from '%s'.", pipeline_path)
        else:
            logger.warning(
                "Preprocessing pipeline artifact not found at '%s'. "
                "inverse_transform will be skipped — output will be in encoded space.",
                pipeline_path,
            )

        return self

    # ------------------------------------------------------------------
    # Main sampling entry point
    # ------------------------------------------------------------------

    def generate(
        self,
        n_rows: int,
        condition_col: Optional[str] = None,
        condition_val: Optional[str] = None,
        batch_size: int = 2048,
        constraint_expressions: Optional[List[str]] = None,
        max_constraint_retries: int = 5,
        return_raw_tensor: bool = False,
    ) -> pd.DataFrame:
        """
        Generate n_rows synthetic records.

        Args:
            n_rows                : Number of synthetic rows to produce.
            condition_col         : Column to condition on (optional).
            condition_val         : Category value to fix (optional).
            batch_size            : Rows per generation batch (OOM control).
            constraint_expressions: Constraints to enforce post-generation.
            max_constraint_retries: Max retries in ConstraintsEngine.
            return_raw_tensor     : If True, return raw tensor (skip decoding).

        Returns:
            pd.DataFrame with decoded, type-cast synthetic data.
            If pipeline not loaded, returns DataFrame in encoded space.
        """
        if self._model is None:
            raise RuntimeError("Sampler is not loaded. Call load() first.")

        if n_rows <= 0:
            raise ValueError(f"n_rows must be > 0, got {n_rows}.")

        logger.info(
            "Generating %d synthetic rows (model='%s', batch_size=%d).",
            n_rows, self.model_type, batch_size,
        )

        # 1. Generate raw tensor
        raw_tensor = self._model.sample(
            n_rows=n_rows,
            condition_col=condition_col,
            condition_val=condition_val,
            batch_size=batch_size,
        )

        if return_raw_tensor:
            # Convert to DataFrame in encoded column order
            return self._tensor_to_encoded_df(raw_tensor)

        # 2. Convert tensor to encoded DataFrame
        encoded_df = self._tensor_to_encoded_df(raw_tensor)
        logger.info("Raw tensor converted to encoded DataFrame: shape=%s.", encoded_df.shape)

        # 3. Apply post-generation constraint correction (on encoded space)
        #    NOTE: constraint expressions reference *original* column names, so correction
        #    must be applied AFTER inverse_transform. We defer it below.

        # 4. Inverse-transform to original space
        if self._pipeline is not None:
            decoded_df = self._pipeline.inverse_transform(encoded_df)
            logger.info("Inverse transform complete: shape=%s.", decoded_df.shape)
        else:
            decoded_df = encoded_df
            logger.warning("Returning encoded DataFrame (pipeline not loaded).")

        # 5. Post-generation constraint correction (on decoded space)
        if constraint_expressions:
            from src.models.constraints import ConstraintsEngine

            engine = ConstraintsEngine(
                expressions=constraint_expressions,
                max_retries=max_constraint_retries,
            )
            decoded_df, correction_stats = engine.post_correction(decoded_df)
            logger.info(
                "Constraint correction: violation_rate %.4f -> %.4f.",
                correction_stats["violation_rate_before"],
                correction_stats["violation_rate_after"],
            )

        return decoded_df

    # ------------------------------------------------------------------
    # Tensor -> DataFrame conversion
    # ------------------------------------------------------------------

    def _tensor_to_encoded_df(self, tensor: torch.Tensor) -> pd.DataFrame:
        """
        Convert raw output tensor to an encoded DataFrame using col_meta.
        Reconstructs one-hot column groups and assigns correct column names.
        """
        if self._col_meta is None:
            raise RuntimeError("col_meta not loaded. Call load() first.")

        arr = tensor.numpy()
        columns: List[str] = []
        offset = 0

        for meta in self._col_meta:
            if meta.col_type == "onehot":
                # During encoding, one-hot columns are named <orig>_<val>
                # We cannot recover original category values from col_meta alone
                # (col_meta only stores dim), so we use positional placeholders
                # that match the encoder's column naming convention.
                # The pipeline's encoder stores the exact column names — we rely
                # on it to handle inverse_transform correctly.
                if self._pipeline is not None:
                    enc = self._pipeline.encoder
                    if enc is not None and meta.name in enc.onehot_cols_:
                        oh_names = enc.onehot_cols_[meta.name]
                        columns.extend(oh_names[:meta.dim])
                        offset += meta.dim
                        continue
                # Fallback: positional names
                for i in range(meta.dim):
                    columns.append(f"{meta.name}__oh_{i}")
                offset += meta.dim
            else:
                columns.append(meta.name)
                offset += 1

        # Ensure column count matches tensor width
        n_tensor_cols = arr.shape[1]
        if len(columns) < n_tensor_cols:
            columns += [f"__extra_{i}" for i in range(len(columns), n_tensor_cols)]
        elif len(columns) > n_tensor_cols:
            columns = columns[:n_tensor_cols]

        return pd.DataFrame(arr, columns=columns)

    # ------------------------------------------------------------------
    # Convenience: generate and save to file
    # ------------------------------------------------------------------

    def generate_and_save(
        self,
        n_rows: int,
        output_path: str,
        **generate_kwargs: Any,
    ) -> str:
        """
        Generate synthetic data and write to CSV or Parquet.

        Args:
            n_rows       : Number of synthetic rows.
            output_path  : Destination file path (.csv or .parquet).
            **generate_kwargs: Forwarded to generate().

        Returns:
            Absolute path to the saved file.
        """
        df = self.generate(n_rows=n_rows, **generate_kwargs)
        output_path = os.path.abspath(output_path)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

        _, ext = os.path.splitext(output_path.lower())
        if ext == ".parquet":
            df.to_parquet(output_path, index=False)
        else:
            df.to_csv(output_path, index=False)

        logger.info(
            "Synthetic data (%d rows, %d cols) saved to '%s'.",
            len(df), len(df.columns), output_path,
        )
        return output_path
