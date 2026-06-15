"""
Model Trainer
-------------
Orchestrates the end-to-end training workflow for a single generative model:

  1. Build ColumnMeta list from the preprocessed DataFrame schema.
  2. Convert the preprocessed DataFrame to a float32 torch.Tensor.
  3. Instantiate the requested model (CTGAN / CTVAE / Diffusion).
  4. Optionally attach DPTrainer for DP-SGD.
  5. Optionally build ConstraintsEngine for soft-loss penalties.
  6. Execute training and return loss history.
  7. Save model checkpoint to artifacts/<dataset_name>/checkpoints/.

The Trainer is the sole entry point for model training. It reads all
configuration from the AppConfig / data_schema objects and exposes a
clean `train()` method. It does NOT manage HPO (see hpo.py).

Responsibilities scoped OUT of Trainer:
  - Hyperparameter search (hpo.py).
  - Data loading and preprocessing (PreprocessingPipeline).
  - Post-generation sampling and constraint correction (sampler.py).
"""

from __future__ import annotations

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from src.models.ctgan import ColumnMeta, TabularCTGAN
from src.models.ctvae import TabularCTVAE
from src.models.diffusion import TabularDiffusion
from src.models.constraints import ConstraintsEngine
from src.training.dp_training import DPTrainer

__all__ = ["ModelTrainer", "build_col_meta"]

logger = logging.getLogger(__name__)

# Supported model type identifiers
SUPPORTED_MODELS = {"ctgan", "ctvae", "diffusion"}


# ---------------------------------------------------------------------------
# ColumnMeta builder
# ---------------------------------------------------------------------------

def build_col_meta(
    df: pd.DataFrame,
    continuous_cols: List[str],
    categorical_cols: List[str],
    max_onehot_cardinality: int = 10,
) -> Tuple[List[ColumnMeta], Dict[str, List[str]], Dict[str, int]]:
    """
    Construct ColumnMeta list from the *preprocessed* DataFrame schema.

    The preprocessed DataFrame contains:
      - One-hot encoded columns (col_cat1, col_cat2, ... for low-cardinality).
      - Label-encoded columns (single float column for high-cardinality cats).
      - Scaled continuous columns (single float each).
      - Binary indicator columns (*_is_missing).

    Returns:
        col_meta      : List[ColumnMeta] describing each column group.
        categories    : {original_cat_col: [cat_val, ...]} for conditional sampler.
        col_name_index: {original_col_name: start_index_in_tensor} for constraint mapping.
    """
    col_meta: List[ColumnMeta] = []
    col_name_index: Dict[str, int] = {}
    categories: Dict[str, List[str]] = {}
    tensor_idx = 0

    # 1. Continuous columns (already scaled to [0,1] or z-scored)
    for col in continuous_cols:
        if col in df.columns:
            col_meta.append(ColumnMeta(name=col, col_type="continuous", dim=1))
            col_name_index[col] = tensor_idx
            tensor_idx += 1

    # 2. Categorical columns — detect one-hot groups vs label-encoded
    for orig_col in categorical_cols:
        # Detect one-hot columns generated from this original column
        onehot_cols = [c for c in df.columns if c.startswith(f"{orig_col}_") and
                       c not in [f"{orig_col}_is_missing"]]

        if len(onehot_cols) > 1:
            # One-hot encoded
            cats = [c[len(orig_col) + 1:] for c in sorted(onehot_cols)]
            categories[orig_col] = cats
            col_meta.append(ColumnMeta(name=orig_col, col_type="onehot", dim=len(onehot_cols)))
            col_name_index[orig_col] = tensor_idx
            tensor_idx += len(onehot_cols)
        elif orig_col in df.columns:
            # Label-encoded (single column)
            col_meta.append(ColumnMeta(name=orig_col, col_type="label", dim=1))
            col_name_index[orig_col] = tensor_idx
            tensor_idx += 1

    # 3. Binary indicator columns (*_is_missing)
    for col in df.columns:
        if col.endswith("_is_missing"):
            col_meta.append(ColumnMeta(name=col, col_type="continuous", dim=1))
            col_name_index[col] = tensor_idx
            tensor_idx += 1

    return col_meta, categories, col_name_index


def _df_to_tensor(df: pd.DataFrame, col_meta: List[ColumnMeta]) -> torch.Tensor:
    """
    Convert a preprocessed DataFrame to a float32 Tensor in the order
    prescribed by col_meta.

    Handles one-hot groups by concatenating their columns in the stored order.
    """
    parts: List[torch.Tensor] = []
    for meta in col_meta:
        if meta.col_type == "onehot":
            # Find matching columns by prefix (e.g. "Contract_" → all one-hot columns)
            actual_cols = sorted([c for c in df.columns if c.startswith(f"{meta.name}_") and
                                   not c.endswith("_is_missing")])
            if actual_cols:
                chunk = torch.tensor(df[actual_cols].values, dtype=torch.float32)
            else:
                chunk = torch.zeros(len(df), meta.dim)
            parts.append(chunk)
        elif meta.name in df.columns:
            parts.append(torch.tensor(df[meta.name].values, dtype=torch.float32).unsqueeze(1))
        else:
            logger.warning("Column '%s' not found in DataFrame — filling with zeros.", meta.name)
            parts.append(torch.zeros(len(df), meta.dim))

    return torch.cat(parts, dim=1)


# ---------------------------------------------------------------------------
# ModelTrainer
# ---------------------------------------------------------------------------

class ModelTrainer:
    """
    Orchestrates training for CTGAN, CTVAE, or TabularDiffusion.

    Parameters
    ----------
    model_type : str
        One of 'ctgan', 'ctvae', 'diffusion'.
    dataset_name : str
        Dataset name — used to resolve artifact paths.
    artifacts_root : str
        Root path for writing model checkpoints.
    """

    def __init__(
        self,
        model_type: str,
        dataset_name: str,
        artifacts_root: str,
    ) -> None:
        if model_type not in SUPPORTED_MODELS:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                f"Must be one of {sorted(SUPPORTED_MODELS)}."
            )
        self.model_type = model_type
        self.dataset_name = dataset_name
        self.artifacts_root = artifacts_root

        self._model = None
        self._col_meta: List[ColumnMeta] = []
        self._categories: Dict[str, List[str]] = {}
        self._col_name_index: Dict[str, int] = {}

        logger.info(
            "ModelTrainer initialised: model_type='%s', dataset='%s'.",
            model_type, dataset_name,
        )

    # ------------------------------------------------------------------
    # Main training entry point
    # ------------------------------------------------------------------

    def train(
        self,
        preprocessed_df: pd.DataFrame,
        continuous_cols: List[str],
        categorical_cols: List[str],
        epochs: int = 100,
        batch_size: int = 256,
        lr: float = 2e-4,
        weight_decay: float = 1e-6,
        max_onehot_cardinality: int = 10,
        # DP-SGD config
        enable_dp: bool = False,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        max_grad_norm: float = 1.0,
        noise_multiplier: Optional[float] = None,
        dp_backend: str = "auto",
        # Constraint config
        constraint_expressions: Optional[List[str]] = None,
        max_constraint_retries: int = 5,
        # Model-specific hyper-parameters
        model_kwargs: Optional[Dict[str, Any]] = None,
        # Early stopping (disabled under DP)
        early_stopping_patience: int = 0,
    ) -> Dict[str, Any]:
        """
        Train a generative model.

        Args:
            preprocessed_df      : DataFrame output of PreprocessingPipeline.fit_transform().
            continuous_cols      : Original continuous column names.
            categorical_cols     : Original categorical column names.
            epochs               : Training epochs.
            batch_size           : Mini-batch size.
            lr                   : Learning rate.
            weight_decay         : L2 regularisation.
            max_onehot_cardinality: Threshold used during encoding (for col_meta building).
            enable_dp            : Enable DP-SGD.
            target_epsilon       : Privacy budget epsilon.
            target_delta         : Privacy failure probability.
            max_grad_norm        : Gradient clipping norm.
            noise_multiplier     : Noise sigma / max_grad_norm (None = auto).
            dp_backend           : 'opacus' | 'custom' | 'auto'.
            constraint_expressions: List of constraint expression strings.
            max_constraint_retries: Max retries in ConstraintsEngine.
            model_kwargs         : Additional kwargs passed to the model constructor.
            early_stopping_patience: 0 = disabled. Auto-disabled under DP.

        Returns:
            dict containing:
                'loss_history'  : loss dict from model.fit().
                'epsilon'       : final consumed epsilon (inf if DP disabled).
                'checkpoint_path': path to saved model checkpoint.
        """
        model_kwargs = model_kwargs or {}

        # 1. Build ColumnMeta + tensor
        logger.info("Building ColumnMeta from preprocessed DataFrame (shape=%s).", preprocessed_df.shape)
        self._col_meta, self._categories, self._col_name_index = build_col_meta(
            preprocessed_df, continuous_cols, categorical_cols, max_onehot_cardinality,
        )
        data_tensor = _df_to_tensor(preprocessed_df, self._col_meta)
        logger.info(
            "Data tensor: shape=%s, dtype=%s.", data_tensor.shape, data_tensor.dtype
        )

        # 2. Build constraints engine (optional)
        constraints_engine: Optional[ConstraintsEngine] = None
        if constraint_expressions:
            constraints_engine = ConstraintsEngine(
                expressions=constraint_expressions,
                max_retries=max_constraint_retries,
            )

        # 3. Build DP trainer (optional)
        dp_trainer: Optional[DPTrainer] = None
        if enable_dp:
            dp_trainer = DPTrainer(
                target_epsilon=target_epsilon,
                target_delta=target_delta,
                max_grad_norm=max_grad_norm,
                noise_multiplier=noise_multiplier,
                backend=dp_backend,
            )
            if early_stopping_patience > 0:
                logger.warning(
                    "early_stopping_patience=%d ignored — DP-SGD is active.",
                    early_stopping_patience,
                )
                early_stopping_patience = 0

        # 4. Instantiate model
        self._model = self._build_model(model_kwargs)

        # 5. Train
        logger.info("Starting training: model_type='%s', epochs=%d.", self.model_type, epochs)
        train_kwargs: Dict[str, Any] = dict(
            data_tensor=data_tensor,
            epochs=epochs,
            batch_size=batch_size,
            weight_decay=weight_decay,
            constraints_engine=constraints_engine,
            col_name_index=self._col_name_index,
            dp_trainer=dp_trainer,
        )

        if self.model_type == "ctgan":
            train_kwargs["lr_g"] = lr
            train_kwargs["lr_d"] = lr
        else:
            train_kwargs["lr"] = lr
            if self.model_type in ("ctvae", "diffusion"):
                train_kwargs["early_stopping_patience"] = early_stopping_patience

        loss_history = self._model.fit(**train_kwargs)

        # 6. Compute final epsilon
        n_rows = len(preprocessed_df)
        if dp_trainer is not None:
            epsilon = dp_trainer.current_epsilon(n_rows, batch_size, epochs)
            logger.info("Training complete. Final epsilon=%.4f (delta=%.2e).", epsilon, target_delta)
        else:
            epsilon = float("inf")
            logger.info("Training complete. DP disabled.")

        # 7. Save checkpoint
        checkpoint_path = self._save_checkpoint()

        return {
            "loss_history": loss_history,
            "epsilon": epsilon,
            "checkpoint_path": checkpoint_path,
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_model(self, model_kwargs: Dict[str, Any]):
        """Instantiate the generative model with col_meta and categories."""
        base_kwargs = dict(
            col_meta=self._col_meta,
            categories=self._categories,
        )
        base_kwargs.update(model_kwargs)

        if self.model_type == "ctgan":
            return TabularCTGAN(**base_kwargs)
        elif self.model_type == "ctvae":
            return TabularCTVAE(**base_kwargs)
        elif self.model_type == "diffusion":
            return TabularDiffusion(**base_kwargs)
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _save_checkpoint(self) -> str:
        """Save model to artifacts_root/<dataset_name>/checkpoints/."""
        checkpoint_dir = os.path.join(
            self.artifacts_root, self.dataset_name, "checkpoints"
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, f"{self.model_type}_model.pt")
        self._model.save(checkpoint_path)
        logger.info("Model checkpoint saved to %s", checkpoint_path)
        return checkpoint_path

    # ------------------------------------------------------------------
    # Expose model for downstream use (sampler)
    # ------------------------------------------------------------------

    @property
    def model(self):
        """Return the trained generative model instance."""
        return self._model

    @property
    def col_meta(self) -> List[ColumnMeta]:
        return self._col_meta

    @property
    def categories(self) -> Dict[str, List[str]]:
        return self._categories

    @property
    def col_name_index(self) -> Dict[str, int]:
        return self._col_name_index
