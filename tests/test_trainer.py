"""
Unit tests for ModelTrainer — build_col_meta, tensor conversion, and
lightweight training smoke-tests (no GPU, very small data, 1 epoch).
"""

import numpy as np
import pandas as pd
import pytest

from src.training.trainer import build_col_meta, _df_to_tensor


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def small_preprocessed_df():
    """
    Simulates the output of PreprocessingPipeline.fit_transform()
    for a minimal dataset with 2 continuous + 2 categorical (onehot + label).
    """
    np.random.seed(42)
    n = 50
    df = pd.DataFrame({
        "tenure":         np.random.uniform(0, 1, n).astype(np.float32),   # continuous scaled
        "MonthlyCharges": np.random.uniform(0, 1, n).astype(np.float32),   # continuous scaled
        # One-hot group for 'Contract' (3 cats)
        "Contract_Month-to-month": np.random.randint(0, 2, n).astype(np.float32),
        "Contract_One year":       np.random.randint(0, 2, n).astype(np.float32),
        "Contract_Two year":       np.random.randint(0, 2, n).astype(np.float32),
        # Label-encoded for 'PaymentMethod' (high cardinality)
        "PaymentMethod":  np.random.randint(0, 4, n).astype(np.float32),
    })
    return df


# ---------------------------------------------------------------------------
# Test: build_col_meta
# ---------------------------------------------------------------------------

def test_build_col_meta_continuous(small_preprocessed_df):
    col_meta, categories, col_idx = build_col_meta(
        small_preprocessed_df,
        continuous_cols=["tenure", "MonthlyCharges"],
        categorical_cols=["Contract", "PaymentMethod"],
    )
    # Should have entries for tenure, MonthlyCharges, Contract (onehot), PaymentMethod (label)
    names = [m.name for m in col_meta]
    assert "tenure" in names
    assert "MonthlyCharges" in names


def test_build_col_meta_onehot_detection(small_preprocessed_df):
    col_meta, categories, col_idx = build_col_meta(
        small_preprocessed_df,
        continuous_cols=["tenure", "MonthlyCharges"],
        categorical_cols=["Contract", "PaymentMethod"],
    )
    onehot_entries = [m for m in col_meta if m.col_type == "onehot"]
    assert len(onehot_entries) >= 1  # Contract should be onehot
    # Contract should have dim=3
    contract_meta = next((m for m in onehot_entries if m.name == "Contract"), None)
    assert contract_meta is not None
    assert contract_meta.dim == 3


def test_build_col_meta_label_detection(small_preprocessed_df):
    col_meta, categories, col_idx = build_col_meta(
        small_preprocessed_df,
        continuous_cols=["tenure", "MonthlyCharges"],
        categorical_cols=["Contract", "PaymentMethod"],
    )
    label_entries = [m for m in col_meta if m.col_type == "label"]
    assert any(m.name == "PaymentMethod" for m in label_entries)


def test_build_col_meta_total_dim(small_preprocessed_df):
    col_meta, categories, col_idx = build_col_meta(
        small_preprocessed_df,
        continuous_cols=["tenure", "MonthlyCharges"],
        categorical_cols=["Contract", "PaymentMethod"],
    )
    total_dim = sum(m.dim for m in col_meta)
    # 2 continuous + 3 onehot (Contract) + 1 label (PaymentMethod) = 6
    assert total_dim == 6


# ---------------------------------------------------------------------------
# Test: _df_to_tensor
# ---------------------------------------------------------------------------

def test_df_to_tensor_shape(small_preprocessed_df):
    col_meta, _, _ = build_col_meta(
        small_preprocessed_df,
        continuous_cols=["tenure", "MonthlyCharges"],
        categorical_cols=["Contract", "PaymentMethod"],
    )
    tensor = _df_to_tensor(small_preprocessed_df, col_meta)
    expected_dim = sum(m.dim for m in col_meta)
    assert tensor.shape == (len(small_preprocessed_df), expected_dim)


def test_df_to_tensor_dtype(small_preprocessed_df):
    import torch
    col_meta, _, _ = build_col_meta(
        small_preprocessed_df,
        continuous_cols=["tenure", "MonthlyCharges"],
        categorical_cols=["Contract", "PaymentMethod"],
    )
    tensor = _df_to_tensor(small_preprocessed_df, col_meta)
    assert tensor.dtype == torch.float32


# ---------------------------------------------------------------------------
# Test: ModelTrainer smoke-test (1 epoch, tiny data, no DP)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ["ctvae", "diffusion"])
def test_trainer_smoke(model_type, small_preprocessed_df, tmp_path):
    """Smoke test: trainer instantiates and runs 1 epoch without crashing."""
    pytest.importorskip("torch")

    from src.training.trainer import ModelTrainer

    trainer = ModelTrainer(
        model_type=model_type,
        dataset_name="test_dataset",
        artifacts_root=str(tmp_path),
    )

    result = trainer.train(
        preprocessed_df=small_preprocessed_df,
        continuous_cols=["tenure", "MonthlyCharges"],
        categorical_cols=["Contract", "PaymentMethod"],
        epochs=1,
        batch_size=16,
        lr=1e-3,
        enable_dp=False,
    )

    assert "loss_history" in result
    assert "checkpoint_path" in result
    assert result["epsilon"] == float("inf")  # DP disabled
    # Checkpoint file should exist
    import os
    assert os.path.exists(result["checkpoint_path"])


@pytest.mark.parametrize("model_type", ["ctvae"])
def test_trainer_with_constraints(model_type, small_preprocessed_df, tmp_path):
    """Smoke test with constraint penalty enabled."""
    pytest.importorskip("torch")

    from src.training.trainer import ModelTrainer

    trainer = ModelTrainer(
        model_type=model_type,
        dataset_name="test_dataset",
        artifacts_root=str(tmp_path),
    )

    result = trainer.train(
        preprocessed_df=small_preprocessed_df,
        continuous_cols=["tenure", "MonthlyCharges"],
        categorical_cols=["Contract", "PaymentMethod"],
        epochs=1,
        batch_size=16,
        lr=1e-3,
        enable_dp=False,
        constraint_expressions=["tenure <= MonthlyCharges"],
    )

    assert "loss_history" in result


# ---------------------------------------------------------------------------
# Test: DPTrainer initialisation and epsilon calibration
# ---------------------------------------------------------------------------

def test_dp_trainer_calibration():
    from src.training.dp_training import _calibrate_noise_multiplier

    sigma = _calibrate_noise_multiplier(
        target_epsilon=1.0,
        target_delta=1e-5,
        dataset_size=5000,
        batch_size=256,
        epochs=100,
        max_grad_norm=1.0,
    )
    assert sigma > 0.0


def test_dp_trainer_init():
    from src.training.dp_training import DPTrainer

    trainer = DPTrainer(
        target_epsilon=2.0,
        target_delta=1e-5,
        max_grad_norm=1.0,
        backend="custom",
    )
    assert trainer.target_epsilon == 2.0
    assert trainer.max_grad_norm == 1.0


def test_dp_trainer_invalid_epsilon():
    from src.training.dp_training import DPTrainer
    with pytest.raises(ValueError, match="target_epsilon"):
        DPTrainer(target_epsilon=-1.0)


# ---------------------------------------------------------------------------
# Test: CTGAN and CTGAN + DP Smoke tests
# ---------------------------------------------------------------------------

def test_ctgan_smoke(small_preprocessed_df, tmp_path):
    """Smoke test: CTGAN trainer instantiates and runs 1 epoch without crashing."""
    pytest.importorskip("torch")
    from src.training.trainer import ModelTrainer

    trainer = ModelTrainer(
        model_type="ctgan",
        dataset_name="test_ctgan",
        artifacts_root=str(tmp_path),
    )

    result = trainer.train(
        preprocessed_df=small_preprocessed_df,
        continuous_cols=["tenure", "MonthlyCharges"],
        categorical_cols=["Contract", "PaymentMethod"],
        epochs=1,
        batch_size=16,
        lr=1e-3,
        enable_dp=False,
    )

    assert "loss_history" in result
    assert "checkpoint_path" in result
    assert result["epsilon"] == float("inf")
    import os
    assert os.path.exists(result["checkpoint_path"])


def test_ctgan_dp_smoke(small_preprocessed_df, tmp_path):
    """Smoke test: CTGAN trainer runs 1 epoch with DP-SGD enabled."""
    pytest.importorskip("torch")
    from src.training.trainer import ModelTrainer

    trainer = ModelTrainer(
        model_type="ctgan",
        dataset_name="test_ctgan_dp",
        artifacts_root=str(tmp_path),
    )

    result = trainer.train(
        preprocessed_df=small_preprocessed_df,
        continuous_cols=["tenure", "MonthlyCharges"],
        categorical_cols=["Contract", "PaymentMethod"],
        epochs=1,
        batch_size=16,
        lr=1e-3,
        enable_dp=True,
        target_epsilon=10.0,
        target_delta=1e-3,
        dp_backend="custom",
    )

    assert "loss_history" in result
    assert "checkpoint_path" in result
    assert result["epsilon"] > 0.0
    assert result["epsilon"] < float("inf")
    import os
    assert os.path.exists(result["checkpoint_path"])

