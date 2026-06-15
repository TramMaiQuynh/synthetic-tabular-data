"""
Unit tests for ML Utility Evaluator
"""

import pytest
import pandas as pd
import numpy as np
from src.evaluation.utility import UtilityEvaluator

def test_utility_evaluator_classification():
    # Make a simple binary classification dataset
    np.random.seed(42)
    n = 100
    real_train = pd.DataFrame({
        "feat1": np.random.uniform(0, 1, n),
        "feat2": np.random.choice(["A", "B"], n),
        "target": np.random.choice([0, 1], n)
    })
    real_test = pd.DataFrame({
        "feat1": np.random.uniform(0, 1, n),
        "feat2": np.random.choice(["A", "B"], n),
        "target": np.random.choice([0, 1], n)
    })
    synth = pd.DataFrame({
        "feat1": np.random.uniform(0, 1, n),
        "feat2": np.random.choice(["A", "B"], n),
        "target": np.random.choice([0, 1], n)
    })
    
    evaluator = UtilityEvaluator(
        target_col="target",
        continuous_cols=["feat1"],
        categorical_cols=["feat2"]
    )
    
    results = evaluator.evaluate(real_train, real_test, synth)
    
    assert results["task"] == "classification"
    assert results["target_column"] == "target"
    assert "RandomForest" in results["metrics"]
    assert "LogisticRegression" in results["metrics"]
    
    rf_metrics = results["metrics"]["RandomForest"]
    assert "TRTR" in rf_metrics
    assert "TSTR" in rf_metrics
    assert "accuracy" in rf_metrics["TRTR"]
    assert "f1_macro" in rf_metrics["TRTR"]


def test_utility_evaluator_regression():
    # Make a simple regression dataset
    np.random.seed(42)
    n = 100
    real_train = pd.DataFrame({
        "feat1": np.random.uniform(0, 1, n),
        "feat2": np.random.choice(["A", "B"], n),
        "target": np.random.uniform(10, 50, n)
    })
    real_test = pd.DataFrame({
        "feat1": np.random.uniform(0, 1, n),
        "feat2": np.random.choice(["A", "B"], n),
        "target": np.random.uniform(10, 50, n)
    })
    synth = pd.DataFrame({
        "feat1": np.random.uniform(0, 1, n),
        "feat2": np.random.choice(["A", "B"], n),
        "target": np.random.uniform(10, 50, n)
    })
    
    evaluator = UtilityEvaluator(
        target_col="target",
        continuous_cols=["feat1"],
        categorical_cols=["feat2"]
    )
    
    results = evaluator.evaluate(real_train, real_test, synth)
    
    assert results["task"] == "regression"
    assert "LinearRegression" in results["metrics"]
    
    lr_metrics = results["metrics"]["LinearRegression"]
    assert "mse" in lr_metrics["TRTR"]
    assert "r2" in lr_metrics["TRTR"]
