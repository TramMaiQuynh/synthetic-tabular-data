"""
Unit tests for Privacy Auditor
"""

import pytest
import pandas as pd
import numpy as np
from src.evaluation.privacy import compute_dcr_nndr, PrivacyAuditor

def test_compute_dcr_nndr():
    # Identical datasets
    real = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    synth = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    
    dcr, nndr = compute_dcr_nndr(real, synth)
    # The closest record to each synthetic point is itself (distance 0)
    assert np.allclose(dcr, [0.0, 0.0])
    # Distance to second closest is distance to the other point, i.e. sqrt(1^2 + 1^2) = sqrt(2)
    # So NNDR = 0 / sqrt(2) = 0
    assert np.allclose(nndr, [0.0, 0.0])


def test_privacy_auditor_evaluate():
    real_train = pd.DataFrame({
        "cont": [1.0, 2.0, 3.0, 4.0, 5.0],
        "cat": ["a", "b", "a", "b", "a"]
    })
    real_test = pd.DataFrame({
        "cont": [1.5, 2.5, 3.5, 4.5, 5.5],
        "cat": ["b", "a", "b", "a", "b"]
    })
    synth = pd.DataFrame({
        "cont": [1.0, 2.0, 3.0, 4.0, 5.0],
        "cat": ["a", "b", "a", "b", "a"]
    })
    
    # Custom pipeline loader mapping categorical to one-hot manually
    def mock_pipeline_loader(df):
        out = np.zeros((len(df), 2), dtype=np.float32)
        out[:, 0] = df["cont"].values
        out[:, 1] = (df["cat"] == "a").astype(np.float32)
        return out
        
    auditor = PrivacyAuditor(
        continuous_cols=["cont"],
        categorical_cols=["cat"],
        sensitive_col="cat",
    )
    
    results = auditor.evaluate(
        real_train, real_test, synth, mock_pipeline_loader
    )
    
    assert "dcr_vals" in results
    assert "nndr_vals" in results
    assert "dcr_mean" in results
    assert "dcr_min" in results
    assert "nndr_mean" in results
    assert "nndr_min" in results
    assert "dcr_leakage_pct" in results
    assert "mia_auc" in results
    assert "aia" in results
    
    assert results["dcr_leakage_pct"] > 0.0 # because synthetic is identical to training set
    assert results["mia_auc"] >= 0.0
    assert "accuracy" in results["aia"]
