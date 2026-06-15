"""
Unit tests for Fidelity Assessor
"""

import pytest
import pandas as pd
import numpy as np
from src.evaluation.fidelity import (
    compute_wasserstein_distance,
    compute_js_divergence,
    compute_cramers_v,
    compute_correlation_ratio,
    FidelityAssessor,
)

def test_compute_wasserstein_distance():
    r = pd.Series([1.0, 2.0, 3.0])
    s1 = pd.Series([1.0, 2.0, 3.0]) # identical
    s2 = pd.Series([2.0, 3.0, 4.0]) # shifted by 1 unit
    
    assert compute_wasserstein_distance(r, s1) == 0.0
    assert compute_wasserstein_distance(r, s2) > 0.0
    # After MinMax normalization: r -> [0, 0.5, 1], s2 -> [0.5, 1.0, 1.5]
    # Wasserstein distance = mean absolute shift = 0.5
    assert np.isclose(compute_wasserstein_distance(r, s2), 0.5)


def test_compute_js_divergence():
    r = pd.Series(["a", "a", "b"])
    s1 = pd.Series(["a", "a", "b"]) # identical
    s2 = pd.Series(["b", "b", "a"]) # different frequencies
    
    assert np.isclose(compute_js_divergence(r, s1), 0.0)
    assert compute_js_divergence(r, s2) > 0.0
    assert compute_js_divergence(r, s2) <= 1.0


def test_compute_cramers_v():
    x = pd.Series(["yes", "yes", "no", "no"])
    y = pd.Series(["high", "high", "low", "low"]) # perfect correlation
    
    assert np.isclose(compute_cramers_v(x, y), 1.0)
    
    x = pd.Series(["yes", "no", "yes", "no"])
    y = pd.Series(["high", "high", "low", "low"]) # no correlation
    assert np.isclose(compute_cramers_v(x, y), 0.0)


def test_compute_correlation_ratio():
    cat = pd.Series(["a", "a", "b", "b"])
    meas = pd.Series([1.0, 1.0, 10.0, 10.0]) # perfect association
    assert np.isclose(compute_correlation_ratio(cat, meas), 1.0)
    
    cat = pd.Series(["a", "b", "a", "b"])
    meas = pd.Series([5.0, 5.0, 5.0, 5.0]) # constant, zero variance
    assert np.isclose(compute_correlation_ratio(cat, meas), 0.0)


def test_fidelity_assessor_evaluate():
    real = pd.DataFrame({
        "cont": [1.0, 2.0, 3.0, 4.0, 5.0],
        "cat": ["a", "b", "a", "b", "a"]
    })
    synth = pd.DataFrame({
        "cont": [1.1, 1.9, 3.1, 3.9, 5.0],
        "cat": ["a", "b", "a", "b", "b"]
    })
    
    assessor = FidelityAssessor(continuous_cols=["cont"], categorical_cols=["cat"])
    results = assessor.evaluate(real, synth)
    
    assert "wasserstein" in results
    assert "js_divergence" in results
    assert "correlation_difference" in results
    assert "real_corr" in results
    assert "synth_corr" in results
    
    assert "cont" in results["wasserstein"]
    assert "cat" in results["js_divergence"]
    assert results["wasserstein"]["cont"] > 0.0
    assert results["js_divergence"]["cat"] >= 0.0
    assert results["correlation_difference"] >= 0.0
