"""
Unit tests for ConstraintsEngine.
Tests: parse, violation_mask, violation_rate, post_correction, JS divergence.
"""

import numpy as np
import pandas as pd
import pytest

from src.models.constraints import ConstraintsEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_df():
    return pd.DataFrame({
        "age": [25, 17, 30, 15, 40],
        "monthly": [50.0, 80.0, 100.0, 120.0, 60.0],
        "total": [500.0, 200.0, 50.0, 1000.0, 300.0],
    })


# ---------------------------------------------------------------------------
# Test: expression parsing
# ---------------------------------------------------------------------------

def test_parse_scalar_constraint():
    engine = ConstraintsEngine(["age >= 18"])
    assert len(engine._constraints) == 1
    c = engine._constraints[0]
    assert c.lhs_col == "age"
    assert c.op == ">="
    assert c.rhs_scalar == 18.0
    assert c.rhs_col is None


def test_parse_column_constraint():
    engine = ConstraintsEngine(["total >= monthly"])
    c = engine._constraints[0]
    assert c.lhs_col == "total"
    assert c.rhs_col == "monthly"
    assert c.rhs_scalar is None


def test_invalid_expression_raises():
    with pytest.raises(ValueError, match="Cannot parse"):
        ConstraintsEngine(["age IS BETWEEN 18 AND 65"])


# ---------------------------------------------------------------------------
# Test: violation_mask
# ---------------------------------------------------------------------------

def test_violation_mask_scalar(simple_df):
    engine = ConstraintsEngine(["age >= 18"])
    mask = engine.violation_mask(simple_df)
    # Rows with age < 18 should be True (violating)
    expected = pd.Series([False, True, False, True, False])
    pd.testing.assert_series_equal(mask.reset_index(drop=True), expected)


def test_violation_mask_column_reference(simple_df):
    engine = ConstraintsEngine(["total >= monthly"])
    mask = engine.violation_mask(simple_df)
    # total < monthly: row 2 (100 < 50? No), row index 2 = total=50, monthly=100 → violates
    assert mask.iloc[2] is True or mask.iloc[2] == True  # noqa: E712


def test_violation_mask_combined(simple_df):
    engine = ConstraintsEngine(["age >= 18", "total >= monthly"])
    mask = engine.violation_mask(simple_df)
    # Should be OR of both masks
    m1 = ConstraintsEngine(["age >= 18"]).violation_mask(simple_df)
    m2 = ConstraintsEngine(["total >= monthly"]).violation_mask(simple_df)
    pd.testing.assert_series_equal(mask.reset_index(drop=True), (m1 | m2).reset_index(drop=True))


def test_no_constraints_no_violations():
    engine = ConstraintsEngine([])
    df = pd.DataFrame({"x": [1, 2, 3]})
    assert engine.violation_rate(df) == 0.0


# ---------------------------------------------------------------------------
# Test: violation_rate
# ---------------------------------------------------------------------------

def test_violation_rate_scalar(simple_df):
    engine = ConstraintsEngine(["age >= 18"])
    rate = engine.violation_rate(simple_df)
    # 2 out of 5 rows violate
    assert abs(rate - 2 / 5) < 1e-9


# ---------------------------------------------------------------------------
# Test: post_correction — scalar clamp
# ---------------------------------------------------------------------------

def test_post_correction_scalar_clamp():
    df = pd.DataFrame({"age": [25, 10, 30, 5, 40], "income": [50000.0] * 5})
    engine = ConstraintsEngine(["age >= 18"], max_retries=3)
    corrected, stats = engine.post_correction(df)
    assert stats["violation_rate_after"] == 0.0
    assert (corrected["age"] >= 18).all()


def test_post_correction_column_clamp():
    df = pd.DataFrame({
        "total": [100.0, 30.0, 200.0],
        "monthly": [80.0, 50.0, 150.0],
    })
    # total >= monthly
    engine = ConstraintsEngine(["total >= monthly"], max_retries=3)
    corrected, stats = engine.post_correction(df)
    # Row 1: total=30, monthly=50 → should be clamped to total >= monthly
    assert corrected["total"].iloc[1] >= corrected["monthly"].iloc[1]
    assert stats["violation_rate_after"] == 0.0


def test_post_correction_returns_stats_keys():
    df = pd.DataFrame({"age": [20, 25]})
    engine = ConstraintsEngine(["age >= 18"])
    _, stats = engine.post_correction(df)
    assert "violation_rate_before" in stats
    assert "violation_rate_after" in stats
    assert "js_divergences" in stats
    assert "n_total" in stats


# ---------------------------------------------------------------------------
# Test: JS divergence — Over-Correction warning
# ---------------------------------------------------------------------------

def test_js_divergence_identical_data():
    arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    jsd = ConstraintsEngine._js_divergence_1d(arr, arr)
    assert jsd < 1e-6  # identical distributions should have near-zero JSD


def test_js_divergence_different_data():
    arr_a = np.arange(1, 101, dtype=float)
    arr_b = np.arange(1000, 1100, dtype=float)
    jsd = ConstraintsEngine._js_divergence_1d(arr_a, arr_b)
    # Non-overlapping distributions should have high JSD
    assert jsd > 0.3


# ---------------------------------------------------------------------------
# Test: soft_loss_penalty (no torch dependency at test time with fallback)
# ---------------------------------------------------------------------------

def test_soft_penalty_returns_numeric():
    engine = ConstraintsEngine(["total >= monthly"])
    try:
        import torch
        tensor_dict = {
            "total": torch.tensor([100.0, 30.0, 200.0]),
            "monthly": torch.tensor([80.0, 50.0, 150.0]),
        }
        penalty = engine.soft_loss_penalty(tensor_dict)
        assert float(penalty) >= 0.0
    except ImportError:
        # torch not available: penalty should be 0.0 or numeric
        tensor_dict = {"total": [100.0], "monthly": [80.0]}
        penalty = engine.soft_loss_penalty(tensor_dict)
        assert penalty == 0.0


# ---------------------------------------------------------------------------
# Test: missing columns handled gracefully
# ---------------------------------------------------------------------------

def test_violation_mask_missing_col():
    engine = ConstraintsEngine(["nonexistent_col >= 0"])
    df = pd.DataFrame({"age": [20, 25]})
    mask = engine.violation_mask(df)
    # Missing column → no violation flagged (graceful skip)
    assert not mask.any()


def test_repr():
    engine = ConstraintsEngine(["age >= 18", "total >= monthly"], max_retries=3)
    r = repr(engine)
    assert "ConstraintsEngine" in r
    assert "2" in r
