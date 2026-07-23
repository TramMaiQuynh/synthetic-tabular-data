"""
Unit Tests for Statistical Tests Module (Friedman, Nemenyi & CD Diagram)
"""

import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.evaluation.statistical_tests import (
    run_statistical_evaluation,
    plot_nemenyi_heatmap,
    compute_critical_difference,
    plot_cd_diagram,
)


@pytest.fixture
def significant_metrics_df():
    """Create synthetic metrics dataframe where models have clear performance differences."""
    rng = np.random.RandomState(42)
    n_folds = 10
    # CTGAN (poor), CTVAE (medium), Diffusion (best)
    ctgan_scores = rng.normal(loc=0.50, scale=0.02, size=n_folds)
    ctvae_scores = rng.normal(loc=0.70, scale=0.02, size=n_folds)
    diff_scores = rng.normal(loc=0.90, scale=0.02, size=n_folds)

    return pd.DataFrame(
        {
            "CTGAN": ctgan_scores,
            "CTVAE": ctvae_scores,
            "Diffusion": diff_scores,
        },
        index=[f"Fold_{i}" for i in range(n_folds)],
    )


@pytest.fixture
def non_significant_metrics_df():
    """Create synthetic metrics dataframe with balanced ranks (no model dominates)."""
    return pd.DataFrame(
        {
            "ModelA": [0.50, 0.55, 0.45, 0.50, 0.55],
            "ModelB": [0.55, 0.45, 0.50, 0.55, 0.45],
            "ModelC": [0.45, 0.50, 0.55, 0.45, 0.50],
        },
        index=[f"Fold_{i}" for i in range(5)],
    )


def test_run_statistical_evaluation_significant(significant_metrics_df):
    res = run_statistical_evaluation(significant_metrics_df, alpha=0.05)

    assert "friedman_stat" in res
    assert "friedman_p_value" in res
    assert res["is_significant"] == True
    assert res["nemenyi_p_values"] is not None
    assert isinstance(res["nemenyi_p_values"], pd.DataFrame)
    assert res["nemenyi_p_values"].shape == (3, 3)


def test_run_statistical_evaluation_non_significant(non_significant_metrics_df):
    res = run_statistical_evaluation(non_significant_metrics_df, alpha=0.05)

    assert res["is_significant"] == False
    assert res["nemenyi_p_values"] is None


def test_input_validation_models_count():
    df_two_models = pd.DataFrame(
        {"M1": [1, 2, 3], "M2": [2, 3, 4]},
        index=["F1", "F2", "F3"],
    )
    with pytest.raises(ValueError, match="yêu cầu ít nhất 3 mô hình"):
        run_statistical_evaluation(df_two_models)


def test_input_validation_rows_count():
    df_few_rows = pd.DataFrame(
        {
            "M1": [1.0, 2.0],
            "M2": [2.0, 3.0],
            "M3": [3.0, 4.0],
        },
        index=["F1", "F2"],
    )
    with pytest.raises(ValueError, match="yêu cầu ít nhất 3 blocks/folds"):
        run_statistical_evaluation(df_few_rows)


def test_plot_nemenyi_heatmap(significant_metrics_df):
    res = run_statistical_evaluation(significant_metrics_df, alpha=0.05)
    fig = plot_nemenyi_heatmap(
        res["nemenyi_p_values"],
        alpha=0.05,
        metric_name="Fidelity Score",
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_nemenyi_heatmap_none():
    fig = plot_nemenyi_heatmap(None)
    assert fig is None


def test_compute_critical_difference():
    # k=3 models, n=10 blocks, alpha=0.05
    cd = compute_critical_difference(n_models=3, n_blocks=10, alpha=0.05)
    assert cd > 0.0
    # Expected CD for k=3, n=10 is q_0.05 * sqrt(12/60) = 2.343 * 0.4472 = ~1.0478
    assert 0.95 <= cd <= 1.15


def test_plot_cd_diagram(significant_metrics_df):
    fig = plot_cd_diagram(
        significant_metrics_df,
        alpha=0.05,
        metric_name="Fidelity Score",
        higher_is_better=True,
    )

    assert fig is not None
    assert isinstance(fig, plt.Figure)
    plt.close(fig)


def test_plot_cd_diagram_invalid_data():
    fig = plot_cd_diagram(None)
    assert fig is None

    df_single_col = pd.DataFrame({"M1": [1, 2, 3]})
    fig_single = plot_cd_diagram(df_single_col)
    assert fig_single is None
