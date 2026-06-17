"""
Unit tests for Compliance Reporter
"""

import os
import pytest
from src.evaluation.report import ComplianceReporter

def test_compliance_reporter_saves_files(tmp_path):
    reporter = ComplianceReporter(dataset_name="test_dataset", output_dir=str(tmp_path))
    
    fidelity = {
        "wasserstein": {"col1": 0.02, "col2": 0.12},
        "js_divergence": {"col3": 0.01},
        "correlation_difference": 0.04,
        "real_corr": {},
        "synth_corr": {},
    }
    
    privacy = {
        "dcr_mean": 1.25,
        "dcr_min": 0.05,
        "dcr_leakage_pct": 0.0,
        "nndr_mean": 0.85,
        "mia_auc": 0.52,
        "aia": {
            "task": "classification",
            "accuracy": 0.85,
            "f1_score": 0.83,
        }
    }
    
    utility = {
        "task": "classification",
        "metrics": {
            "RandomForest": {
                "TRTR": {"f1_macro": 0.82},
                "TSTR": {"f1_macro": 0.79},
            }
        }
    }
    
    plots = {
        "distributions": "distributions_grid.png",
        "correlation": "correlation_comparison.png",
        "dcr": "dcr_distribution.png",
    }
    
    md_path, html_path = reporter.generate_report(
        fidelity, privacy, utility, plots, target_col="target_col", sensitive_col="sensitive_col"
    )
    
    assert os.path.exists(md_path)
    assert os.path.exists(html_path)
    
    # Check contents
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    assert "Compliance Audit Report" in md_text
    assert "test_dataset" in md_text
    assert "RandomForest" in md_text
    assert "target_col" in md_text
    assert "sensitive_col" in md_text
    
    with open(html_path, "r", encoding="utf-8") as f:
        html_text = f.read()
    assert "<!DOCTYPE html>" in html_text
    assert "distributions_grid.png" in html_text
    assert "target_col" in html_text
    assert "sensitive_col" in html_text


def test_compliance_reporter_dynamic_threshold(tmp_path):
    reporter = ComplianceReporter(dataset_name="test_dataset", output_dir=str(tmp_path))
    
    fidelity = {
        "wasserstein": {"col1": 0.02},
        "js_divergence": {"col3": 0.01},
        "correlation_difference": 0.04,
        "real_corr": {},
        "synth_corr": {},
    }
    
    privacy = {
        "dcr_mean": 1.25,
        "dcr_min": 0.05,
        "dcr_leakage_pct": 1.5,
        "dcr_leakage_threshold": 0.0075,
        "nndr_mean": 0.85,
        "mia_auc": 0.52,
        "aia": None
    }
    
    utility = {
        "task": "classification",
        "metrics": {}
    }
    
    plots = {
        "distributions": "distributions_grid.png",
        "correlation": "correlation_comparison.png",
        "dcr": "dcr_distribution.png",
    }
    
    md_path, html_path = reporter.generate_report(
        fidelity, privacy, utility, plots, target_col="target_col"
    )
    
    with open(md_path, "r", encoding="utf-8") as f:
        md_text = f.read()
    assert "DCR Leakage Percentage (<0.0075)" in md_text
    
    with open(html_path, "r", encoding="utf-8") as f:
        html_text = f.read()
    assert "Share of rows with L2 distance &lt; 0.0075" in html_text

