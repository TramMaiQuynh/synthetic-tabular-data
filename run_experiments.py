"""
Batch Experiment Runner — Scientific Model Comparison with K-Fold CV + Statistical Tests
=========================================================================================
Runs all experiment groups using the K-Fold Cross-Validation pipeline, aggregates
per-fold metrics across all experiments, and performs Friedman + Nemenyi statistical
tests to determine if performance differences between generative models are
statistically significant.

Experiment Groups
-----------------
A: Architecture Comparison (telco_customer_churn)
   — CTGAN / CTVAE / Diffusion (no DP, K-Fold CV + HPO)
   Purpose: Isolate model architecture effect while controlling for dataset,
            folds, HPO budget, and seed.
   Output:  Per-fold metrics + Friedman/Nemenyi on (models × folds) matrix.

B: Privacy-Utility Trade-off (telco_customer_churn, best architecture = CTVAE)
   — CTVAE at epsilon={inf, 10, 3, 1.5}  (K-Fold CV + HPO)
   Purpose: Isolate privacy budget effect on the best architecture.
   NOTE: B1-NoDP (eps=inf) is identical to A1-CTVAE. Reuse A1 result.

C: Generalization Check (adult_income, bank_marketing)
   — CTGAN / CTVAE / Diffusion on adult_income (K-Fold CV + HPO)
   — CTGAN / CTVAE / Diffusion on bank_marketing (K-Fold CV + HPO)
   Purpose: Validate architecture ranking across different domains.

Usage
-----
    python run_experiments.py              # run all groups
    python run_experiments.py --groups A B  # run only groups A and B
    python run_experiments.py --dry-run     # print plan without running
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)-8s] %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

PYTHON = sys.executable


# ---------------------------------------------------------------------------
# Constants: experiment control variables
# ---------------------------------------------------------------------------
N_SPLITS = 5           # K-Fold: number of folds
N_TRIALS = 20          # HPO: number of Optuna trials per fold
BASE_SEED = 42         # Base random seed (each fold uses seed + fold_number)

DEFAULT_DATA_PATHS = {
    "telco_customer_churn": "data/Telco-Customer-Churn.csv",
    "adult_income":         "data/adult/adult.data",
    "bank_marketing":       "data/bank+marketing/bank/bank.csv",
}


# ---------------------------------------------------------------------------
# Experiment definition
# ---------------------------------------------------------------------------
@dataclass
class Experiment:
    """Single experimental condition (model + dataset + DP config)."""

    group: str
    name: str
    dataset: str
    model_type: str               # "ctgan", "ctvae", "diffusion"
    enable_dp: bool
    epsilon: Optional[float]      # None when DP disabled
    n_splits: int = N_SPLITS      # K-Fold splits
    n_trials: int = N_TRIALS      # HPO trials per fold
    seed: int = BASE_SEED

    # Filled after the run — per-fold metrics
    per_fold_metrics: List[Dict[str, float]] = None  # type: ignore[assignment]

    # Aggregated metrics (mean ± std across folds)
    avg_js: float = float("nan")
    std_js: float = float("nan")
    avg_mia_auc: float = float("nan")
    std_mia_auc: float = float("nan")
    avg_tstr_rf_f1: float = float("nan")
    std_tstr_rf_f1: float = float("nan")

    wall_time_sec: float = float("nan")
    status: str = "PENDING"


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
EXPERIMENTS: List[Experiment] = [
    # -----------------------------------------------------------------------
    # Group A: Architecture Comparison on Telco (no DP)
    #   Control: dataset, n_splits, n_trials, seed
    #   Independent: model_type
    # -----------------------------------------------------------------------
    Experiment("A", "A1-CTVAE",     "telco_customer_churn", "ctvae",    False, None),
    Experiment("A", "A2-CTGAN",     "telco_customer_churn", "ctgan",    False, None),
    Experiment("A", "A3-Diffusion", "telco_customer_churn", "diffusion", False, None),

    # -----------------------------------------------------------------------
    # Group B: Privacy-Utility Trade-off (CTVAE on Telco)
    #   Control: dataset, model_type, n_splits, n_trials, seed
    #   Independent: epsilon (privacy budget)
    #   NOTE: eps=inf (no DP) = A1-CTVAE, reused below
    # -----------------------------------------------------------------------
    Experiment("B", "B2-DP-e10",  "telco_customer_churn", "ctvae", True,  10.0),
    Experiment("B", "B3-DP-e3",   "telco_customer_churn", "ctvae", True,   3.0),
    Experiment("B", "B4-DP-e1.5", "telco_customer_churn", "ctvae", True,   1.5),

    # -----------------------------------------------------------------------
    # Group C: Generalization Check on Adult
    # -----------------------------------------------------------------------
    Experiment("C", "C1-Adult-CTVAE",     "adult_income", "ctvae",    False, None),
    Experiment("C", "C2-Adult-CTGAN",     "adult_income", "ctgan",    False, None),
    Experiment("C", "C3-Adult-Diffusion", "adult_income", "diffusion", False, None),

    # Group D: Generalization Check on Bank Marketing
    Experiment("D", "D1-Bank-CTVAE",     "bank_marketing", "ctvae",    False, None),
    Experiment("D", "D2-Bank-CTGAN",     "bank_marketing", "ctgan",    False, None),
    Experiment("D", "D3-Bank-Diffusion", "bank_marketing", "diffusion", False, None),
]

# Map group letter to description for reporting
GROUP_LABELS = {
    "A": "Architecture Comparison (Telco)",
    "B": "Privacy-Utility Trade-off (CTVAE, Telco)",
    "C": "Generalization Check (Adult)",
    "D": "Generalization Check (Bank Marketing)",
}


# ---------------------------------------------------------------------------
# Helpers: command building
# ---------------------------------------------------------------------------

def _build_cmd(exp: Experiment) -> List[str]:
    """Build the CLI command for run_pipeline.py with K-Fold arguments."""
    cmd = [
        PYTHON, "run_pipeline.py",
        "--dataset",    exp.dataset,
        "--model-type", exp.model_type,
        "--n-splits",   str(exp.n_splits),
        "--n-trials",   str(exp.n_trials),
        "--seed",       str(exp.seed),
    ]
    if exp.enable_dp:
        cmd.append("--enable-dp")

    if exp.dataset in DEFAULT_DATA_PATHS:
        cmd += ["--data-path", DEFAULT_DATA_PATHS[exp.dataset]]

    return cmd


# ---------------------------------------------------------------------------
# Helpers: result parsing from K-Fold outputs
# ---------------------------------------------------------------------------

def _get_kfold_metrics_path(exp: Experiment) -> str:
    """Resolve path to the per-fold metrics CSV written by run_pipeline.py."""
    dp_suffix = f"dp_eps{exp.epsilon}" if exp.enable_dp else "nodp"
    return os.path.join(
        "artifacts", exp.dataset,
        f"{exp.model_type}_{dp_suffix}",
        "kfold_per_fold_metrics.csv",
    )


def _get_kfold_json_path(exp: Experiment) -> str:
    """Resolve path to the aggregated K-Fold results JSON."""
    dp_suffix = f"dp_eps{exp.epsilon}" if exp.enable_dp else "nodp"
    return os.path.join(
        "artifacts", exp.dataset,
        f"{exp.model_type}_{dp_suffix}",
        "kfold_results.json",
    )


def _parse_kfold_results(exp: Experiment) -> None:
    """
    Parse per-fold metrics CSV and aggregated JSON after a K-Fold run.

    The K-Fold pipeline (run_pipeline.py) writes:
      - kfold_per_fold_metrics.csv: one row per fold
      - kfold_results.json: aggregated mean + std per metric
    """
    csv_path = _get_kfold_metrics_path(exp)
    json_path = _get_kfold_json_path(exp)

    if not os.path.exists(csv_path):
        logger.error("Per-fold metrics not found: %s", csv_path)
        return

    # Read per-fold metrics
    df = pd.read_csv(csv_path, index_col="Fold")

    # Convert to list of dicts
    folds = []
    for fold_label in df.index:
        row = df.loc[fold_label].to_dict()
        folds.append(row)
    exp.per_fold_metrics = folds

    # Read aggregated JSON (for summary reporting)
    if os.path.exists(json_path):
        with open(json_path, "r", encoding="utf-8") as f:
            agg = json.load(f)

        def _get_agg(key: str) -> float:
            return agg.get(key, {}).get("mean", float("nan"))

        def _get_std(key: str) -> float:
            return agg.get(key, {}).get("std", float("nan"))

        exp.avg_js = _get_agg("avg_js")
        exp.std_js = _get_std("avg_js")
        exp.avg_mia_auc = _get_agg("mia_auc")
        exp.std_mia_auc = _get_std("mia_auc")
        exp.avg_tstr_rf_f1 = _get_agg("tstr_rf_f1")
        exp.std_tstr_rf_f1 = _get_std("tstr_rf_f1")
    else:
        logger.warning("Aggregated JSON not found: %s", json_path)


# ---------------------------------------------------------------------------
# Helpers: summary output
# ---------------------------------------------------------------------------

def _fmt(val: float, decimals: int = 4) -> str:
    if math.isnan(val):
        return "N/A"
    return f"{val:.{decimals}f}"


def _save_experiment_results(
    experiments: List[Experiment], out_path: str
) -> str:
    """
    Save all experiment results (including per-fold metrics) to a CSV.

    Each row represents one fold of one experiment, enabling downstream
    statistical tests (Friedman, Nemenyi) that require per-block observations.

    Returns:
        Absolute path to the saved CSV.
    """
    rows = []
    for e in experiments:
        if e.per_fold_metrics is None:
            # Experiment did not produce fold-level data (e.g., FAILED)
            continue
        for fold_idx, fold_data in enumerate(e.per_fold_metrics, start=1):
            row = {
                "group":        e.group,
                "experiment":   e.name,
                "dataset":      e.dataset,
                "model_type":   e.model_type,
                "enable_dp":    e.enable_dp,
                "epsilon":      e.epsilon if e.enable_dp else float("inf"),
                "fold":         fold_idx,
                "seed":         e.seed + fold_idx,
            }
            row.update(fold_data)
            rows.append(row)

    if not rows:
        logger.warning("No per-fold results to save.")
        return ""

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    logger.info("All per-fold experiment results saved to: %s", out_path)
    return os.path.abspath(out_path)


def _print_summary(experiments: List[Experiment]) -> None:
    """Print a formatted table of aggregated experiment results."""
    width = 140
    print("\n" + "=" * width)
    print(f"{'EXPERIMENT RESULTS SUMMARY (Mean ± Std across folds)':^{width}}")
    print("=" * width)
    header = "{:<18} {:<12} {:<10} {:<8} {:<20} {:<20} {:<20} {:<10}"
    print(header.format(
        "Experiment", "Model", "Dataset", "DP/Eps",
        "AvgJSD (↓)", "MIA AUC (↓)", "TSTR RF F1 (↑)", "Status",
    ))
    print("-" * width)

    for e in experiments:
        dp_str = f"eps={e.epsilon}" if e.enable_dp else "No DP"
        js_str = f"{_fmt(e.avg_js)} ± {_fmt(e.std_js)}"
        mia_str = f"{_fmt(e.avg_mia_auc)} ± {_fmt(e.std_mia_auc)}"
        tstr_str = f"{_fmt(e.avg_tstr_rf_f1)} ± {_fmt(e.std_tstr_rf_f1)}"

        print(header.format(
            e.name[:16],
            e.model_type.upper(),
            e.dataset[:8],
            dp_str[:7],
            js_str,
            mia_str,
            tstr_str,
            e.status,
        ))

    print("=" * width)
    print("  Key: ↓ = lower is better  |  ↑ = higher is better")
    print()


def _run_statistical_tests(
    experiments: List[Experiment],
    output_dir: str,
    alpha: float = 0.05,
) -> None:
    """
    Run Friedman + Nemenyi statistical tests on grouped experiments.

    For each group of experiments that share the same dataset and DP config,
    this function constructs a (folds × models) matrix for each metric and
    runs the Friedman test. If significant, Nemenyi post-hoc is applied.

    Args:
        experiments: List of completed experiments.
        output_dir: Directory for saving statistical test reports.
        alpha: Significance level (default: 0.05).
    """
    try:
        from src.evaluation.statistical_tests import (
            run_statistical_evaluation,
            plot_nemenyi_heatmap,
            plot_cd_diagram,
        )
    except ImportError:
        logger.warning(
            "src.evaluation.statistical_tests not available. "
            "Skipping statistical tests."
        )
        return

    os.makedirs(output_dir, exist_ok=True)

    # Cross-group references: some statistical tests need experiments from
    # other groups. For example, Group B (Privacy-Utility Trade-off) needs
    # the no-DP baseline (A1-CTVAE) to show utility degradation from DP.
    CROSS_GROUP_REFERENCES = {
        "B": ["A1-CTVAE"],  # Group B reuses A1 as no-DP baseline
    }

    # Build lookup dict for cross-references (by experiment name)
    exp_by_name: Dict[str, Experiment] = {
        e.name: e for e in experiments
        if e.per_fold_metrics is not None and len(e.per_fold_metrics) > 0
    }

    # Group experiments by group letter, resolving cross-references
    groups: Dict[str, List[Experiment]] = {}
    for e in experiments:
        if e.per_fold_metrics is None or len(e.per_fold_metrics) == 0:
            continue
        if e.group not in groups:
            groups[e.group] = []
        groups[e.group].append(e)

    # Resolve cross-group references
    for group_letter, ref_names in CROSS_GROUP_REFERENCES.items():
        for ref_name in ref_names:
            if ref_name in exp_by_name:
                ref_exp = exp_by_name[ref_name]
                if ref_exp.group != group_letter and ref_exp not in groups.get(group_letter, []):
                    # Add referenced experiment to this group for statistical testing
                    if group_letter not in groups:
                        groups[group_letter] = []
                    if ref_exp not in groups[group_letter]:
                        groups[group_letter].append(ref_exp)
                        logger.info(
                            "Cross-group reference: added '%s' (group %s) "
                            "to group %s for statistical tests.",
                            ref_name, ref_exp.group, group_letter,
                        )

    for group_letter, group_exps in sorted(groups.items()):
        if len(group_exps) < 3:
            logger.info(
                "Group %s: only %d models (need ≥3 for Friedman). Skipping.",
                group_letter, len(group_exps),
            )
            continue

        label = GROUP_LABELS.get(group_letter, f"Group {group_letter}")
        n_folds = group_exps[0].n_splits
        model_names = [e.name for e in group_exps]

        # Verify all experiments have the same number of folds
        for e in group_exps:
            if len(e.per_fold_metrics) != n_folds:
                logger.warning(
                    "Group %s: %s has %d folds, expected %d. Skipping group.",
                    group_letter, e.name, len(e.per_fold_metrics), n_folds,
                )
                continue

        logger.info("=" * 60)
        logger.info("Statistical Tests — %s (%s)", label, group_letter)
        logger.info("  Models: %s", model_names)
        logger.info("  Folds : %d", n_folds)

        # Metrics to test (each builds a folds × models matrix)
        test_metrics = [
            ("avg_js", "Average JSD (Fidelity)", False),       # lower = better
            ("mia_auc", "MIA AUC (Privacy)", False),            # lower = better
            ("tstr_rf_f1", "TSTR RandomForest F1 (Utility)", True),  # higher = better
        ]

        for metric_key, metric_label, higher_is_better in test_metrics:
            # Build the folds × models DataFrame
            matrix_data: Dict[str, List[float]] = {}
            for e in group_exps:
                values = [
                    fold.get(metric_key, float("nan"))
                    for fold in e.per_fold_metrics
                ]
                # Filter out NaN
                clean_vals = [v for v in values if not math.isnan(v)]
                if len(clean_vals) == n_folds:
                    matrix_data[e.name] = clean_vals

            if len(matrix_data) < 3:
                logger.info(
                    "  Skipping %s: only %d models with valid data (need ≥3).",
                    metric_label, len(matrix_data),
                )
                continue

            df_metrics = pd.DataFrame(
                matrix_data,
                index=[f"Fold_{i+1}" for i in range(n_folds)],
            )

            logger.info("  Metric: %s", metric_label)
            logger.info("  Matrix shape: %s", df_metrics.shape)

            # Run Friedman + Nemenyi
            result = run_statistical_evaluation(df_metrics, alpha=alpha)

            # Save heatmap if Nemenyi was computed
            if result["nemenyi_p_values"] is not None:
                heatmap_path = os.path.join(
                    output_dir,
                    f"group_{group_letter}_{metric_key}_nemenyi_heatmap.png",
                )
                fig_h = plot_nemenyi_heatmap(
                    result["nemenyi_p_values"],
                    alpha=alpha,
                    save_path=heatmap_path,
                    metric_name=f"{label} — {metric_label}",
                )
                if fig_h is not None:
                    import matplotlib.pyplot as plt
                    plt.close(fig_h)

                cd_path = os.path.join(
                    output_dir,
                    f"group_{group_letter}_{metric_key}_cd_diagram.png",
                )
                fig_cd = plot_cd_diagram(
                    df_metrics,
                    alpha=alpha,
                    save_path=cd_path,
                    metric_name=f"{label} — {metric_label}",
                    higher_is_better=higher_is_better,
                )
                if fig_cd is not None:
                    import matplotlib.pyplot as plt
                    plt.close(fig_cd)

            # Log conclusion
            if result["is_significant"]:
                logger.info(
                    "  ✅ Friedman: p=%.4e < α=%.2f → significant! "
                    "Models differ statistically.",
                    result["friedman_p_value"], alpha,
                )
            else:
                logger.info(
                    "  ❌ Friedman: p=%.4e ≥ α=%.2f → NOT significant. "
                    "No evidence of difference.",
                    result["friedman_p_value"], alpha,
                )

    logger.info("Statistical tests complete.")


def _print_statistical_summary(
    experiments: List[Experiment],
    alpha: float = 0.05,
) -> None:
    """
    Print a concise summary of statistical test outcomes per group.
    """
    groups: Dict[str, List[Experiment]] = {}
    for e in experiments:
        if e.per_fold_metrics is None or len(e.per_fold_metrics) == 0:
            continue
        if e.group not in groups:
            groups[e.group] = []
        groups[e.group].append(e)

    print("\n" + "=" * 60)
    print("STATISTICAL SIGNIFICANCE SUMMARY (Friedman test, α={})".format(alpha))
    print("=" * 60)

    for group_letter in sorted(groups):
        group_exps = groups[group_letter]
        if len(group_exps) < 3:
            continue

        label = GROUP_LABELS.get(group_letter, f"Group {group_letter}")
        model_names = [e.name for e in group_exps]

        print(f"\n  Group {group_letter}: {label}")
        print(f"    Models: {', '.join(model_names)}")

        for metric_key, metric_label in [
            ("avg_js", "JSD"),
            ("mia_auc", "MIA AUC"),
            ("tstr_rf_f1", "TSTR RF F1"),
        ]:
            matrix_data: Dict[str, List[float]] = {}
            for e in group_exps:
                values = [
                    fold.get(metric_key, float("nan"))
                    for fold in e.per_fold_metrics
                ]
                clean = [v for v in values if not math.isnan(v)]
                if len(clean) == e.n_splits:
                    matrix_data[e.name] = clean

            if len(matrix_data) >= 3:
                from scipy.stats import friedmanchisquare
                arrays = [np.array(v) for v in matrix_data.values()]
                try:
                    _, p_val = friedmanchisquare(*arrays)
                    sig = "SIGNIFICANT" if p_val < alpha else "not significant"
                    print(
                        f"    {metric_label:<12}: p = {p_val:.4e} "
                        f"({sig})"
                    )
                except Exception:
                    print(f"    {metric_label:<12}: test failed")

    print("=" * 60)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scientific Batch Experiment Runner with K-Fold CV + Statistical Tests"
    )
    parser.add_argument(
        "--groups", nargs="+", default=["A", "B", "C", "D"],
        help="Experiment groups to run (A, B, C, D). Default: all",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan without executing any training",
    )
    parser.add_argument(
        "--no-stats", action="store_true",
        help="Skip statistical tests (only run experiments).",
    )
    parser.add_argument(
        "--alpha", type=float, default=0.05,
        help="Significance level for statistical tests (default: 0.05).",
    )
    parser.add_argument(
        "--out", type=str,
        default=os.path.join("results", "experiment_results.csv"),
        help="Path to output CSV (default: results/experiment_results.csv)",
    )
    args = parser.parse_args()

    selected = [e for e in EXPERIMENTS if e.group in args.groups]

    # Print experiment plan
    print("\n" + "=" * 80)
    print(f"  K-FOLD BATCH EXPERIMENT PLAN  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Groups selected : {args.groups}")
    print(f"  Total runs      : {len(selected)}")
    print(f"  Folds per run   : {N_SPLITS}  |  HPO trials per fold: {N_TRIALS}")
    print(f"  Base seed       : {BASE_SEED}  |  Alpha: {args.alpha}")
    print(f"  Output CSV      : {os.path.abspath(args.out)}")
    print("=" * 80)

    print("\n  Run plan:")
    for i, exp in enumerate(selected, 1):
        dp_label = f"DP eps={exp.epsilon}" if exp.enable_dp else "No DP"
        print(
            f"    [{i:02d}] {exp.name:<18} "
            f"{exp.model_type.upper():<12} "
            f"{exp.dataset:<25} "
            f"{dp_label:<20} "
            f"{exp.n_splits}-fold"
        )
    print()

    if args.dry_run:
        logger.info("Dry-run mode — no experiments will be executed.")
        return

    # ------------------------------------------------------------------
    # Execute experiments
    # ------------------------------------------------------------------
    for i, exp in enumerate(selected, 1):
        dp_label = f"eps={exp.epsilon}" if exp.enable_dp else "no DP"
        print(f"\n{'='*80}")
        print(
            f"  [{i:02d}/{len(selected):02d}]  {exp.name}  ->  "
            f"{exp.model_type.upper()} | {dp_label} | {exp.n_splits}-fold CV"
        )
        print(f"{'='*80}")

        cmd = _build_cmd(exp)
        logger.info("CMD: %s", " ".join(cmd))

        env = os.environ.copy()
        if exp.enable_dp and exp.epsilon is not None:
            env["SYNTH_OVERRIDE_EPSILON"] = str(exp.epsilon)

        t0 = time.time()
        try:
            subprocess.run(cmd, env=env, check=True, text=True)
            exp.status = "OK"
        except subprocess.CalledProcessError as exc:
            logger.error("Experiment %s FAILED (exit code %d)", exp.name, exc.returncode)
            exp.status = "FAILED"
        except FileNotFoundError:
            logger.error(
                "Python interpreter not found at '%s'.\n"
                "Please check Python installation and path settings.",
                PYTHON,
            )
            exp.status = "ERROR"
            break

        exp.wall_time_sec = time.time() - t0

        if exp.status == "OK":
            _parse_kfold_results(exp)
            logger.info(
                "%s done in %.0fs — JSD=%.4f±%.4f | MIA=%.4f±%.4f | TSTR_RF=%.4f±%.4f",
                exp.name,
                exp.wall_time_sec,
                exp.avg_js if not math.isnan(exp.avg_js) else -1,
                exp.std_js if not math.isnan(exp.std_js) else -1,
                exp.avg_mia_auc if not math.isnan(exp.avg_mia_auc) else -1,
                exp.std_mia_auc if not math.isnan(exp.std_mia_auc) else -1,
                exp.avg_tstr_rf_f1 if not math.isnan(exp.avg_tstr_rf_f1) else -1,
                exp.std_tstr_rf_f1 if not math.isnan(exp.std_tstr_rf_f1) else -1,
            )

    # ------------------------------------------------------------------
    # Save results and print summary
    # ------------------------------------------------------------------
    _print_summary(selected)
    results_path = _save_experiment_results(selected, args.out)

    # Summary statistics
    success = sum(1 for e in selected if e.status == "OK")
    failed = sum(1 for e in selected if e.status != "OK" and e.status != "PENDING")
    print(f"\n  Completed: {success}/{len(selected)}  |  Failed: {failed}")
    print(f"  Per-fold results: {results_path}\n")

    # ------------------------------------------------------------------
    # Statistical tests
    # ------------------------------------------------------------------
    if not args.no_stats:
        output_dir = os.path.join(
            os.path.dirname(os.path.abspath(args.out)) or ".",
            "statistical_tests",
        )
        _run_statistical_tests(selected, output_dir, alpha=args.alpha)
        _print_statistical_summary(selected, alpha=args.alpha)
    else:
        logger.info("Statistical tests skipped (--no-stats flag).")

    print("BATCH EXPERIMENT RUNNER COMPLETED.")


if __name__ == "__main__":
    main()