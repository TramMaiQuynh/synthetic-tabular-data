"""
Batch Experiment Runner — Scientific Model Comparison
======================================================
Runs all experiment groups and aggregates
results into a single CSV for tabular reporting.

Experiment Groups
-----------------
A: Architecture Comparison   — CTGAN / CTVAE / Diffusion  (same dataset, no DP, same seed)
B: Privacy-Utility Trade-off — CTVAE at epsilon=inf, 10, 3, 1.5  (same dataset, same seed)
C: Generalization Check      — Best model on adult_income  (cross-dataset validation)

Usage
-----
    .venv\\Scripts\\python run_experiments.py              # run all groups
    .venv\\Scripts\\python run_experiments.py --groups A B  # run only groups A and B
    .venv\\Scripts\\python run_experiments.py --dry-run     # print plan without running
"""

from __future__ import annotations

import argparse
import csv
import logging
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Optional

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
# Experiment definition
# ---------------------------------------------------------------------------
@dataclass
class Experiment:
    group: str
    name: str
    dataset: str
    model_type: str
    epochs: int
    enable_dp: bool
    epsilon: Optional[float]    # None when DP is disabled
    seed: int = 42

    # Filled after the run
    avg_js: float = float("nan")
    avg_wasserstein: float = float("nan")
    corr_diff: float = float("nan")
    dcr_leakage_pct: float = float("nan")
    dcr_mean: float = float("nan")
    nndr_mean: float = float("nan")
    mia_auc: float = float("nan")
    tstr_rf_f1: float = float("nan")
    tstr_gb_f1: float = float("nan")
    tstr_lr_f1: float = float("nan")
    trtr_rf_f1: float = float("nan")
    trtr_gb_f1: float = float("nan")
    trtr_lr_f1: float = float("nan")
    wall_time_sec: float = float("nan")
    status: str = "PENDING"
    report_html: str = ""


# ---------------------------------------------------------------------------
# Experiment registry
# ---------------------------------------------------------------------------
EPOCHS_MAIN = 100   # keep equal across all experiments — critical control variable

EXPERIMENTS: List[Experiment] = [
    # -----------------------------------------------------------------------
    # Group A: Architecture Comparison (no DP — isolate model architecture effect)
    # Control variables: dataset, epochs, batch_size, lr, seed
    # Independent variable: model_type
    # -----------------------------------------------------------------------
    Experiment("A", "A1-CTVAE",     "telco_customer_churn", "ctvae",    EPOCHS_MAIN, False, None),
    Experiment("A", "A2-CTGAN",     "telco_customer_churn", "ctgan",    EPOCHS_MAIN, False, None),
    Experiment("A", "A3-Diffusion", "telco_customer_churn", "diffusion",EPOCHS_MAIN, False, None),

    # -----------------------------------------------------------------------
    # Group B: Privacy-Utility Trade-off (best architecture = CTVAE)
    # Control variables: dataset, model, epochs, batch_size, lr, seed
    # Independent variable: DP epsilon (privacy budget)
    # NOTE: B1-NoDP is intentionally OMITTED — it is identical to A1-CTVAE.
    # When reporting Group B, reuse the A1-CTVAE result as the no-DP baseline.
    # -----------------------------------------------------------------------
    Experiment("B", "B2-DP-e10",  "telco_customer_churn", "ctvae", EPOCHS_MAIN, True,  10.0),
    Experiment("B", "B3-DP-e3",   "telco_customer_churn", "ctvae", EPOCHS_MAIN, True,   3.0),
    Experiment("B", "B4-DP-e1.5", "telco_customer_churn", "ctvae", EPOCHS_MAIN, True,   1.5),

    # -----------------------------------------------------------------------
    # Group C: Generalization Check (best model on a different dataset)
    # Validates that the best model is not over-fit to one domain
    # -----------------------------------------------------------------------
    Experiment("C", "C1-Adult-CTVAE", "adult_income", "ctvae", EPOCHS_MAIN, False, None),
    Experiment("C", "C2-Adult-CTGAN", "adult_income", "ctgan", EPOCHS_MAIN, False, None),
    Experiment("C", "C3-Adult-Diffusion", "adult_income", "diffusion", EPOCHS_MAIN, False, None),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
DATA_PATHS = {
    "telco_customer_churn": os.path.join("data", "Telco-Customer-Churn.csv"),
    "adult_income":         os.path.join("data", "adult", "adult.data"),
}


def _build_cmd(exp: Experiment) -> List[str]:
    cmd = [
        PYTHON, "run_pipeline.py",
        "--dataset",    exp.dataset,
        "--model-type", exp.model_type,
        "--epochs",     str(exp.epochs),
        "--enable-dp",  "True" if exp.enable_dp else "False",
        "--seed",       str(exp.seed),
    ]
    if exp.dataset in DATA_PATHS:
        cmd += ["--data-path", DATA_PATHS[exp.dataset]]
    return cmd


def _parse_results(exp: Experiment) -> None:
    """Read the latest compliance report markdown to extract key metrics."""
    report_md = os.path.join(
        "artifacts", exp.dataset, "evaluation", "compliance_report.md"
    )
    report_html = os.path.join(
        "artifacts", exp.dataset, "evaluation", "compliance_report.html"
    )
    exp.report_html = os.path.abspath(report_html)

    if not os.path.exists(report_md):
        logger.warning("Report not found: %s", report_md)
        return

    import re
    with open(report_md, encoding="utf-8") as f:
        content = f.read()

    def _find(pattern: str) -> float:
        m = re.search(pattern, content, re.IGNORECASE)
        return float(m.group(1)) if m else float("nan")

    # Executive summary line: "Avg JSD: 0.0602"
    exp.avg_js          = _find(r"Avg JSD:\s*([\d.]+)")
    # Report line: "**Average Correlation Difference (Joint Distribution):** 0.1175"
    exp.corr_diff       = _find(r"Average Correlation Difference.*?:\*\*\s*([\d.]+)")
    # Report line: "**DCR Leakage Percentage (<0.0057):** 0.00%"
    exp.dcr_leakage_pct = _find(r"DCR Leakage Percentage.*?:\*\*\s*([\d.]+)%")
    # Report line: "**Distance to Closest Record (DCR) Mean:** 1.6374"
    exp.dcr_mean        = _find(r"Distance to Closest Record.*?Mean:\*\*\s*([\d.]+)")
    # Report line: "**Nearest Neighbor Distance Ratio (NNDR) Mean:** 0.8785"
    exp.nndr_mean       = _find(r"Nearest Neighbor Distance Ratio.*?Mean:\*\*\s*([\d.]+)")
    # Report line: "**Membership Inference Attack (MIA) AUC:** 0.5024"
    exp.mia_auc         = _find(r"Membership Inference Attack.*?AUC:\*\*\s*([\d.]+)")

    # TSTR/TRTR rows: each classifier is a SINGLE markdown table line
    # Do NOT use re.DOTALL — it would cross-match between different classifier rows
    # Support both F1-Macro (classification) and R2-Score (regression) metric names
    rf_m  = re.search(r"RandomForest.*?(?:F1-Macro|R2-Score):\s*([\d.]+).*?(?:F1-Macro|R2-Score):\s*([\d.]+)", content)
    gb_m  = re.search(r"GradientBoosting.*?(?:F1-Macro|R2-Score):\s*([\d.]+).*?(?:F1-Macro|R2-Score):\s*([\d.]+)", content)
    lr_m  = re.search(r"LogisticRegression.*?(?:F1-Macro|R2-Score):\s*([\d.]+).*?(?:F1-Macro|R2-Score):\s*([\d.]+)", content)

    if rf_m:
        exp.trtr_rf_f1, exp.tstr_rf_f1 = float(rf_m.group(1)), float(rf_m.group(2))
    if gb_m:
        exp.trtr_gb_f1, exp.tstr_gb_f1 = float(gb_m.group(1)), float(gb_m.group(2))
    if lr_m:
        exp.trtr_lr_f1, exp.tstr_lr_f1 = float(lr_m.group(1)), float(lr_m.group(2))

    # Wasserstein — average across all continuous columns
    w_vals = [float(x) for x in re.findall(r"Wasserstein Distance\s*\|\s*([\d.]+)", content)]
    if w_vals:
        exp.avg_wasserstein = sum(w_vals) / len(w_vals)


def _fmt(val: float, decimals: int = 4) -> str:
    if math.isnan(val):
        return "N/A"
    return f"{val:.{decimals}f}"


def _save_csv(experiments: List[Experiment], out_path: str) -> None:
    if not experiments:
        logger.warning("No experiment results to save.")
        return
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    rows = []
    for e in experiments:
        d = asdict(e)
        d["epsilon_label"] = f"eps={e.epsilon}" if e.epsilon is not None else "No DP"
        rows.append(d)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Results saved to %s", out_path)


def _print_summary(experiments: List[Experiment]) -> None:
    w = 130
    print("\n" + "=" * w)
    print(f"{'EXPERIMENT RESULTS SUMMARY':^{w}}")
    print("=" * w)
    col = "{:<18} {:<12} {:<10} {:<6} {:<8} {:<10} {:<10} {:<10} {:<10} {:<12} {:<8}"
    print(col.format(
        "Experiment", "Model", "Dataset", "DP", "Epsilon",
        "AvgJSD(v)", "CorrDiff(v)", "MIA_AUC", "DCRLeak%(v)",
        "TSTR_RF_F1(^)", "Status"
    ))
    print("-" * w)
    for e in experiments:
        dp_label = "ON" if e.enable_dp else "OFF"
        eps_label = str(e.epsilon) if e.epsilon is not None else "inf"
        print(col.format(
            e.name, e.model_type.upper(), e.dataset[:10],
            dp_label, eps_label,
            _fmt(e.avg_js), _fmt(e.corr_diff),
            _fmt(e.mia_auc), _fmt(e.dcr_leakage_pct, 2),
            _fmt(e.tstr_rf_f1), e.status,
        ))
    print("=" * w)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scientific Batch Experiment Runner for Synthetic Tabular Data"
    )
    parser.add_argument(
        "--groups", nargs="+", default=["A", "B", "C"],
        help="Experiment groups to run (A, B, C). Default: all"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print the plan without executing any training"
    )
    parser.add_argument(
        "--out", type=str,
        default=os.path.join("results", "experiment_results.csv"),
        help="Path to output CSV file (default: results/experiment_results.csv)"
    )
    args = parser.parse_args()

    selected = [e for e in EXPERIMENTS if e.group in args.groups]

    print("\n" + "=" * 80)
    print(f"  BATCH EXPERIMENT PLAN  —  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"  Groups selected : {args.groups}")
    print(f"  Total runs      : {len(selected)}")
    print(f"  Epochs per run  : {EPOCHS_MAIN}  |  Seed: 42 (FIXED)")
    print(f"  Output CSV      : {os.path.abspath(args.out)}")
    print("=" * 80)

    print("\n  Run plan:")
    for i, exp in enumerate(selected, 1):
        dp_label = f"DP eps={exp.epsilon}" if exp.enable_dp else "No DP"
        print(f"    [{i:02d}] {exp.name:<18} {exp.model_type.upper():<12} {exp.dataset:<25} {dp_label}")
    print()

    if args.dry_run:
        logger.info("Dry-run mode — no experiments will be executed.")
        return

    for i, exp in enumerate(selected, 1):
        dp_label = f"eps={exp.epsilon}" if exp.enable_dp else "no DP"
        print(f"\n{'='*80}")
        print(f"  [{i:02d}/{len(selected):02d}]  {exp.name}  ->  {exp.model_type.upper()} | {dp_label}")
        print(f"{'='*80}")

        cmd = _build_cmd(exp)
        logger.info("CMD: %s", " ".join(cmd))

        env = os.environ.copy()
        # Pass epsilon override via environment variable
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
                PYTHON
            )
            exp.status = "ERROR"
            break

        exp.wall_time_sec = time.time() - t0

        if exp.status == "OK":
            _parse_results(exp)
            logger.info(
                "%s done in %.0fs — JSD=%.4f | MIA=%.4f | TSTR_RF=%.4f",
                exp.name, exp.wall_time_sec,
                exp.avg_js if not math.isnan(exp.avg_js) else -1,
                exp.mia_auc if not math.isnan(exp.mia_auc) else -1,
                exp.tstr_rf_f1 if not math.isnan(exp.tstr_rf_f1) else -1,
            )

    _print_summary(selected)
    _save_csv(selected, args.out)

    success = sum(1 for e in selected if e.status == "OK")
    failed  = sum(1 for e in selected if e.status != "OK" and e.status != "PENDING")
    print(f"\n  Completed: {success}/{len(selected)}  |  Failed: {failed}")
    print(f"  Full results: {os.path.abspath(args.out)}\n")


if __name__ == "__main__":
    main()
