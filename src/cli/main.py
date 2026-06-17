"""
Synthetic Tabular Data -- Command-Line Interface
================================================
Entry point exposing four fully-functional subcommands:

    train       Train a generative model (CTGAN / CTVAE / Diffusion).
                Reads all defaults from YAML config; CLI flags override them.

    sample      Load a trained checkpoint and generate synthetic data.
                Supports unconditional and conditional generation, plus
                post-hoc constraint enforcement.

    evaluate    Run the full evaluation suite (fidelity, privacy, utility)
                on a pair of real + synthetic CSV files and export
                Markdown + HTML compliance reports.

    hpo         Run Bayesian hyperparameter optimisation (Optuna / TPE) and
                optionally train with the discovered best parameters.

Usage examples
--------------
# Train on Telco dataset with default YAML config
python -m src.cli.main train --dataset telco_customer_churn

# Train with explicit overrides
python -m src.cli.main train \\
    --dataset telco_customer_churn \\
    --model-type ctgan \\
    --epochs 200 \\
    --batch-size 512 \\
    --lr 1e-4 \\
    --enable-dp \\
    --epsilon 3.0

# Sample 5 000 rows unconditionally
python -m src.cli.main sample \\
    --dataset telco_customer_churn \\
    --n-rows 5000 \\
    --output data/synth_output.csv

# Sample with conditional generation (customers who churned)
python -m src.cli.main sample \\
    --dataset telco_customer_churn \\
    --n-rows 2000 \\
    --condition-col Churn \\
    --condition-val Yes \\
    --output data/churned_customers.csv

# Evaluate
python -m src.cli.main evaluate \\
    --dataset telco_customer_churn \\
    --real-csv data/Telco-Customer-Churn.csv \\
    --synth-csv data/synth_output.csv \\
    --target-col Churn \\
    --sensitive-col gender

# HPO (30 trials) then auto-train with best parameters
python -m src.cli.main hpo \\
    --dataset telco_customer_churn \\
    --model-type ctvae \\
    --n-trials 30 \\
    --train-after-hpo
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import textwrap
from typing import List, Optional

# ---------------------------------------------------------------------------
# Logging bootstrap — set up BEFORE any heavy imports so that module-level
# loggers in src.* also adopt this configuration.
# ---------------------------------------------------------------------------

def _configure_logging(verbose: bool) -> None:
    """Configure root logger with timestamped, coloured-free output."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)-8s] %(name)s - %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt, stream=sys.stdout)
    # Suppress noisy third-party loggers
    for noisy in ("optuna", "matplotlib", "PIL", "numba"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _resolve_artifacts_root(args: argparse.Namespace) -> str:
    """Return the canonical absolute path to the artifacts root directory."""
    root = getattr(args, "artifacts_root", None) or "artifacts"
    return os.path.abspath(root)


def _load_config(dataset_name: str):
    """Load and validate AppConfig for the given dataset (fail-fast on errors)."""
    from src.config.config_loader import ConfigLoader  # noqa: PLC0415
    return ConfigLoader.load_config(dataset_name)


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> int:
    """
    Train a generative model.

    Reads default parameters from the YAML config stack
    (config/default_config.yaml + config/<dataset>/model_config.yaml, etc.).
    Any CLI flag explicitly provided by the user *overrides* the YAML value.

    Returns 0 on success, non-zero on failure.
    """
    logger = logging.getLogger("cli.train")
    logger.info("=== TRAIN subcommand started ===")

    # 1. Load YAML config ---------------------------------------------------
    config = _load_config(args.dataset)
    artifacts_root = _resolve_artifacts_root(args)

    # 2. Resolve effective hyperparameters (CLI > YAML) ----------------------
    model_type   = args.model_type   or config.model.model_type
    epochs       = args.epochs       if args.epochs       is not None else config.model.epochs
    batch_size   = args.batch_size   if args.batch_size   is not None else config.model.batch_size
    lr           = args.lr           if args.lr           is not None else config.model.learning_rate
    enable_dp    = args.enable_dp    if args.enable_dp    else config.privacy.enable_differential_privacy
    target_eps   = args.epsilon      if args.epsilon      is not None else config.privacy.target_epsilon
    target_delta = args.delta        if args.delta        is not None else config.privacy.target_delta
    max_grad_norm = args.max_grad_norm if args.max_grad_norm is not None else config.privacy.max_grad_norm
    early_stop   = args.early_stopping_patience if args.early_stopping_patience is not None else 0

    # Constraint expressions from CLI (--constraint "A > B")
    constraint_expressions: Optional[List[str]] = args.constraints or None

    logger.info(
        "Effective config: model=%s, epochs=%d, batch=%d, lr=%g, dp=%s",
        model_type, epochs, batch_size, lr, enable_dp,
    )
    if enable_dp:
        logger.info("  DP-SGD -> eps=%.3f, delta=%.2e, max_grad_norm=%.2f",
                    target_eps, target_delta, max_grad_norm)

    # 3. Preprocessing -------------------------------------------------------
    from src.preprocessing.pipeline import PreprocessingPipeline  # noqa: PLC0415

    data_path = os.path.abspath(args.data_path)
    if not os.path.exists(data_path):
        logger.error("Data file not found: %s", data_path)
        return 1

    logger.info("Loading and preprocessing data from: %s", data_path)
    pipeline = PreprocessingPipeline(args.dataset)
    df_raw = pipeline.load_data(data_path)
    logger.info("Raw shape: %s", df_raw.shape)

    df_preprocessed = pipeline.fit_transform(df_raw)
    logger.info("Preprocessed shape: %s", df_preprocessed.shape)

    pipeline.save_artifacts()
    logger.info("Preprocessing artifacts saved.")

    # 4. Model training ------------------------------------------------------
    from src.training.trainer import ModelTrainer  # noqa: PLC0415

    trainer = ModelTrainer(
        model_type=model_type,
        dataset_name=args.dataset,
        artifacts_root=artifacts_root,
    )

    # Build model_kwargs from CLI (architecture overrides)
    model_kwargs = {}
    if args.hidden_dim is not None:
        n_layers = args.n_layers if args.n_layers is not None else 2
        model_kwargs["hidden_dims"] = tuple([args.hidden_dim] * n_layers)
    if args.latent_dim is not None:
        model_kwargs["latent_dim"] = args.latent_dim
    if args.noise_dim is not None:
        model_kwargs["noise_dim"] = args.noise_dim

    train_result = trainer.train(
        preprocessed_df=df_preprocessed,
        continuous_cols=pipeline.continuous_cols,
        categorical_cols=pipeline.categorical_cols,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        enable_dp=enable_dp,
        target_epsilon=target_eps,
        target_delta=target_delta,
        max_grad_norm=max_grad_norm,
        constraint_expressions=constraint_expressions,
        model_kwargs=model_kwargs,
        early_stopping_patience=early_stop,
    )

    logger.info("Training complete.")
    logger.info("  Checkpoint : %s", train_result["checkpoint_path"])
    logger.info("  Final epsilon : %s", train_result["epsilon"])

    # 5. Optional immediate sampling after training --------------------------
    if args.sample_after_train:
        n_rows = args.sample_rows or 1000
        output_csv = os.path.join(
            artifacts_root, args.dataset, f"{model_type}_synthetic.csv"
        )
        logger.info("Sampling %d rows immediately after training -> %s", n_rows, output_csv)
        from src.inference.sampler import SyntheticSampler  # noqa: PLC0415
        sampler = SyntheticSampler(
            model_type=model_type,
            dataset_name=args.dataset,
            artifacts_root=artifacts_root,
        )
        sampler.load()
        saved = sampler.generate_and_save(n_rows=n_rows, output_path=output_csv)
        logger.info("Synthetic data saved to: %s", saved)

    logger.info("=== TRAIN subcommand completed successfully ===")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: sample
# ---------------------------------------------------------------------------

def cmd_sample(args: argparse.Namespace) -> int:
    """
    Load a trained model checkpoint and generate synthetic data.

    Supports:
      - Unconditional generation (default).
      - Conditional generation: fix one categorical column to a specific value
        using --condition-col and --condition-val.
      - Post-hoc constraint enforcement via --constraints.
      - Output as CSV (default) or Parquet (--parquet).
    """
    logger = logging.getLogger("cli.sample")
    logger.info("=== SAMPLE subcommand started ===")

    # Resolve config for defaults
    config = _load_config(args.dataset)
    artifacts_root = _resolve_artifacts_root(args)

    model_type = args.model_type or config.model.model_type
    n_rows     = args.n_rows
    batch_size = args.batch_size if args.batch_size is not None else 2048

    # Validate conditional generation arguments
    condition_col = getattr(args, "condition_col", None)
    condition_val = getattr(args, "condition_val", None)
    if bool(condition_col) != bool(condition_val):
        logger.error(
            "--condition-col and --condition-val must be provided together."
        )
        return 1

    if condition_col:
        logger.info(
            "Conditional generation: %s = '%s'", condition_col, condition_val
        )

    # Constraint expressions
    constraint_expressions: Optional[List[str]] = args.constraints or None

    # Resolve output path
    if args.output:
        output_path = os.path.abspath(args.output)
    else:
        ext = ".parquet" if args.parquet else ".csv"
        output_path = os.path.abspath(
            os.path.join(artifacts_root, args.dataset, f"{model_type}_synthetic{ext}")
        )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    logger.info(
        "Loading model: type=%s, dataset=%s, artifacts_root=%s",
        model_type, args.dataset, artifacts_root,
    )

    from src.inference.sampler import SyntheticSampler  # noqa: PLC0415

    sampler = SyntheticSampler(
        model_type=model_type,
        dataset_name=args.dataset,
        artifacts_root=artifacts_root,
        device=args.device,
    )
    sampler.load(
        checkpoint_path=args.checkpoint or None,
        pipeline_path=args.pipeline_path or None,
    )

    logger.info("Generating %d synthetic rows...", n_rows)
    df = sampler.generate(
        n_rows=n_rows,
        condition_col=condition_col,
        condition_val=condition_val,
        batch_size=batch_size,
        constraint_expressions=constraint_expressions,
        max_constraint_retries=args.max_retries,
    )

    logger.info("Generated DataFrame shape: %s", df.shape)

    # Save output
    _, ext = os.path.splitext(output_path.lower())
    if ext == ".parquet" or args.parquet:
        if not output_path.endswith(".parquet"):
            output_path = output_path.rsplit(".", 1)[0] + ".parquet"
        df.to_parquet(output_path, index=False)
    else:
        df.to_csv(output_path, index=False)

    logger.info("Synthetic data saved: %s", output_path)
    _print_summary_stats(df)

    logger.info("=== SAMPLE subcommand completed successfully ===")
    return 0


def _print_summary_stats(df) -> None:
    """Print a concise summary of the generated DataFrame to stdout."""
    logger = logging.getLogger("cli.sample")
    logger.info("--- Output summary ---")
    logger.info("  Shape       : %d rows x %d columns", *df.shape)
    logger.info("  Null values : %d total", int(df.isnull().sum().sum()))
    logger.info("  Dtypes      : %s", dict(df.dtypes.value_counts()))


# ---------------------------------------------------------------------------
# Subcommand: evaluate
# ---------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> int:
    """
    Run the full evaluation suite (fidelity + privacy + utility).

    Loads real and synthetic data from CSV files, runs all metrics, and
    exports Markdown + HTML compliance reports to the artifacts directory.
    """
    logger = logging.getLogger("cli.evaluate")
    logger.info("=== EVALUATE subcommand started ===")

    artifacts_root = _resolve_artifacts_root(args)

    # Validate input files
    real_csv   = os.path.abspath(args.real_csv)
    synth_csv  = os.path.abspath(args.synth_csv)
    for path, label in [(real_csv, "real-csv"), (synth_csv, "synth-csv")]:
        if not os.path.exists(path):
            logger.error("File not found (%s): %s", label, path)
            return 1

    logger.info("Loading real data  : %s", real_csv)
    import pandas as pd  # noqa: PLC0415
    df_real  = pd.read_csv(real_csv)
    logger.info("Loading synth data : %s", synth_csv)
    df_synth = pd.read_csv(synth_csv)

    logger.info("Real  shape : %s", df_real.shape)
    logger.info("Synth shape : %s", df_synth.shape)

    # Optional train/test split files
    real_train_df = None
    real_test_df  = None
    if args.real_train_csv and args.real_test_csv:
        real_train_df = pd.read_csv(os.path.abspath(args.real_train_csv))
        real_test_df  = pd.read_csv(os.path.abspath(args.real_test_csv))
        logger.info(
            "Using explicit train/test split: train=%s, test=%s",
            real_train_df.shape, real_test_df.shape,
        )

    from src.evaluation.orchestrator import EvaluationSuite  # noqa: PLC0415

    suite = EvaluationSuite(
        dataset_name=args.dataset,
        artifacts_root=artifacts_root,
    )

    results = suite.run_evaluation(
        real_df=df_real,
        synth_df=df_synth,
        real_train_df=real_train_df,
        real_test_df=real_test_df,
        target_col=args.target_col or None,
        sensitive_col=args.sensitive_col or None,
    )

    # ── Results summary ──────────────────────────────────────────────────────
    fidelity = results["fidelity"]
    privacy  = results["privacy"]
    utility  = results["utility"]
    paths    = results["report_paths"]

    logger.info("=== Evaluation Results ===")
    logger.info("  [Fidelity]  Avg JS Divergence       : %.4f", fidelity["avg_js"])
    logger.info("  [Fidelity]  Avg Wasserstein Distance: %.4f", fidelity["avg_wasserstein"])
    logger.info("  [Fidelity]  Correlation Difference  : %.4f", fidelity["correlation_difference"])
    logger.info("  [Privacy]   DCR Mean                : %.4f", privacy["dcr_mean"])
    logger.info("  [Privacy]   DCR Leakage %%           : %.2f%%", privacy["dcr_leakage_pct"])
    logger.info("  [Privacy]   MIA Attacker AUC-ROC    : %.4f", privacy["mia_auc"])
    logger.info("  [Utility]   Task                    : %s", utility["task"])
    logger.info("  [Reports]   Markdown: %s", paths["markdown"])
    logger.info("  [Reports]   HTML    : %s", paths["html"])

    logger.info("=== EVALUATE subcommand completed successfully ===")
    return 0


# ---------------------------------------------------------------------------
# Subcommand: hpo
# ---------------------------------------------------------------------------

def cmd_hpo(args: argparse.Namespace) -> int:
    """
    Run Bayesian hyperparameter optimisation (Optuna TPE sampler).

    Runs n_trials optimisation trials using a weighted objective of
    statistical fidelity (Wasserstein distance) and constraint satisfaction.
    When --train-after-hpo is set, the system automatically trains a final
    model with the discovered best parameters.
    """
    logger = logging.getLogger("cli.hpo")
    logger.info("=== HPO subcommand started ===")

    config = _load_config(args.dataset)
    artifacts_root = _resolve_artifacts_root(args)

    model_type   = args.model_type or config.model.model_type
    n_trials     = args.n_trials
    enable_dp    = args.enable_dp if args.enable_dp else config.privacy.enable_differential_privacy
    target_eps   = args.epsilon   if args.epsilon   is not None else config.privacy.target_epsilon
    target_delta = args.delta     if args.delta     is not None else config.privacy.target_delta
    fidelity_weight = args.fidelity_weight
    timeout      = args.timeout or None
    n_jobs       = args.n_jobs

    # Constraint expressions for HPO objective
    constraint_expressions: Optional[List[str]] = args.constraints or None

    # Data loading + preprocessing
    data_path = os.path.abspath(args.data_path)
    if not os.path.exists(data_path):
        logger.error("Data file not found: %s", data_path)
        return 1

    from src.preprocessing.pipeline import PreprocessingPipeline  # noqa: PLC0415

    logger.info("Preprocessing data for HPO: %s", data_path)
    pipeline = PreprocessingPipeline(args.dataset)
    df_raw = pipeline.load_data(data_path)
    df_preprocessed = pipeline.fit_transform(df_raw)
    pipeline.save_artifacts()
    logger.info("Preprocessed shape: %s", df_preprocessed.shape)

    # Run HPO
    from src.training.hpo import HPORunner  # noqa: PLC0415

    hpo = HPORunner(
        model_type=model_type,
        dataset_name=args.dataset,
        artifacts_root=artifacts_root,
        n_trials=n_trials,
        timeout=timeout,
        fidelity_weight=fidelity_weight,
        enable_dp=enable_dp,
        total_epsilon=target_eps,
        target_delta=target_delta,
        hpo_budget_fraction=args.hpo_budget_fraction,
        n_jobs=n_jobs,
    )

    best_params = hpo.run(
        preprocessed_df=df_preprocessed,
        continuous_cols=pipeline.continuous_cols,
        categorical_cols=pipeline.categorical_cols,
        constraint_expressions=constraint_expressions,
    )

    logger.info("=== HPO Best Parameters ===")
    for k, v in best_params.items():
        logger.info("  %-30s : %s", k, v)

    # Optionally train immediately with best params
    if args.train_after_hpo:
        logger.info("--train-after-hpo set: training final model with best params...")

        from src.training.trainer import ModelTrainer  # noqa: PLC0415

        # Extract the residual epsilon allocated for the final training run
        final_epsilon = best_params.pop("_epsilon_for_training", target_eps)

        trainer = ModelTrainer(
            model_type=model_type,
            dataset_name=args.dataset,
            artifacts_root=artifacts_root,
        )

        model_kwargs = best_params.pop("model_kwargs", {})
        train_result = trainer.train(
            preprocessed_df=df_preprocessed,
            continuous_cols=pipeline.continuous_cols,
            categorical_cols=pipeline.categorical_cols,
            epochs=best_params.get("epochs", config.model.epochs),
            batch_size=best_params.get("batch_size", config.model.batch_size),
            lr=best_params.get("lr", config.model.learning_rate),
            enable_dp=enable_dp,
            target_epsilon=final_epsilon,
            target_delta=target_delta,
            max_grad_norm=best_params.get("max_grad_norm", config.privacy.max_grad_norm),
            constraint_expressions=constraint_expressions,
            model_kwargs=model_kwargs,
        )

        logger.info("Final training complete.")
        logger.info("  Checkpoint : %s", train_result["checkpoint_path"])
        logger.info("  Final epsilon : %s", train_result["epsilon"])

    logger.info("=== HPO subcommand completed successfully ===")
    return 0


# ---------------------------------------------------------------------------
# Argument parser construction
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    """
    Build and return the top-level argument parser with all subcommands.

    Design principles:
      - Mutually exclusive flags are grouped.
      - All required / positional arguments are explicitly marked required=True.
      - Defaults are set to None so we can distinguish "not provided" from
        a user-supplied value (enabling YAML-override logic).
    """
    # ── Top-level parser ────────────────────────────────────────────────────
    parser = argparse.ArgumentParser(
        prog="python -m src.cli.main",
        description=textwrap.dedent("""\
            Synthetic Tabular Data Generator -- CLI
            ----------------------------------------
            Generative models : CTGAN | CTVAE | Diffusion
            Privacy           : DP-SGD (Opacus / custom)
            Evaluation        : Fidelity | Privacy | Utility
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging.",
    )
    parser.add_argument(
        "--artifacts-root",
        metavar="DIR",
        default=None,
        help="Root directory for artifacts (default: ./artifacts).",
    )

    subparsers = parser.add_subparsers(dest="subcommand", title="subcommands")
    subparsers.required = True

    # ── train ───────────────────────────────────────────────────────────────
    p_train = subparsers.add_parser(
        "train",
        help="Train a generative model.",
        description=textwrap.dedent("""\
            Train CTGAN, CTVAE, or Diffusion model.
            All hyperparameters default to the YAML config stack; CLI flags
            override them where provided.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_train.set_defaults(func=cmd_train)

    # Required
    p_train.add_argument(
        "--dataset", "-d",
        required=True,
        metavar="NAME",
        help="Dataset name (must correspond to a config/<NAME>/ folder).",
    )
    p_train.add_argument(
        "--data-path",
        required=True,
        metavar="FILE",
        help="Path to the raw input CSV / Excel / Parquet file.",
    )

    # Model selection
    p_train.add_argument(
        "--model-type", "-m",
        choices=["ctgan", "ctvae", "diffusion"],
        default=None,
        help="Generative model type (overrides YAML config.model.model_type).",
    )

    # Training hyperparameters
    hpg = p_train.add_argument_group("training hyperparameters")
    hpg.add_argument("--epochs",      type=int,   default=None, metavar="N",
                     help="Number of training epochs.")
    hpg.add_argument("--batch-size",  type=int,   default=None, metavar="N",
                     help="Mini-batch size.")
    hpg.add_argument("--lr",          type=float, default=None, metavar="LR",
                     help="Learning rate (e.g. 2e-4).")
    hpg.add_argument("--early-stopping-patience", type=int, default=None, metavar="N",
                     help="Epochs without improvement before stopping (0 = disabled).")

    # Architecture overrides
    arch = p_train.add_argument_group("architecture overrides")
    arch.add_argument("--hidden-dim",  type=int, default=None, metavar="D",
                      help="Hidden layer width (applied to all layers).")
    arch.add_argument("--n-layers",    type=int, default=None, metavar="L",
                      help="Number of hidden layers.")
    arch.add_argument("--latent-dim",  type=int, default=None, metavar="D",
                      help="VAE latent space dimension (CTVAE only).")
    arch.add_argument("--noise-dim",   type=int, default=None, metavar="D",
                      help="Noise dimension (CTGAN only).")

    # Differential privacy
    dpg = p_train.add_argument_group("differential privacy (DP-SGD)")
    dpg.add_argument("--enable-dp",   action="store_true", default=False,
                     help="Enable DP-SGD (overrides YAML if set).")
    dpg.add_argument("--epsilon",      type=float, default=None, metavar="EPS",
                     help="Privacy budget epsilon (default from YAML).")
    dpg.add_argument("--delta",        type=float, default=None, metavar="DELTA",
                     help="Privacy failure probability delta (default from YAML).")
    dpg.add_argument("--max-grad-norm", type=float, default=None, metavar="C",
                     help="DP gradient clipping norm C (default from YAML).")

    # Constraints
    cg = p_train.add_argument_group("constraint expressions")
    cg.add_argument(
        "--constraint", "-c",
        dest="constraints",
        action="append",
        metavar="EXPR",
        help=(
            "Logical constraint expression (e.g. 'tenure >= 0'). "
            "May be repeated for multiple constraints."
        ),
    )

    # Post-train sampling shortcut
    sg = p_train.add_argument_group("post-train sampling (optional)")
    sg.add_argument("--sample-after-train", action="store_true", default=False,
                    help="Generate synthetic data immediately after training.")
    sg.add_argument("--sample-rows", type=int, default=1000, metavar="N",
                    help="Number of rows to generate (used with --sample-after-train).")

    # ── sample ──────────────────────────────────────────────────────────────
    p_sample = subparsers.add_parser(
        "sample",
        help="Generate synthetic data from a trained checkpoint.",
        description=textwrap.dedent("""\
            Load a trained model checkpoint and generate synthetic tabular data.
            Supports unconditional and conditional (class-conditioned) generation.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_sample.set_defaults(func=cmd_sample)

    p_sample.add_argument("--dataset", "-d", required=True, metavar="NAME",
                          help="Dataset name.")
    p_sample.add_argument("--model-type", "-m",
                          choices=["ctgan", "ctvae", "diffusion"],
                          default=None,
                          help="Model type (overrides YAML if provided).")
    p_sample.add_argument("--n-rows", "-n", type=int, required=True, metavar="N",
                          help="Number of synthetic rows to generate.")
    p_sample.add_argument("--output", "-o", metavar="FILE", default=None,
                          help="Output file path (.csv or .parquet).")
    p_sample.add_argument("--parquet", action="store_true", default=False,
                          help="Save output as Parquet instead of CSV.")
    p_sample.add_argument("--batch-size", type=int, default=None, metavar="N",
                          help="Generation batch size (default: 2048).")
    p_sample.add_argument("--device", default=None, metavar="DEVICE",
                          help="Compute device, e.g. 'cpu' or 'cuda:0' (auto if omitted).")
    p_sample.add_argument("--checkpoint", metavar="FILE", default=None,
                          help="Override checkpoint path (default: artifacts/<dataset>/checkpoints/<model>_model.pt).")
    p_sample.add_argument("--pipeline-path", metavar="FILE", default=None,
                          help="Override preprocessing pipeline artifact path.")

    # Conditional generation
    cond = p_sample.add_argument_group("conditional generation")
    cond.add_argument("--condition-col", metavar="COL", default=None,
                      help="Column to condition on (must be categorical).")
    cond.add_argument("--condition-val", metavar="VAL", default=None,
                      help="Category value to fix for conditional generation.")

    # Constraints
    p_sample.add_argument(
        "--constraint", "-c",
        dest="constraints",
        action="append",
        metavar="EXPR",
        help="Post-generation constraint expression. May be repeated.",
    )
    p_sample.add_argument("--max-retries", type=int, default=5, metavar="N",
                          help="Max constraint enforcement retries (default: 5).")

    # ── evaluate ────────────────────────────────────────────────────────────
    p_eval = subparsers.add_parser(
        "evaluate",
        help="Run the full evaluation suite (fidelity + privacy + utility).",
        description=textwrap.dedent("""\
            Evaluate synthetic data quality against a real dataset.
            Generates Markdown and HTML compliance reports.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_eval.set_defaults(func=cmd_evaluate)

    p_eval.add_argument("--dataset", "-d", required=True, metavar="NAME",
                        help="Dataset name (for config lookup and output paths).")
    p_eval.add_argument("--real-csv", required=True, metavar="FILE",
                        help="Path to the real (original) dataset CSV.")
    p_eval.add_argument("--synth-csv", required=True, metavar="FILE",
                        help="Path to the synthetic dataset CSV.")
    p_eval.add_argument("--target-col", metavar="COL", default=None,
                        help="ML utility target column (default: from data_schema.yaml).")
    p_eval.add_argument("--sensitive-col", metavar="COL", default=None,
                        help="Column for Attribute Inference Attack (AIA) audit.")

    # Optional explicit train/test splits
    split = p_eval.add_argument_group("optional explicit train/test split")
    split.add_argument("--real-train-csv", metavar="FILE", default=None,
                       help="Real training split CSV (auto-split 80/20 if omitted).")
    split.add_argument("--real-test-csv",  metavar="FILE", default=None,
                       help="Real test split CSV (auto-split 80/20 if omitted).")

    # ── hpo ─────────────────────────────────────────────────────────────────
    p_hpo = subparsers.add_parser(
        "hpo",
        help="Run Bayesian hyperparameter optimisation (Optuna / TPE).",
        description=textwrap.dedent("""\
            Run n_trials Optuna TPE-sampler trials to find optimal
            hyperparameters for the generative model. Optionally train
            immediately with the best discovered parameters.
        """),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p_hpo.set_defaults(func=cmd_hpo)

    p_hpo.add_argument("--dataset", "-d", required=True, metavar="NAME",
                       help="Dataset name.")
    p_hpo.add_argument("--data-path", required=True, metavar="FILE",
                       help="Path to raw input data file.")
    p_hpo.add_argument("--model-type", "-m",
                       choices=["ctgan", "ctvae", "diffusion"],
                       default=None,
                       help="Model type to optimise (overrides YAML).")
    p_hpo.add_argument("--n-trials", type=int, default=20, metavar="N",
                       help="Number of Optuna trials (default: 20).")
    p_hpo.add_argument("--timeout", type=float, default=None, metavar="SEC",
                       help="Wall-clock timeout in seconds for the HPO study.")
    p_hpo.add_argument("--n-jobs", type=int, default=1, metavar="N",
                       help="Parallel Optuna jobs (default: 1 = sequential).")
    p_hpo.add_argument("--fidelity-weight", type=float, default=0.7, metavar="W",
                       help="Weight of fidelity in HPO objective [0,1] (default: 0.7).")
    p_hpo.add_argument("--hpo-budget-fraction", type=float, default=0.2, metavar="F",
                       help="Fraction of total epsilon budget allocated to HPO (DP only).")
    p_hpo.add_argument("--train-after-hpo", action="store_true", default=False,
                       help="Train a final model with the best HPO parameters.")

    # DP flags (shared with train)
    dp_hpo = p_hpo.add_argument_group("differential privacy")
    dp_hpo.add_argument("--enable-dp",   action="store_true", default=False,
                        help="Enable DP-SGD in HPO trials and final training.")
    dp_hpo.add_argument("--epsilon",     type=float, default=None, metavar="EPS",
                        help="Total privacy budget epsilon.")
    dp_hpo.add_argument("--delta",       type=float, default=None, metavar="DELTA",
                        help="Privacy failure probability delta.")

    # Constraints
    p_hpo.add_argument(
        "--constraint", "-c",
        dest="constraints",
        action="append",
        metavar="EXPR",
        help="Constraint expression for objective scoring. May be repeated.",
    )

    return parser


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    """
    Parse arguments, configure logging, dispatch to the correct subcommand,
    and return an exit code.

    Args:
        argv: Argument list (default: sys.argv[1:]).

    Returns:
        int: 0 on success, non-zero on error.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    _configure_logging(verbose=args.verbose)

    try:
        exit_code = args.func(args)
    except KeyboardInterrupt:
        logging.getLogger("cli").warning("Interrupted by user.")
        exit_code = 130
    except Exception as exc:  # noqa: BLE001
        logging.getLogger("cli").exception("Unhandled error: %s", exc)
        exit_code = 1

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
