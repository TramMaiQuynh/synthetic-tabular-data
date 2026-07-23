"""
End-to-End Scientific Pipeline Execution Script
------------------------------------------------
Executes a K-Fold Cross Validation pipeline for generative model evaluation.

Methodology:
    1. Load raw data, drop PII columns.
    2. K-Fold Cross Validation (stratified when target is categorical).
       For each fold:
       a. Split data into train/test (seed = base_seed + fold).
       b. Fit PreprocessingPipeline on train set only (prevents test data leakage).
       c. Run Hyperparameter Optimisation (HPO) via HPORunner on train data.
       d. Train generative model with best HPO parameters.
       e. Generate synthetic data (n_rows = len(train) to avoid sample-size bias).
       f. Run Evaluation Suite (fidelity, privacy, utility) on fold-specific split.
    3. Aggregate K-fold results (mean, std) across all folds.
    4. Output: K-fold metrics as JSON report + cleaned summary.

Required arguments:
    --dataset       Dataset name (folder under config/).
    --model-type    Model type: ctgan, ctvae, diffusion.
    --data-path     Path to raw data file.

Optional arguments:
    --n-splits      Number of folds for Cross Validation (default: 5).
    --n-trials      Number of Optuna HPO trials (default: 20).
    --enable-dp     Enable DP-SGD.
    --seed          Base random seed (default: 42; each fold uses seed + fold).
"""

import os
import argparse
import json
import math
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, KFold

from src.config.config_loader import ConfigLoader
from src.preprocessing.pipeline import PreprocessingPipeline
from src.training.trainer import ModelTrainer, set_global_seed
from src.inference.sampler import SyntheticSampler
from src.evaluation.orchestrator import EvaluationSuite


# ---------------------------------------------------------------------------
# Constants for experiment reproducibility
# ---------------------------------------------------------------------------
DEFAULT_DATA_PATHS = {
    "telco_customer_churn": "data/Telco-Customer-Churn.csv",
    "adult_income":         "data/adult/adult.data",
    "bank_marketing":       "data/bank+marketing/bank/bank.csv",
}


def _parse_args() -> argparse.Namespace:
    """Build and parse CLI arguments."""
    parser = argparse.ArgumentParser(
        description="K-Fold Cross-Validation Pipeline for Generative Tabular Models"
    )

    # Required (dataset and model)
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Dataset name (folder name under config/)",
    )
    parser.add_argument(
        "--model-type", type=str, required=True,
        choices=["ctgan", "ctvae", "diffusion"],
        help="Generative model type (ctgan, ctvae, or diffusion)",
    )
    parser.add_argument(
        "--data-path", type=str, default=None,
        help="Path to raw dataset file (uses default if not specified)",
    )

    # K-Fold and HPO
    parser.add_argument(
        "--n-splits", type=int, default=5,
        help="Number of folds for Cross Validation (default: 5). "
             "Must be >= 3 for meaningful statistical aggregation.",
    )
    parser.add_argument(
        "--n-trials", type=int, default=20,
        help="Number of Optuna HPO trials per fold (default: 20). "
             "Set to 0 to skip HPO and use default parameters.",
    )

    # DP-SGD
    parser.add_argument(
        "--enable-dp", action="store_true", default=False,
        help="Enable Differential Privacy (DP-SGD)",
    )

    # Reproducibility
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed. Each fold uses seed + fold_number (default: 42).",
    )

    return parser.parse_args()


def _resolve_data_path(args: argparse.Namespace) -> str:
    """Resolve data file path from CLI argument or default mapping."""
    if args.data_path is not None:
        return os.path.abspath(args.data_path)
    if args.dataset in DEFAULT_DATA_PATHS:
        return os.path.abspath(DEFAULT_DATA_PATHS[args.dataset])
    raise ValueError(
        f"No default data path mapped for dataset '{args.dataset}'. "
        f"Please specify --data-path explicitly."
    )


def _resolve_config(dataset_name: str):
    """Load and return AppConfig + data_schema dict."""
    config = ConfigLoader.load_config(dataset_name)
    schema = ConfigLoader.load_schema(dataset_name)
    return config, schema


def _build_kfold_splitter(
    df: pd.DataFrame,
    target_col_name: str,
    n_splits: int,
    seed: int,
) -> KFold:
    """
    Build a KFold (or StratifiedKFold) splitter.

    Stratification is used when the target column is categorical AND each
    class has at least n_splits samples (otherwise stratification fails).
    This ensures the target proportion is preserved across folds, which is
    critical for classification tasks where some classes are rare.

    Args:
        df: DataFrame (PII columns already removed).
        target_col_name: Name of the target column.
        n_splits: Number of folds.
        seed: Random seed for shuffle.

    Returns:
        KFold or StratifiedKFold instance.
    """
    stratify_series = None
    if target_col_name and target_col_name in df.columns:
        col = df[target_col_name]
        is_categorical = (
            pd.api.types.is_object_dtype(col)
            or isinstance(col.dtype, pd.CategoricalDtype)
            or col.nunique() <= 20
        )
        if is_categorical and col.value_counts().min() >= n_splits:
            stratify_series = col

    if stratify_series is not None:
        return StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=seed
        )
    return KFold(n_splits=n_splits, shuffle=True, random_state=seed)


def _load_raw_data(
    pipeline: PreprocessingPipeline, data_path: str
) -> pd.DataFrame:
    """Load raw data and remove PII columns."""
    df_raw = pipeline.load_data(data_path)
    pii_cols = [c for c in pipeline.pii_columns if c in df_raw.columns]
    return df_raw.drop(columns=pii_cols)


def _run_fold(
    fold: int,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    df_no_pii: pd.DataFrame,
    dataset_name: str,
    model_type: str,
    config: "AppConfig",
    schema: dict,
    artifacts_root: str,
    n_trials: int,
    enable_dp: bool,
    target_epsilon: float,
    is_stratified: bool = False,
) -> dict:
    """
    Execute a single K-Fold iteration:
        - Split → Preprocess → (HPO →) Train → Sample → Evaluate.

    Each fold is fully isolated: a new PreprocessingPipeline is fit on the
    fold's train set only, preventing data leakage from test to train.

    Args:
        fold: Fold index (1-based; used for seed calculation and logging).
        train_idx: Indices of training samples in df_no_pii.
        test_idx: Indices of test samples in df_no_pii.
        df_no_pii: PII-free raw DataFrame (full dataset).
        dataset_name: Dataset identifier.
        model_type: Generative model type.
        config: Loaded AppConfig object.
        schema: data_schema dict.
        artifacts_root: Root path for artifacts.
        n_trials: Number of HPO trials (0 = skip HPO).
        enable_dp: Flag to enable DP-SGD.
        target_epsilon: Target privacy budget epsilon.
        is_stratified: Whether the K-Fold splitter uses stratification.

    Returns:
        dict containing the fold's evaluation results.
    """
    fold_seed = config.seed + fold  # seed isolation per fold

    split_type = "stratified" if is_stratified else "random"
    print(f"\n{'='*30} Fold {fold}/{config.n_splits} ({split_type}, seed={fold_seed}) {'='*30}")

    # ------------------------------------------------------------------
    # 1. Train/test split for this fold
    # ------------------------------------------------------------------
    df_train = df_no_pii.iloc[train_idx].reset_index(drop=True)
    df_test = df_no_pii.iloc[test_idx].reset_index(drop=True)
    print(f"  Train: {df_train.shape[0]} rows  |  Test: {df_test.shape[0]} rows")

    # ------------------------------------------------------------------
    # 2. Preprocessing (fit ONLY on train — test data never leaks)
    # ------------------------------------------------------------------
    pipeline = PreprocessingPipeline(dataset_name)
    df_preprocessed = pipeline.fit_transform(df_train, model_type=model_type)
    fold_pipeline_path = os.path.join(
        artifacts_root, dataset_name,
        f"fold_{fold}_pipeline.joblib"
    )
    pipeline.save_artifacts(fold_pipeline_path)
    print(f"  Preprocessed shape: {df_preprocessed.shape}")

    # ------------------------------------------------------------------
    # 3. HPO (if n_trials > 0)
    # ------------------------------------------------------------------
    if n_trials > 0:
        print(f"  Running HPO ({n_trials} trials, seed={fold_seed})...")
        from src.training.hpo import HPORunner

        hpo = HPORunner(
            model_type=model_type,
            dataset_name=dataset_name,
            artifacts_root=artifacts_root,
            n_trials=n_trials,
            enable_dp=enable_dp,
            total_epsilon=target_epsilon,
            target_delta=config.privacy.target_delta,
            seed=fold_seed,
        )
        best_params = hpo.run(
            preprocessed_df=df_preprocessed,
            continuous_cols=pipeline.continuous_cols,
            categorical_cols=pipeline.categorical_cols,
            pipeline=pipeline,
        )
        # Extract params with safe defaults
        best_epochs = best_params.get("epochs", config.model.epochs)
        best_batch_size = best_params.get(
            "batch_size", config.model.batch_size
        )
        best_lr = best_params.get("lr", config.model.learning_rate)
        best_model_kwargs = best_params.get("model_kwargs", {})
        best_max_grad_norm = best_params.get(
            "max_grad_norm", config.privacy.max_grad_norm
        )
        final_epsilon = best_params.get(
            "_epsilon_for_training", target_epsilon
        )
    else:
        print("  HPO skipped (--n-trials=0). Using default parameters.")
        best_epochs = config.model.epochs
        best_batch_size = config.model.batch_size
        best_lr = config.model.learning_rate
        best_model_kwargs = {}
        best_max_grad_norm = config.privacy.max_grad_norm
        final_epsilon = target_epsilon

    # ------------------------------------------------------------------
    # 4. Training
    # ------------------------------------------------------------------
    print(f"  Training {model_type.upper()} (epochs={best_epochs}, batch={best_batch_size}, lr={best_lr})...")
    trainer = ModelTrainer(
        model_type=model_type,
        dataset_name=dataset_name,
        artifacts_root=artifacts_root,
        checkpoint_dir=os.path.join(
            artifacts_root, dataset_name, f"checkpoints_fold_{fold}"
        ),
    )
    train_result = trainer.train(
        preprocessed_df=df_preprocessed,
        continuous_cols=pipeline.continuous_cols,
        categorical_cols=pipeline.categorical_cols,
        epochs=best_epochs,
        batch_size=best_batch_size,
        lr=best_lr,
        weight_decay=config.model.weight_decay
                      if hasattr(config.model, "weight_decay")
                      else 1e-6,
        seed=fold_seed,
        enable_dp=enable_dp,
        target_epsilon=final_epsilon,
        target_delta=config.privacy.target_delta,
        max_grad_norm=best_max_grad_norm,
        model_kwargs=best_model_kwargs,
    )
    checkpoint_path = train_result["checkpoint_path"]
    print(f"  Checkpoint: {checkpoint_path}")

    # ------------------------------------------------------------------
    # 5. Sampling (n_rows = len(df_train) to avoid sample-size bias)
    # ------------------------------------------------------------------
    n_generate = len(df_train)
    print(f"  Generating {n_generate} synthetic rows...")
    sampler = SyntheticSampler(
        model_type=model_type,
        dataset_name=dataset_name,
        artifacts_root=artifacts_root,
    )
    sampler.load(
        checkpoint_path=checkpoint_path,
        pipeline_path=fold_pipeline_path,
    )
    df_synthetic = sampler.generate(n_rows=n_generate)
    print(f"  Synthetic shape: {df_synthetic.shape}")

    # Save synthetic data with fold-specific filename to prevent overwriting
    # across folds. This enables per-fold inspection and reproducibility.
    _dp_suffix = f"dp_eps{target_epsilon}" if enable_dp else "nodp"
    synth_csv_path = os.path.join(
        artifacts_root, dataset_name, f"{model_type}_{_dp_suffix}",
        f"synthetic_fold_{fold}.csv",
    )
    os.makedirs(os.path.dirname(synth_csv_path), exist_ok=True)
    df_synthetic.to_csv(synth_csv_path, index=False)
    print(f"  Synthetic data saved to: {synth_csv_path}")

    # ------------------------------------------------------------------
    # 6. Evaluation
    # ------------------------------------------------------------------
    print("  Running Evaluation Suite...")
    suite = EvaluationSuite(
        dataset_name=dataset_name,
        artifacts_root=artifacts_root,
        eval_dir=os.path.join(
            artifacts_root, dataset_name, f"evaluation_fold_{fold}"
        ),
    )

    target_col = schema.get("target_column", "")
    quasi_ids = getattr(config.ingestion, "quasi_identifiers", [])
    sensitive_col = quasi_ids[0] if quasi_ids else ""

    # Pipeline loader for privacy metrics (DCR, NNDR, MIA)
    def _pipeline_loader(df: pd.DataFrame) -> np.ndarray:
        processed = pipeline.transform(df)
        return processed.values.astype("float32")

    fold_results = suite.run_evaluation(
        real_df=df_no_pii,
        synth_df=df_synthetic,
        real_train_df=df_train,
        real_test_df=df_test,
        target_col=target_col,
        sensitive_col=sensitive_col,
        pipeline_loader_fn=_pipeline_loader,
    )

    fold_metrics = _extract_fold_metrics(fold_results)
    print(f"  Fold {fold} metrics: "
          f"JSD={fold_metrics['avg_js']:.4f}, "
          f"MIA={fold_metrics['mia_auc']:.4f}, "
          f"TSTR_RF={fold_metrics['tstr_rf_f1']:.4f}")

    return fold_metrics


def _extract_fold_metrics(fold_results: dict) -> dict:
    """
    Extract scalar metrics from the EvaluationSuite result dict.

    The EvaluationSuite returns a nested dict (see orchestrator.py line 244).
    This helper flattens the relevant keys into a single-level dict.

    Args:
        fold_results: Output of EvaluationSuite.run_evaluation().

    Returns:
        dict with scalar metric values.
    """
    fidelity = fold_results.get("fidelity", {})
    privacy = fold_results.get("privacy", {})
    utility = fold_results.get("utility", {})

    # Fidelity
    avg_js = fidelity.get("avg_js", float("nan"))
    avg_wasserstein = fidelity.get("avg_wasserstein", float("nan"))
    corr_diff = fidelity.get("correlation_difference", float("nan"))

    # Privacy
    dcr_mean = privacy.get("dcr_mean", float("nan"))
    dcr_leakage_pct = privacy.get("dcr_leakage_pct", float("nan"))
    mia_auc = privacy.get("mia_auc", float("nan"))

    # Utility (TSTR vs TRTR for 3 classifiers)
    utility_metrics = utility.get("metrics", {})
    tstr_rf_f1 = float("nan")
    tstr_gb_f1 = float("nan")
    tstr_lr_f1 = float("nan")
    trtr_rf_f1 = float("nan")
    trtr_gb_f1 = float("nan")
    trtr_lr_f1 = float("nan")

    for model_name, scores in utility_metrics.items():
        tstr = scores.get("TSTR", {})
        trtr = scores.get("TRTR", {})
        task = utility.get("task", "classification")
        if task == "classification":
            tstr_val = tstr.get("f1_macro", float("nan"))
            trtr_val = trtr.get("f1_macro", float("nan"))
        else:
            tstr_val = tstr.get("r2", float("nan"))
            trtr_val = trtr.get("r2", float("nan"))

        if "RandomForest" in model_name:
            tstr_rf_f1 = tstr_val
            trtr_rf_f1 = trtr_val
        elif "GradientBoosting" in model_name:
            tstr_gb_f1 = tstr_val
            trtr_gb_f1 = trtr_val
        elif "Logistic" in model_name or "Linear" in model_name:
            tstr_lr_f1 = tstr_val
            trtr_lr_f1 = trtr_val

    return {
        "avg_js": avg_js,
        "avg_wasserstein": avg_wasserstein,
        "corr_diff": corr_diff,
        "dcr_mean": dcr_mean,
        "dcr_leakage_pct": dcr_leakage_pct,
        "mia_auc": mia_auc,
        "tstr_rf_f1": tstr_rf_f1,
        "tstr_gb_f1": tstr_gb_f1,
        "tstr_lr_f1": tstr_lr_f1,
        "trtr_rf_f1": trtr_rf_f1,
        "trtr_gb_f1": trtr_gb_f1,
        "trtr_lr_f1": trtr_lr_f1,
    }


def _aggregate_fold_metrics(
    all_metrics: List[dict],
) -> dict:
    """
    Compute mean ± std across all fold metrics.

    Args:
        all_metrics: List of per-fold metric dicts.

    Returns:
        dict with mean, std for each metric.
    """
    metric_names = [
        "avg_js", "avg_wasserstein", "corr_diff",
        "dcr_mean", "dcr_leakage_pct", "mia_auc",
        "tstr_rf_f1", "tstr_gb_f1", "tstr_lr_f1",
        "trtr_rf_f1", "trtr_gb_f1", "trtr_lr_f1",
    ]
    aggregated = {}
    for metric in metric_names:
        values = [
            m[metric] for m in all_metrics
            if not math.isnan(m.get(metric, float("nan")))
        ]
        if values:
            aggregated[metric] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),  # sample std (ddof=1)
                "values": values,
            }
        else:
            aggregated[metric] = {
                "mean": float("nan"),
                "std": float("nan"),
                "values": [],
            }
    return aggregated


def _print_aggregated_results(
    aggregated: dict,
    dataset_name: str,
    model_type: str,
    n_splits: int,
) -> None:
    """Print a formatted summary of K-fold aggregated results."""
    width = 80
    print("\n" + "=" * width)
    print(f"K-FOLD CROSS-VALIDATION SUMMARY ({n_splits} folds)")
    print(f"Dataset: {dataset_name.upper()}  |  Model: {model_type.upper()}")
    print("=" * width)
    print(f"{'Metric':<30} {'Mean':<12} {'Std':<12} {'N':<6}")
    print("-" * width)

    metrics_display = [
        ("JSD (Fidelity)", "avg_js", "{:.4f}"),
        ("Wasserstein (Fidelity)", "avg_wasserstein", "{:.4f}"),
        ("Correlation Diff (Fidelity)", "corr_diff", "{:.4f}"),
        ("DCR Mean (Privacy)", "dcr_mean", "{:.4f}"),
        ("DCR Leakage % (Privacy)", "dcr_leakage_pct", "{:.2f}"),
        ("MIA AUC (Privacy)", "mia_auc", "{:.4f}"),
        ("TSTR RF F1 (Utility)", "tstr_rf_f1", "{:.4f}"),
        ("TSTR GB F1 (Utility)", "tstr_gb_f1", "{:.4f}"),
        ("TSTR LR F1 (Utility)", "tstr_lr_f1", "{:.4f}"),
        ("TRTR RF F1 (Baseline)", "trtr_rf_f1", "{:.4f}"),
        ("TRTR GB F1 (Baseline)", "trtr_gb_f1", "{:.4f}"),
        ("TRTR LR F1 (Baseline)", "trtr_lr_f1", "{:.4f}"),
    ]

    for label, key, fmt in metrics_display:
        if key in aggregated:
            mean_val = aggregated[key]["mean"]
            std_val = aggregated[key]["std"]
            n_vals = len(aggregated[key]["values"])
            if not math.isnan(mean_val):
                print(f"  {label:<28} {fmt.format(mean_val):<12} "
                      f"{fmt.format(std_val):<12} {n_vals:<6}")
            else:
                print(f"  {label:<28} {'N/A':<12} {'N/A':<12} {0:<6}")

    print("=" * width)


def main() -> None:
    """Main entry point: run K-Fold CV pipeline."""
    args = _parse_args()

    # ------------------------------------------------------------------
    # Initial setup: global seed, config, data
    # ------------------------------------------------------------------
    # NOTE: set_global_seed is called only ONCE here for initial data loading
    # reproducibility. Each fold uses its own seed (base + fold).
    set_global_seed(args.seed)

    dataset_name = args.dataset
    model_type = args.model_type
    data_path = _resolve_data_path(args)
    n_splits = args.n_splits
    n_trials = args.n_trials
    enable_dp = args.enable_dp
    base_seed = args.seed

    # Load configuration
    config, schema = _resolve_config(dataset_name)
    target_epsilon = config.privacy.target_epsilon
    _env_eps = os.environ.get("SYNTH_OVERRIDE_EPSILON")
    if _env_eps is not None:
        try:
            target_epsilon = float(_env_eps)
        except ValueError:
            pass  # ignore invalid override

    artifacts_root = os.path.abspath("artifacts")

    print("=" * 80)
    print(f"K-FOLD CV PIPELINE: {dataset_name.upper()}")
    print(f"  Model        : {model_type.upper()}")
    print(f"  Data         : {data_path}")
    print(f"  Folds        : {n_splits}")
    print(f"  HPO trials   : {n_trials}")
    print(f"  DP-SGD       : {enable_dp}")
    if enable_dp:
        print(f"  Target eps   : {target_epsilon}")
    print(f"  Base seed    : {base_seed}")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Load raw data and prepare K-Fold splitter
    # ------------------------------------------------------------------
    preproc = PreprocessingPipeline(dataset_name)
    df_no_pii = _load_raw_data(preproc, data_path)
    target_col_name = schema.get("target_column", "")

    splitter = _build_kfold_splitter(
        df_no_pii, target_col_name, n_splits, base_seed
    )

    # Store splitter params for _run_fold (inject via config hack or pass
    # as argument). Since _run_fold needs n_splits and splitter type for
    # logging, we use a simple namespace approach.
    class _FoldConfig:
        """Minimal config object for _run_fold. Contains only accessed attributes."""
        pass

    fold_config = _FoldConfig()
    fold_config.seed = base_seed
    fold_config.n_splits = n_splits
    fold_config.privacy = config.privacy
    fold_config.model = config.model
    fold_config.ingestion = config.ingestion

    # Determine if splitter uses stratification (for logging)
    is_stratified = isinstance(splitter, StratifiedKFold)

    # ------------------------------------------------------------------
    # K-Fold loop
    # ------------------------------------------------------------------
    all_metrics: List[dict] = []

    for fold, (train_idx, test_idx) in enumerate(
        splitter.split(df_no_pii, df_no_pii[target_col_name]
                       if target_col_name in df_no_pii.columns else None),
        start=1,
    ):
        fold_metrics = _run_fold(
            fold=fold,
            train_idx=train_idx,
            test_idx=test_idx,
            df_no_pii=df_no_pii,
            dataset_name=dataset_name,
            model_type=model_type,
            config=fold_config,
            schema=schema,
            artifacts_root=artifacts_root,
            n_trials=n_trials,
            enable_dp=enable_dp,
            target_epsilon=target_epsilon,
            is_stratified=is_stratified,
        )
        all_metrics.append(fold_metrics)

    # ------------------------------------------------------------------
    # Aggregate and output results
    # ------------------------------------------------------------------
    aggregated = _aggregate_fold_metrics(all_metrics)

    # Save aggregated results as JSON
    dp_suffix = f"dp_eps{target_epsilon}" if enable_dp else "nodp"
    output_dir = os.path.join(
        artifacts_root, dataset_name, f"{model_type}_{dp_suffix}"
    )
    os.makedirs(output_dir, exist_ok=True)

    json_path = os.path.join(output_dir, "kfold_results.json")
    # Convert numpy types to Python native for JSON serialisation
    def _convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(aggregated, f, indent=2, default=_convert)
    print(f"\n  K-fold results saved to: {json_path}")

    # Also save per-fold raw metrics as CSV
    csv_path = os.path.join(output_dir, "kfold_per_fold_metrics.csv")
    df_metrics = pd.DataFrame(all_metrics)
    df_metrics.index = [f"Fold_{i+1}" for i in range(len(all_metrics))]
    df_metrics.to_csv(csv_path, index_label="Fold")
    print(f"  Per-fold metrics saved to: {csv_path}")

    # Print summary
    _print_aggregated_results(aggregated, dataset_name, model_type, n_splits)

    print("K-FOLD CV PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()