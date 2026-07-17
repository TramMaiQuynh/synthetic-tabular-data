"""
End-to-End Pipeline Execution Script
------------------------------------
Loads configuration dynamically from config/ folder (YAML files)
and runs the entire pipeline using those parameters.
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config.config_loader import ConfigLoader
from src.preprocessing.pipeline import PreprocessingPipeline
from src.training.trainer import ModelTrainer, set_global_seed
from src.inference.sampler import SyntheticSampler
from src.evaluation.orchestrator import EvaluationSuite

def main():
    parser = argparse.ArgumentParser(description="End-to-End Pipeline Execution Script")
    parser.add_argument(
        "--dataset", 
        type=str, 
        default="telco_customer_churn", 
        help="Dataset name (folder name under config/)"
    )
    parser.add_argument(
        "--data-path", 
        type=str, 
        default=None, 
        help="Path to raw dataset file (e.g. data/Telco-Customer-Churn.csv)"
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default=None,
        choices=["ctgan", "ctvae", "diffusion"],
        help="Override generative model type (ctgan, ctvae, or diffusion)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override training epochs"
    )
    parser.add_argument(
        "--enable-dp",
        type=str,
        default=None,
        choices=["True", "False"],
        help="Override differential privacy setting (True or False)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility (default: 42)"
    )
    args = parser.parse_args()

    # Set global seed IMMEDIATELY — before preprocessing, training, or any
    # random operation. This is the only position that guarantees full
    # pipeline reproducibility across all three modules.
    set_global_seed(args.seed)

    dataset_name = args.dataset
    
    # Resolve default data paths if not specified
    if args.data_path is None:
        if dataset_name == "telco_customer_churn":
            data_path = os.path.abspath(os.path.join("data", "Telco-Customer-Churn.csv"))
        elif dataset_name == "adult_income":
            data_path = os.path.abspath(os.path.join("data", "adult", "adult.data"))
        else:
            raise ValueError(
                f"No default data path mapped for dataset '{dataset_name}'. "
                f"Please specify --data-path explicitly."
            )
    else:
        data_path = os.path.abspath(args.data_path)

    print("=" * 80)
    print(f"STARTING END-TO-END PIPELINE RUN ON DATASET: {dataset_name.upper()}")
    print(f"Data source: {data_path}")
    print("=" * 80)
    
    artifacts_root = os.path.abspath("artifacts")
    
    # Load configuration from YAML files (merged default + dataset-specific)
    print(f"Loading configuration from YAML files for dataset '{dataset_name}'...")
    config = ConfigLoader.load_config(dataset_name)
    
    model_type = args.model_type if args.model_type is not None else config.model.model_type
    epochs = args.epochs if args.epochs is not None else config.model.epochs
    batch_size = config.model.batch_size
    lr = config.model.learning_rate
    
    if args.enable_dp is not None:
        enable_dp = args.enable_dp == "True"
    else:
        enable_dp = config.privacy.enable_differential_privacy

    # Allow batch runner to override epsilon per-experiment via environment variable
    target_epsilon = config.privacy.target_epsilon
    _env_eps = os.environ.get("SYNTH_OVERRIDE_EPSILON")
    if _env_eps is not None:
        try:
            target_epsilon = float(_env_eps)
            print(f"    [ENV OVERRIDE] target_epsilon -> {target_epsilon}")
        except ValueError:
            print(f"    [WARNING] SYNTH_OVERRIDE_EPSILON='{_env_eps}' is not a valid float, ignoring.")
    
    print(f"    Selected Model:      {model_type.upper()}")
    print(f"    Epochs:              {epochs}")
    print(f"    Batch Size:          {batch_size}")
    print(f"    Learning Rate:       {lr}")
    print(f"    Random Seed:         {args.seed}")
    print(f"    Differential Privacy: {enable_dp}")
    if enable_dp:
        print(f"      - Target Epsilon:  {target_epsilon}")
        print(f"      - Target Delta:    {config.privacy.target_delta}")
    
    # 1. Preprocessing
    print("\n[1] Ingesting and Preprocessing Data...")
    pipeline = PreprocessingPipeline(dataset_name)
    df_raw = pipeline.load_data(data_path)
    print(f"    Loaded raw shape: {df_raw.shape}")

    # --- TRAIN / TEST SPLIT ---
    # The split MUST happen here — before fit_transform and before training —
    # so that the generative model never sees the hold-out test set.
    # This is required for valid MIA, DCR Leakage, AIA, and TSTR evaluations.
    pii_cols_present = [c for c in pipeline.pii_columns if c in df_raw.columns]
    df_no_pii = df_raw.drop(columns=pii_cols_present)

    # Attempt stratified split when the target column is categorical
    target_col_name = config.ingestion.target_column
    stratify_series = None
    if target_col_name and target_col_name in df_no_pii.columns:
        col = df_no_pii[target_col_name]
        is_categorical_target = (
            pd.api.types.is_object_dtype(col)
            or isinstance(col.dtype, pd.CategoricalDtype)
            or col.nunique() <= 20
        )
        if is_categorical_target and col.value_counts().min() >= 2:
            stratify_series = col

    df_train, df_test = train_test_split(
        df_no_pii,
        test_size=0.2,
        random_state=args.seed,
        stratify=stratify_series,
    )
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)
    split_type = "stratified" if stratify_series is not None else "random"
    print(f"    Train/Test split ({split_type}): Real Train={df_train.shape[0]} rows, Real Test={df_test.shape[0]} rows")
    # --------------------------

    # Preprocessor is fit ONLY on Real Train — parameters must not leak from test set
    df_preprocessed = pipeline.fit_transform(df_train, model_type=model_type)
    print(f"    Preprocessed shape: {df_preprocessed.shape}")

    pipeline.save_artifacts()
    
    # 2. Training
    print(f"\n[2] Training Generative Model ({model_type.upper()})...")
    trainer = ModelTrainer(
        model_type=model_type,
        dataset_name=dataset_name,
        artifacts_root=artifacts_root
    )
    
    train_results = trainer.train(
        preprocessed_df=df_preprocessed,
        continuous_cols=pipeline.continuous_cols,
        categorical_cols=pipeline.categorical_cols,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        seed=args.seed,
        enable_dp=enable_dp,
        target_epsilon=target_epsilon,
        target_delta=config.privacy.target_delta,
        max_grad_norm=config.privacy.max_grad_norm
    )
    print(f"    Training complete. Model checkpoint saved at: {train_results['checkpoint_path']}")
    
    # 3. Sampling (Synthetic Data Generation)
    print("\n[3] Generating Synthetic Samples...")
    sampler = SyntheticSampler(
        model_type=model_type,
        dataset_name=dataset_name,
        artifacts_root=artifacts_root
    )
    sampler.load()
    
    df_synthetic = sampler.generate(n_rows=1000)
    print(f"    Generated synthetic shape: {df_synthetic.shape}")
    
    # Save output to data/ folder
    output_name = f"{dataset_name}_synthetic_output.csv"
    output_csv = os.path.abspath(os.path.join("data", output_name))
    df_synthetic.to_csv(output_csv, index=False)
    print(f"    Saved synthetic samples to: {output_csv}")
    
    # 4. Evaluation Suite
    print("\n[4] Running Evaluation Suite & Generating Reports...")
    suite = EvaluationSuite(dataset_name=dataset_name, artifacts_root=artifacts_root)

    target_col = config.ingestion.target_column
    sensitive_col = config.ingestion.quasi_identifiers[0] if config.ingestion.quasi_identifiers else ""

    # Wrap the already-fitted Stage-2 pipeline as the numeric loader for
    # privacy metrics (DCR, NNDR, MIA, AIA).
    # CRITICAL: using the train-fitted pipeline ensures that scaler parameters
    # (min/max) are estimated exclusively from Real Train, not from the full
    # dataset.  If we let the orchestrator create its own helper pipeline
    # (the fallback when pipeline_loader_fn is None), it would refit on
    # real_df (100%), leaking test-set distribution into normalization bounds
    # and invalidating DCR and MIA results.
    def _pipeline_loader(df: pd.DataFrame) -> np.ndarray:
        processed = pipeline.transform(df)
        return processed.values.astype("float32")

    # Pass pre-split real_train_df and real_test_df explicitly so that the
    # orchestrator fallback split (which is invalid when the generator was
    # trained on 100% of data) is never triggered.
    results = suite.run_evaluation(
        real_df=df_raw,
        synth_df=df_synthetic,
        real_train_df=df_train,
        real_test_df=df_test,
        target_col=target_col,
        sensitive_col=sensitive_col,
        pipeline_loader_fn=_pipeline_loader,
    )
    
    print("\n" + "=" * 80)
    print("PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print(f"  - Average JS Divergence:  {results['fidelity']['avg_js']:.4f}")
    print(f"  - MIA Attacker AUC-ROC:   {results['privacy']['mia_auc']:.4f}")
    print(f"  - HTML Compliance Report: {results['report_paths']['html']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
