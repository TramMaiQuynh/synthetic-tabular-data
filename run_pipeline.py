"""
End-to-End Pipeline Execution Script
------------------------------------
Loads configuration dynamically from config/ folder (YAML files)
and runs the entire pipeline using those parameters.
"""

import os
import pandas as pd
from src.config.config_loader import ConfigLoader
from src.preprocessing.pipeline import PreprocessingPipeline
from src.training.trainer import ModelTrainer
from src.inference.sampler import SyntheticSampler
from src.evaluation.orchestrator import EvaluationSuite

def main():
    print("=" * 80)
    print("STARTING END-TO-END PIPELINE RUN ON TELCO CUSTOMER CHURN DATA")
    print("=" * 80)
    
    # Paths & Configuration
    dataset_name = "telco_customer_churn"
    data_path = os.path.abspath(os.path.join("data", "Telco-Customer-Churn.csv"))
    artifacts_root = os.path.abspath("artifacts")
    
    # Load configuration from YAML files (merged default + dataset-specific)
    print(f"Loading configuration from YAML files for dataset '{dataset_name}'...")
    config = ConfigLoader.load_config(dataset_name)
    
    model_type = config.model.model_type
    epochs = config.model.epochs
    batch_size = config.model.batch_size
    lr = config.model.learning_rate
    enable_dp = config.privacy.enable_differential_privacy
    
    print(f"    Selected Model:      {model_type.upper()}")
    print(f"    Epochs:              {epochs}")
    print(f"    Batch Size:          {batch_size}")
    print(f"    Learning Rate:       {lr}")
    print(f"    Differential Privacy: {enable_dp}")
    if enable_dp:
        print(f"      - Target Epsilon:  {config.privacy.target_epsilon}")
        print(f"      - Target Delta:    {config.privacy.target_delta}")
    
    # 1. Preprocessing
    print("\n[1] Ingesting and Preprocessing Data...")
    pipeline = PreprocessingPipeline(dataset_name)
    df_raw = pipeline.load_data(data_path)
    print(f"    Loaded raw shape: {df_raw.shape}")
    
    df_preprocessed = pipeline.fit_transform(df_raw)
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
        enable_dp=enable_dp,
        target_epsilon=config.privacy.target_epsilon,
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
    output_csv = os.path.abspath(os.path.join("data", "Telco_Synthetic_Output.csv"))
    df_synthetic.to_csv(output_csv, index=False)
    print(f"    Saved synthetic samples to: {output_csv}")
    
    # 4. Evaluation Suite
    print("\n[4] Running Evaluation Suite & Generating Reports...")
    suite = EvaluationSuite(dataset_name=dataset_name, artifacts_root=artifacts_root)
    results = suite.run_evaluation(
        real_df=df_raw,
        synth_df=df_synthetic,
        target_col="Churn",
        sensitive_col="gender"
    )
    
    print("\n" + "=" * 80)
    print("PIPELINE RUN COMPLETED SUCCESSFULLY!")
    print(f"  - Average JS Divergence:  {results['fidelity']['avg_js']:.4f}")
    print(f"  - MIA Attacker AUC-ROC:   {results['privacy']['mia_auc']:.4f}")
    print(f"  - HTML Compliance Report: {results['report_paths']['html']}")
    print("=" * 80)

if __name__ == "__main__":
    main()
