"""
Verification Script for Preprocessing & Validation Pipelines
------------------------------------------------------------
Processes the real Telco-Customer-Churn.csv dataset end-to-end:
1. Loads the data using dynamic memory-based chunking.
2. Performs Fail-Fast Input Validation.
3. Fits and transforms the dataset.
4. Validates that the preprocessed dataset only contains numbers.
5. Saves transformation artifacts.
6. Performs Inverse Transformation and verifies restoration.
7. Audits and validates the final outputs.
"""

import os
import pandas as pd
import numpy as np
from src.preprocessing.pipeline import PreprocessingPipeline
from src.validators.schema_validator import SchemaValidator

def main():
    print("="*80)
    print("STARTING PIPELINE VERIFICATION RUN ON TELCO CUSTOMER CHURN DATA")
    print("="*80)
    
    # 1. Initialize Pipeline and Validator
    dataset_name = "telco_customer_churn"
    data_path = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "data", "Telco-Customer-Churn.csv"
    ))
    
    print(f"[1] Initializing pipeline and validator for '{dataset_name}'...")
    pipeline = PreprocessingPipeline(dataset_name)
    validator = SchemaValidator(dataset_name)
    
    # 2. Ingest Data using Dynamic Chunking
    print(f"[2] Loading dataset from: {data_path}...")
    # This will exercise the dynamic row-based chunking logic
    df_raw = pipeline.load_data(data_path)
    print(f"    Loaded shape: {df_raw.shape}")
    print(f"    Dynamic chunk size estimate: {pipeline.estimate_chunk_rows(data_path)} rows")
    
    # 3. Fail-Fast Input Validation
    print("[3] Running Fail-Fast Input Validation...")
    validator.validate_input(df_raw, raise_error=True)
    
    # 4. Run Preprocessing Fit & Transform
    print("[4] Executing Preprocessing (Fit & Transform)...")
    df_preprocessed = pipeline.fit_transform(df_raw)
    print(f"    Preprocessed shape: {df_preprocessed.shape}")
    
    # Assert that all columns are numerical and there are no NaNs
    assert not df_preprocessed.isnull().any().any(), "Preprocessed dataset contains NaN values!"
    for col in df_preprocessed.columns:
        assert pd.api.types.is_numeric_dtype(df_preprocessed[col]), f"Column '{col}' is not numeric ({df_preprocessed[col].dtype})!"
        
    # Assert PII column was dropped
    assert "customerID" not in df_preprocessed.columns, "PII column 'customerID' was not dropped!"
    print("    Preprocessed data verified: all columns are numeric, no NaNs, and PII dropped.")
    
    # 5. Save Artifacts
    print("[5] Saving fitted pipeline artifacts to artifacts/...")
    pipeline.save_artifacts()
    artifact_path = os.path.join(pipeline.artifacts_dir, "preprocessing_pipeline.joblib")
    assert os.path.exists(artifact_path), f"Artifact file not found at {artifact_path}"
    print(f"    Artifact successfully saved at: {artifact_path}")
    
    # 6. Run Inverse Transform
    print("[6] Running Inverse Transform...")
    df_restored = pipeline.inverse_transform(df_preprocessed)
    print(f"    Restored shape: {df_restored.shape}")
    
    # Verify columns are restored correctly (except for the PII customerID which was deleted)
    expected_restored_cols = [c for c in df_raw.columns if c not in pipeline.pii_columns]
    assert set(df_restored.columns) == set(expected_restored_cols), "Restored columns do not match expected!"
    print("    Restored columns match expected (original minus PII).")
    
    # Compare original values for a non-null row to ensure scaling and encoding inverse functions are correct
    # Find a row without NaNs in original dataset
    complete_rows = df_raw.dropna()
    if not complete_rows.empty:
        sample_idx = complete_rows.index[0]
        print(f"    Comparing raw vs restored values for row index {sample_idx}:")
        for col in expected_restored_cols:
            raw_val = df_raw.loc[sample_idx, col]
            restored_val = df_restored.loc[sample_idx, col]
            
            # Since float numbers might have very tiny precision differences, or types cast to float, check equivalence
            if pd.api.types.is_numeric_dtype(df_raw[col]):
                assert np.isclose(float(raw_val), float(restored_val), rtol=1e-3), f"Divergence in numeric column '{col}': raw={raw_val}, restored={restored_val}"
            else:
                assert str(raw_val) == str(restored_val), f"Divergence in categorical column '{col}': raw='{raw_val}', restored='{restored_val}'"
        print("    Success: Raw and restored values match perfectly!")
        
    # 7. Audit & Correct Synthetic Data representation
    print("[7] Auditing and correcting output representation...")
    # Simulate a generated synthetic output by making minor changes to restored dataframe
    df_synthetic = df_restored.copy()
    
    # Introduce an out-of-bounds continuous value
    # tenure min=0, max=72 (approx, check schema continuous_features tenure)
    # We set a row to 100.0 (out of bounds)
    tenure_col = "tenure"
    if tenure_col in df_synthetic.columns:
        df_synthetic.loc[0, tenure_col] = 100.0
        
    # Introduce an invalid category
    gender_col = "gender"
    if gender_col in df_synthetic.columns:
        df_synthetic.loc[0, gender_col] = "Alien"
        
    # Audit synthetic data
    report = validator.audit_output(df_synthetic)
    print(f"    Audit detects violations: {not report['is_valid']}")
    assert not report['is_valid'], "Schema audit failed to flag out-of-bounds/invalid values!"
    
    # Correct the synthetic data
    df_corrected, corrected_report = validator.audit_and_correct(df_synthetic)
    print("    Auditing corrected dataframe...")
    final_report = validator.audit_output(df_corrected)
    assert final_report['is_valid'], "Corrected dataframe still contains violations!"
    print("    Success: Output audited and corrected successfully to 100% compliance!")
    
    # 8. Run Module 3 Evaluation Suite
    print("[8] Running Module 3 Evaluation Suite (Fidelity, Privacy, Utility, Visuals, Reports)...")
    from src.evaluation.orchestrator import EvaluationSuite
    
    # Instantiate Evaluation Suite for Telco dataset
    suite = EvaluationSuite(dataset_name=dataset_name)
    
    # Sample real and synthetic dfs to make evaluation fast and robust
    sample_size = min(500, len(df_raw))
    sample_indices = df_raw.sample(n=sample_size, random_state=42).index
    df_raw_sampled = df_raw.loc[sample_indices].reset_index(drop=True)
    df_synth_sampled = df_corrected.loc[sample_indices].reset_index(drop=True)
    
    # Run evaluation
    results = suite.run_evaluation(
        real_df=df_raw_sampled,
        synth_df=df_synth_sampled,
        target_col="Churn",
        sensitive_col="gender"
    )
    
    print("    Evaluation Suite completed successfully.")
    
    # Check that report files exist
    md_report = results["report_paths"]["markdown"]
    html_report = results["report_paths"]["html"]
    
    assert os.path.exists(md_report), f"Markdown report not found at {md_report}"
    assert os.path.exists(html_report), f"HTML report not found at {html_report}"
    
    # Basic metric assertions to ensure correct structures were generated
    assert "fidelity" in results
    assert "privacy" in results
    assert "utility" in results
    assert results["fidelity"]["avg_js"] >= 0.0
    assert results["privacy"]["mia_auc"] >= 0.0
    assert "Churn" in suite.target_col or results["utility"]["task"] is not None
    
    print(f"    Markdown compliance report saved to: {md_report}")
    print(f"    HTML compliance report saved to: {html_report}")
    
    print("="*80)
    print("ALL PIPELINE VERIFICATION TESTS PASSED SUCCESSFULLY!")
    print("="*80)

if __name__ == "__main__":
    main()
