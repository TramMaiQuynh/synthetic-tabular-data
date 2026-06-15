import os
import pytest
import pandas as pd
import numpy as np
from src.preprocessing.pipeline import PreprocessingPipeline

def test_pipeline_fit_transform_and_artifacts():
    # We will use 'adult_income' dataset for the integration test
    pipeline = PreprocessingPipeline("adult_income")
    
    # Assert features were parsed from schema
    assert "age" in pipeline.continuous_cols
    assert "education" in pipeline.categorical_cols
    
    # Create a dummy dataframe matching the schema structure
    # In data_schema.yaml:
    # Continuous: age, capital-gain, capital-loss, education-num, fnlwgt, hours-per-week
    # Categorical: education, marital-status, native-country, occupation, race, relationship, sex, workclass, income
    df = pd.DataFrame({
        "age": [25.0, np.nan, 45.0],
        "capital-gain": [0.0, 1000.0, 500.0],
        "capital-loss": [0.0, 0.0, 0.0],
        "education-num": [10.0, 12.0, np.nan],
        "fnlwgt": [123456.0, 234567.0, 345678.0],
        "hours-per-week": [40.0, 45.0, 35.0],
        "education": ["Bachelors", np.nan, "HS-grad"],
        "marital-status": ["Never-married", "Married-civ-spouse", np.nan],
        "native-country": ["United-States", "Cuba", "Jamaica"],
        "occupation": ["Adm-clerical", "Exec-managerial", np.nan],
        "race": ["White", "Black", "White"],
        "relationship": ["Not-in-family", "Husband", "Wife"],
        "sex": ["Male", "Female", "Male"],
        "workclass": ["Private", "Private", np.nan],
        "income": ["<=50K", ">50K", "<=50K"]
    })
    
    # Run fit_transform
    transformed_df = pipeline.fit_transform(df)
    
    # Check that output contains no NaNs
    assert not transformed_df.isnull().any().any()
    
    # Verify indicator columns were created for missing values (e.g. age, education-num)
    assert "age_is_missing" in transformed_df.columns
    assert "education-num_is_missing" in transformed_df.columns
    
    # Verify that continuous columns are scaled between 0 and 1
    for col in pipeline.continuous_cols:
        if col in transformed_df.columns:
            assert transformed_df[col].min() >= 0.0
            assert transformed_df[col].max() <= 1.0
            
    # Verify saving and loading artifacts
    pipeline.save_artifacts()
    
    # Check that artifact file was written
    artifact_file = os.path.join(pipeline.artifacts_dir, "preprocessing_pipeline.joblib")
    assert os.path.exists(artifact_file)
    
    # Initialize a new pipeline and load artifacts
    new_pipeline = PreprocessingPipeline("adult_income")
    new_pipeline.load_artifacts()
    
    # Check that transform works
    transformed_new = new_pipeline.transform(df)
    pd.testing.assert_frame_equal(transformed_new, transformed_df)
    
    # Check inverse_transform
    restored_df = new_pipeline.inverse_transform(transformed_df)
    
    # Check that NaN positions are restored
    assert pd.isnull(restored_df.loc[1, "age"])
    assert pd.isnull(restored_df.loc[2, "education-num"])
    assert pd.isnull(restored_df.loc[1, "education"])
    
    # Check original shapes and columns are matching (except indicators and PII)
    # The columns should be in the same order as in df (or at least match the set of columns)
    assert set(restored_df.columns) == set(df.columns)
