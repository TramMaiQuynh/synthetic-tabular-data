import pytest
import pandas as pd
import numpy as np
from src.validators.schema_validator import SchemaValidator

def test_validator_input_fail_fast():
    validator = SchemaValidator("adult_income")
    
    # Valid raw data frame matching schema expectations
    valid_df = pd.DataFrame({
        "age": [30, 45, 60],
        "capital-gain": [0, 100, 200],
        "capital-loss": [0, 0, 0],
        "education-num": [13, 9, 16],
        "fnlwgt": [100000, 200000, 300000],
        "hours-per-week": [40, 50, 35],
        "education": ["Bachelors", "HS-grad", "Doctorate"],
        "marital-status": ["Never-married", "Divorced", "Widowed"],
        "native-country": ["United-States", "Germany", "Japan"],
        "occupation": ["Adm-clerical", "Sales", "Prof-specialty"],
        "race": ["White", "Black", "Other"],
        "relationship": ["Husband", "Wife", "Not-in-family"],
        "sex": ["Male", "Female", "Male"],
        "workclass": ["Private", "Local-gov", "Self-emp-inc"],
        "income": ["<=50K", ">50K", "<=50K"]
    })
    
    assert validator.validate_input(valid_df, raise_error=True)
    
    # 1. Missing column check
    invalid_df_missing = valid_df.drop(columns=["age"])
    with pytest.raises(ValueError, match="Missing required column"):
        validator.validate_input(invalid_df_missing, raise_error=True)
        
    # 2. Type mismatch check
    invalid_df_type = valid_df.copy()
    invalid_df_type["age"] = ["thirty", "forty-five", "sixty"]
    with pytest.raises(ValueError, match="is continuous but contains non-numeric data type"):
        validator.validate_input(invalid_df_type, raise_error=True)

def test_validator_output_audit_and_correct():
    validator = SchemaValidator("adult_income")
    
    # Create a synthetic data frame with intentional violations
    # age min=17, max=90
    # continuous capital-gain min=0, max=99999
    # education is categorical with specific allowed values
    df_synthetic = pd.DataFrame({
        "age": [10.0, 50.0, 105.0],                 # 10 is under min, 105 is over max
        "capital-gain": [-500.0, 0.0, 150000.0],     # -500 is under min, 150000 is over max
        "capital-loss": [0.0, 0.0, 0.0],
        "education-num": [13.0, 9.0, 16.0],
        "fnlwgt": [100000.0, 200000.0, 300000.0],
        "hours-per-week": [40.0, 50.0, 35.0],
        "education": ["Bachelors", "HS-grad", "InvalidEduClass"],  # InvalidEduClass is invalid
        "marital-status": ["Never-married", "Divorced", "Widowed"],
        "native-country": ["United-States", "Germany", "Japan"],
        "occupation": ["Adm-clerical", "Sales", "Prof-specialty"],
        "race": ["White", "Black", "Other"],
        "relationship": ["Husband", "Wife", "Not-in-family"],
        "sex": ["Male", "Female", "Male"],
        "workclass": ["Private", "Local-gov", "Self-emp-inc"],
        "income": ["<=50K", ">50K", "<=50K"]
    })
    
    # Audit output
    report = validator.audit_output(df_synthetic)
    assert not report["is_valid"]
    
    # Verify continuous violations
    assert "age" in report["continuous_violations"]
    assert report["continuous_violations"]["age"]["under_min_count"] == 1
    assert report["continuous_violations"]["age"]["over_max_count"] == 1
    
    assert "capital-gain" in report["continuous_violations"]
    
    # Verify categorical violations
    assert "education" in report["categorical_violations"]
    assert report["categorical_violations"]["education"]["invalid_count"] == 1
    assert "InvalidEduClass" in report["categorical_violations"]["education"]["invalid_samples"]
    
    # Correct violations
    corrected_df, audit_report = validator.audit_and_correct(df_synthetic)
    
    # Verify age was clamped to min=17, max=90
    assert corrected_df.loc[0, "age"] == 17.0
    assert corrected_df.loc[2, "age"] == 90.0
    
    # Verify capital-gain was clamped to min=0, max=99999
    assert corrected_df.loc[0, "capital-gain"] == 0.0
    assert corrected_df.loc[2, "capital-gain"] == 99999.0
    
    # Verify invalid category was corrected to fallback (the first class, Bachelors)
    assert corrected_df.loc[2, "education"] == "Bachelors"
    
    # Verify audited corrected df is now valid
    new_report = validator.audit_output(corrected_df)
    assert new_report["is_valid"]
