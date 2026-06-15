import pytest
import pandas as pd
import numpy as np
from src.preprocessing.imputer import TabularImputer

def test_imputer_fit_transform():
    # Create test dataframe
    df = pd.DataFrame({
        "age": [18, np.nan, 30, 40],
        "salary": [1000, 2000, np.nan, 4000],
        "gender": ["male", "female", "male", np.nan]
    })
    
    imputer = TabularImputer(numeric_strategy="median", categorical_strategy="mode")
    continuous = ["age", "salary"]
    categorical = ["gender"]
    
    # Fit and transform
    res_df = imputer.fit_transform(df, continuous, categorical)
    
    # Check that NaNs are filled
    assert not res_df["age"].isnull().any()
    assert not res_df["salary"].isnull().any()
    assert not res_df["gender"].isnull().any()
    
    # Check values filled (median of age [18, 30, 40] is 30)
    assert res_df.loc[1, "age"] == 30.0
    # Check median of salary [1000, 2000, 4000] is 2000
    assert res_df.loc[2, "salary"] == 2000.0
    # Check mode of gender is male
    assert res_df.loc[3, "gender"] == "male"
    
    # Check missing indicators
    assert "age_is_missing" in res_df.columns
    assert "salary_is_missing" in res_df.columns
    assert "gender_is_missing" in res_df.columns
    
    assert res_df.loc[1, "age_is_missing"] == 1.0
    assert res_df.loc[0, "age_is_missing"] == 0.0
    
    # Test inverse_transform
    restored_df = imputer.inverse_transform(res_df)
    
    # Check missing indicator columns are dropped
    assert "age_is_missing" not in restored_df.columns
    assert "salary_is_missing" not in restored_df.columns
    assert "gender_is_missing" not in restored_df.columns
    
    # Check NaN positions are restored
    assert pd.isnull(restored_df.loc[1, "age"])
    assert pd.isnull(restored_df.loc[2, "salary"])
    assert pd.isnull(restored_df.loc[3, "gender"])
    
    # Check non-NaN values are untouched
    assert restored_df.loc[0, "age"] == 18.0
    assert restored_df.loc[0, "salary"] == 1000.0
    assert restored_df.loc[0, "gender"] == "male"
