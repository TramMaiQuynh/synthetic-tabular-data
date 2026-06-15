import pytest
import pandas as pd
import numpy as np
from src.preprocessing.scaler import TabularScaler

def test_scaler_minmax():
    df = pd.DataFrame({
        "age": [10.0, 20.0, 30.0],
        "height": [100.0, 150.0, 200.0]
    })
    
    scaler = TabularScaler(strategy="minmax", feature_range=(0.0, 1.0))
    res_df = scaler.fit_transform(df, ["age", "height"])
    
    # Assert age scaled to [0, 1]
    assert res_df["age"].min() == 0.0
    assert res_df["age"].max() == 1.0
    assert res_df.loc[1, "age"] == 0.5
    
    # Assert height scaled to [0, 1]
    assert res_df["height"].min() == 0.0
    assert res_df["height"].max() == 1.0
    assert res_df.loc[1, "height"] == 0.5
    
    # Inverse transform
    restored_df = scaler.inverse_transform(res_df)
    pd.testing.assert_frame_equal(restored_df, df)

def test_scaler_standard():
    df = pd.DataFrame({
        "age": [10.0, 20.0, 30.0]
    })
    
    scaler = TabularScaler(strategy="standard")
    res_df = scaler.fit_transform(df, ["age"])
    
    # Assert mean is 0, std is 1 after scaling
    assert np.isclose(res_df["age"].mean(), 0.0)
    assert np.isclose(res_df["age"].std(ddof=1), 1.0)
    
    # Inverse transform
    restored_df = scaler.inverse_transform(res_df)
    pd.testing.assert_frame_equal(restored_df, df)
