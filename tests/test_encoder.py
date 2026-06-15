import pytest
import pandas as pd
import numpy as np
from src.preprocessing.encoder import TabularEncoder

def test_encoder_cardinality_and_inverse():
    # max_onehot_cardinality = 2
    # 'city' has 3 unique values (high cardinality) -> Label
    # 'color' has 2 unique values (low cardinality) -> Onehot
    df = pd.DataFrame({
        "color": ["red", "blue", "red", "blue"],
        "city": ["Hanoi", "Saigon", "DaNang", "Hanoi"]
    })
    
    encoder = TabularEncoder(max_onehot_cardinality=2, handle_unknown="ignore")
    encoder.fit(df, ["color", "city"])
    
    # Assert correct types
    assert encoder.encoding_types_["color"] == "onehot"
    assert encoder.encoding_types_["city"] == "label"
    
    # Transform
    res_df = encoder.transform(df)
    
    # Check onehot columns
    assert "color_red" in res_df.columns
    assert "color_blue" in res_df.columns
    assert "color" not in res_df.columns
    
    # Check label columns
    assert "city" in res_df.columns
    assert res_df["city"].dtype == np.float32
    
    # Inverse transform
    restored_df = encoder.inverse_transform(res_df)
    
    # Assert values restored
    pd.testing.assert_frame_equal(restored_df, df)

def test_encoder_unseen_categories():
    df_train = pd.DataFrame({
        "color": ["red", "blue", "red"],
        "city": ["Hanoi", "Saigon", "DaNang"]
    })
    
    encoder = TabularEncoder(max_onehot_cardinality=2, handle_unknown="ignore")
    encoder.fit(df_train, ["color", "city"])
    
    # Create test data with unseen categories
    df_test = pd.DataFrame({
        "color": ["green", "blue"],      # 'green' is unseen
        "city": ["Hanoi", "HaiPhong"]    # 'HaiPhong' is unseen
    })
    
    res_df = encoder.transform(df_test)
    
    # One-hot unseen should have 0s
    assert res_df.loc[0, "color_red"] == 0.0
    assert res_df.loc[0, "color_blue"] == 0.0
    
    # Label unseen should map to 'Unknown' index (which is index 3)
    unknown_idx = encoder.label_maps_["city"]["Unknown"]
    assert res_df.loc[1, "city"] == unknown_idx
    
    # Inverse transform of unseen
    restored_df = encoder.inverse_transform(res_df)
    assert restored_df.loc[0, "color"] is None or pd.isnull(restored_df.loc[0, "color"])
    assert restored_df.loc[1, "city"] == "Unknown"
