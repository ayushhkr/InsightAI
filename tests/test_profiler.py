import pandas as pd
from src.profiler import profile_dataset

def test_profile_empty():
    df = pd.DataFrame()
    profile = profile_dataset(df)
    assert profile["num_rows"] == 0

def test_profile_with_nans_and_dupes():
    df = pd.DataFrame({
        "A": [1.0, 2.0, None, 1.0],
        "B": ["x", "y", "x", "x"]
    })
    profile = profile_dataset(df)
    assert profile["num_rows"] == 4
    assert profile["missing_counts"]["A"] == 1
    assert profile["duplicate_rows"] == 1
