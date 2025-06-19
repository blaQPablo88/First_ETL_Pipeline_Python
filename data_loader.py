import pandas as pd

def load_data(train_path: str, test_path: str, sample_path: str):
    print("Loading data...")
    train_df = pd.read_parquet(train_path, engine="pyarrow")
    test_df = pd.read_parquet(test_path, engine="pyarrow")
    sample_df = pd.read_csv(sample_path)
    return train_df, test_df, sample_df
