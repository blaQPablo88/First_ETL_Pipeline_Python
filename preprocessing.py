import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

def clean_infinite_values(df: pd.DataFrame) -> pd.DataFrame:
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df.dropna(axis=1)

def standard_scale(X_train: pd.DataFrame, X_test: pd.DataFrame):
    scaler = StandardScaler()
    num_features = X_train.select_dtypes(include=[np.number]).columns.tolist()
    transformer = ColumnTransformer([
        ("scaler", scaler, num_features)
    ], remainder='passthrough')

    X_train_scaled = transformer.fit_transform(X_train)
    X_test_scaled = transformer.transform(X_test)
    return X_train_scaled, X_test_scaled, transformer
