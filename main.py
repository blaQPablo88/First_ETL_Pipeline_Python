from src.data_loader import load_data
from src.preprocessing import clean_infinite_values, standard_scale
from src.feature_selection import select_features
from src.train_models import train_and_evaluate

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

TRAIN_PATH = "kaggle/input/drw-crypto-market-prediction/train.parquet"
TEST_PATH = "kaggle/input/drw-crypto-market-prediction/test.parquet"
SAMPLE_PATH = "kaggle/input/drw-crypto-market-prediction/sample_submission.csv"

# Load and Clean Data
train_df, test_df, sample_df = load_data(TRAIN_PATH, TEST_PATH, SAMPLE_PATH)
train_df = clean_infinite_values(train_df)

X = train_df.drop(columns=["label"])
y = train_df["label"]
X = X.select_dtypes(include=[float, int])  # Drop object types for now
test_df = test_df[X.columns]

# Feature Selection
X_selected, selector, selected_cols = select_features(X, y, k=200)
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
test_final = test_df[selected_cols]

# Scaling
X_train_scaled, X_test_scaled, transformer = standard_scale(X_train, X_test)
test_scaled = transformer.transform(test_final)

# Train & Evaluate Models
models = {
    "LinearRegression": LinearRegression(),
    "LinearRegression_Optim": LinearRegression(n_jobs=-1)
}
results = train_and_evaluate(models, X_train_scaled, y_train, X_test_scaled, y_test, test_scaled, sample_df["ID"])
results.to_csv("model_performance_summary.csv", index=False)
