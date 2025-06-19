from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mse)
    r2 = r2_score(true, predicted)
    corr = np.corrcoef(true, predicted)[0, 1]

    print(f"Pearson Correlation: {corr:.4f}")
    print(f"R2 Score: {r2:.4f} | MAE: {mae:.4f} | RMSE: {rmse:.4f}")
    return corr
