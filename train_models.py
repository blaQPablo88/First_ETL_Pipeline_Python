import pandas as pd
import numpy as np

def train_and_evaluate(models: dict, X_train, y_train, X_test, y_test, test_final, id_column):
    results = []
    
    for name, model in models.items():
        print(f"\n{name} ➜")
        try:
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            corr = np.corrcoef(y_test, y_pred)[0, 1]
            print(f"Pearson Correlation: {corr:.4f}")
            results.append((name, corr))

            preds = model.predict(test_final)
            pd.DataFrame({"ID": id_column, "prediction": preds}).to_csv(f"{name}_prediction.csv", index=False)
        except Exception as e:
            print(f"Error with {name}: {e}")
    
    return pd.DataFrame(results, columns=["Model", "Pearson Correlation"])
