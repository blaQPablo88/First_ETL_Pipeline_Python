from sklearn.feature_selection import SelectKBest, f_regression
import pandas as pd

def select_features(X, y, k=200):
    selector = SelectKBest(score_func=f_regression, k=k)
    X_selected = selector.fit_transform(X, y)
    selected_columns = X.columns[selector.get_support()]
    return pd.DataFrame(X_selected, columns=selected_columns), selector, selected_columns
