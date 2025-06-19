import numpy as np
import pandas as pd
import os
import gc  # For garbage collection
import warnings

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Stats
from scipy.stats import pearsonr

# Preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Model Evaluation
from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    mean_squared_log_error
)

# Regression Models
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    AdaBoostRegressor,
    GradientBoostingRegressor
)

# Advanced Models
from xgboost import XGBRegressor


# print(np.__version__)
# pd.test()


# retrieving files from 'kaggle/inout' dir and listing the down below
for dirname, _, filenames in os.walk('kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        
# defining Training-paths 
TRAIN_PATH = 'kaggle/input/drw-crypto-market-prediction/train.parquet'
TEST_PATH = 'kaggle/input/drw-crypto-market-prediction/test.parquet'
SAMPLE_SUB_PATH = 'kaggle/input/drw-crypto-market-prediction/sample_submission.csv'


# Load data
print('Loading data')
train_df = pd.read_parquet(TRAIN_PATH, engine='pyarrow')
print('Loading test data')
test_df = pd.read_parquet(TEST_PATH, engine='pyarrow')
print('Loading sample submission')
sample_data_submission = pd.read_csv(SAMPLE_SUB_PATH)


print(f'Train shape: {train_df.shape}')
print(f'Test shape: {test_df.shape}')


train_df.head()
test_df.head()
sample_data_submission

train_df.isnull().sum()
test_df.isnull().sum()

numerical_features = [feature for feature in train_df.columns if train_df[feature].dtype!='O']
categorical_features = [feature for feature in train_df.columns if train_df[feature].dtype=='O']

numerical_features

print("Total Numerical Features:",len(numerical_features))
print("Total Categorical Features:",len(categorical_features))

categorical_features

# with the following function we can select highly correlated features
# it will remove the first feature that is correlated with anything other feature

# def correlation(dataset, threshold):
#     col_corr = set()  # Set of all the names of correlated columns
#     corr_matrix = dataset.corr()
#     for i in range(len(corr_matrix.columns)):
#         for j in range(i):
#             if(corr_matrix.iloc[i, j]) > threshold: # we are interested in absolute coeff value
#                 colname = corr_matrix.columns[i]  # getting the name of column
#                 col_corr.add(colname)
#     return col_corr

# highly_correlated_feature = correlation(train,0.9)


train_df.head()

print("Shape of train:",train_df.shape)


# Observation: There exist some values with are either infinity or out of range of float64 so for this let's replace those values with np.nan then drop all nan values

train_df.replace([np.inf, -np.inf], np.nan, inplace=True)
train_df.isnull().sum().sort_values(ascending=False)

train_df.isnull().sum().sort_values(ascending=False)[lambda x: x > 0]

cols_to_drop = train_df.isnull().sum().sort_values(ascending=False)[lambda x: x > 0].index
train_df.drop(columns=cols_to_drop, inplace=True)
train_df.shape

X = train_df.drop(columns=['label']) # Independent features
y = train_df['label']
from sklearn.feature_selection import SelectKBest, f_regression

selector = SelectKBest(score_func=f_regression, k=200)  # choose top 200
X_selected = selector.fit_transform(X, y)

mask = selector.get_support()

# Get names of selected features
selected_features = X.columns[mask]
print(selected_features)


# columns that we will use for training 
# cols_to_keep = ['X19', 'X20', 'X21', 'X22', 'X27', 'X28', 'X29', 'X219', 'X287', 'X289',
#        'X291', 'X531', 'X858', 'X860', 'X863'] ## keeping only these features for training 

# cols_to_keep = ['bid_qty','ask_qty','buy_qty','sell_qty','volume'] + list(selected_features)

# adding fix with existance handling
manual_features = ['bid_qty','ask_qty','buy_qty','sell_qty','volume']
cols_to_keep = [f for f in manual_features if f in X.columns] + list(selected_features)

X = X[cols_to_keep]
X.head()


# Split before selecting features
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit SelectKBest only on training data
selector = SelectKBest(score_func=f_regression, k=200)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)
test_selected = selector.transform(test_df[selector.get_support()])

# Update columns
selected_features = X_train.columns[selector.get_support()]
X_train = pd.DataFrame(X_train_selected, columns=selected_features)
X_test = pd.DataFrame(X_test_selected, columns=selected_features)
test = pd.DataFrame(test_selected, columns=selected_features)

print("Shape of X_train:",X_train.shape)
print("Shape of X_test:",X_test.shape)


print("Shape of y_train:",y_train.shape)
print("Shape of y_test:",y_test.shape)


scaler = StandardScaler()
numerical_features_list = [feature for feature in X.columns if X[feature].dtype!='O']

transformer = ColumnTransformer(transformers=[
    ('standard_scalling', scaler, numerical_features_list),
], remainder='passthrough')  # Keeps other columns as they are
X_train_trf = transformer.fit_transform(X_train)
X_test_trf = transformer.transform(X_test)


# Creating a function to evaluat model
def evaluate_model(true, predicted):
    mae=mean_absolute_error(true,predicted)
    mse=mean_squared_error(true,predicted)
    rmse=np.sqrt(mse)
    r2=r2_score(true,predicted)

    r = np.corrcoef(true, predicted)[0, 1]
    print()
    print(f"Pearson Correlation Coefficient: {r}")
    print("R2 Score:{:.4f}".format(r2))
    print("MAE:{:.4f}".format(mae))
    print("MSE:{:.4f}".format(mse))
    print("RMSE:{:.4f}".format(rmse))
    
    # ---------
    return 0


test=test_df.drop(columns=['label'], errors='ignore') # dropping target feature from test dataframe
test = test[cols_to_keep]
test_trf = transformer.transform(test)
sample_data_submission


id_column = sample_data_submission['ID']
## Model training
models={
    "Linear_Regression":LinearRegression(),

    "Linear_Regression_with_params":LinearRegression(
                        fit_intercept=True,                 
                        copy_X=True,              
                        n_jobs=-1,                
                        positive=False            
                        ),
    
    # "Lasso":Lasso(),
    
    # "Ridge":Ridge(),
    
    # "ElasticNet":ElasticNet(),
    
    # "DecisionTreeRegressor":DecisionTreeRegressor(),
    
    # "DecisionTreeRegressor_with_params":DecisionTreeRegressor(
    #                                     criterion='squared_error',   
    #                                     splitter='best',             
    #                                     max_depth=10,                
    #                                     min_samples_split=10,       
    #                                     min_samples_leaf=4,         
    #                                     max_features='sqrt',        
    #                                     random_state=42             
    #                                     ),
    
    # "AdaBoost":AdaBoostRegressor(),
    
    # "GradientBoost":GradientBoostingRegressor(),
    
    # "XGBRegressor":XGBRegressor(
    #                 max_depth=10,
    #                 colsample_bytree=0.75,
    #                 subsample=0.9,
    #                 n_estimators=2000,
    #                 learning_rate=0.01,
    #                 gamma=0.01,
    #                 max_delta_step=2,
    #                 eval_metric="rmse",
    #                 enable_categorical=True,
    #                 device = 'cuda'),
    
    # "LGBMRegressor":LGBMRegressor(
    #                 n_estimators=1000,
    #                 learning_rate=0.05,
    #                 max_depth=7,
    #                 num_leaves=31,
    #                 min_child_samples=20,
    #                 subsample=0.8,
    #                 colsample_bytree=0.8,
    #                 random_state=42,
    #                 n_jobs=-1
    #                 ),
    
    # "CatBoostRegressor":CatBoostRegressor(
    #                     iterations= 3500,
    #                     depth= 12,
    #                     loss_function= 'RMSE',
    #                     l2_leaf_reg= 3,
    #                     random_seed= 42,
    #                     eval_metric= 'RMSE',
    #                     silent=True
    #                     ),
    
    # "RandomForest":RandomForestRegressor(
    #                 n_estimators=300,
    #                 max_depth=20,
    #                 min_samples_split=5,
    #                 min_samples_leaf=2,
    #                 max_features='sqrt',
    #                 random_state=42,
    #                 n_jobs=-1
    #                 ),
}


model_name_list = []
corrcoef_list = []

for i in range(len(list(models))):

    model_name = list(models.keys())[i]
    model=list(models.values())[i]
    
    print(model_name,"=============>")
    print()
    
    try:
        model.fit(X_train_trf,y_train) # Train Model on X_train
    except Exception as e:
        print(f"{model_name} failed with error: {e}")
        continue

    # Make Predictions
    y_train_pred=model.predict(X_train_trf)
    y_test_pred=model.predict(X_test_trf)
    
    print()
    print("Evaluating Train Dataset")
    evaluate_model(y_train,y_train_pred)

    print(f"\n{'-'*50}\n")
    
    print("Evaluating Test Dataset")
    evaluate_model(y_test,y_test_pred)
    print("="*60)
    print("\n")

    # appending the vlaues in list 
    model_name_list.append(model_name)
    corrcoef_list.append(np.corrcoef(y_test, y_test_pred)[0, 1])

    # prediction
    prediction = model.predict(test_trf)
    result = pd.DataFrame(
        {
            'ID':id_column,
            'prediction':prediction
        }
    )
    
    result.to_csv('{}_prediction.csv'.format(model_name),index=False)
    print("File saved as '{}_prediction.csv'....".format(model_name))
    print()

    # creating dataframe contains model name and their performance on X_test 
    performance_df = pd.DataFrame({
        'ML Algo Name': model_name_list,
        'Pearson Correlation Coefficient': corrcoef_list
    })


print(performance_df)
performance_df.to_csv("model_performance_summary.csv", index=False)