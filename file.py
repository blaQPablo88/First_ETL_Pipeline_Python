import numpy as np
import pandas as pd
import os
import gc # For garbage collection

# This here for regression
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler

# This here for processing and scalability
from xgboost import XGBRegressor

# Find out more about
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

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

