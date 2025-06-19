
# Refactoring-Regression-Pipeline-with-GenAI
Refactoring my ["First Pipeline"](https://github.com/blaQPablo88/First_Pipeline_Python) project using GenAI fundementals

# Installation to System (https://numpy.org/install/)
**Windows
choc install numpy

**Linux
sudo apt install python3-numpy 
~ Run script using:
import numpy as np
print(np.__version__)

# Or install to current project 
pip install numpy


# Installing Pandas (https://pandas.pydata.org/getting_started.html)
** Linux
~ pip install pandas

N.B install all modules ('using pip modle_name'), in the .py file, before running, and install kaggle 'pip install kaggle'
kaggle kernels pull yashsahu02/drw-crypto-market-prediction

# What's a .parquet file?
A Parquet file is a type of data file format designed for efficient storage and fast processing of large datasets, especially in big data and analytics workflows.

What is Parquet?
Apache Parquet is an open-source, columnar storage file format optimized for use with big data processing frameworks like Apache Spark, Hadoop, and Python libraries like pandas and pyarrow.

It stores data by columns rather than by rows, which makes it highly efficient for analytical queries that often only need a subset of columns.

It supports efficient compression and encoding schemes to reduce file size and speed up data reading.

Why use Parquet?
Efficient storage: Because it's columnar, it compresses data better than row-based formats like CSV.

Faster queries: Reads only the columns needed, improving speed.

Schema-aware: The file contains metadata describing data types, so it's easy to read the data correctly.

Compatibility: Widely used in data pipelines, cloud data warehouses, and machine learning workflows.


# Project Structure
crypto-regression/
│
├── data/
│   └── kaggle/    # Original data path
├── models/
│   └── saved_predictions/
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── feature_selection.py
│   ├── train_models.py
│   └── utils.py
├── main.py
└── README.md
