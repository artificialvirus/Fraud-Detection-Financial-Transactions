# File: /scripts/data_preprocessing.py
# This file contains the data preprocessing code.
# It is responsible for loading the dataset, checking for missing values, and visualizing the data distribution.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.impute import SimpleImputer

def load_data(filepath):
    data = pd.read_csv(filepath)
    return data

def preprocess_data(data):
    # Ensure correct data types
    data['Class'] = data['Class'].astype(int)

    # Check for missing values and handle them using SimpleImputer
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

    # Separate features and target
    X = data.drop('Class', axis=1)
    y = data['Class']

    # Check for outliers or anomalies (optional step)
    # Example: Removing extreme outliers
    # Note: This step must be done carefully as it can reintroduce NaNs if not handled properly.
    quantiles = X.quantile([0.01, 0.99])
    X = X.apply(lambda x: x[(x > quantiles.loc[0.01, x.name]) & (x < quantiles.loc[0.99, x.name])])

    # Impute missing values again if outlier removal introduced any NaNs
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Apply SMOTE to the training data
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

    return X_train_res, X_test, y_train_res, y_test
