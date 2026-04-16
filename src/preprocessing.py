import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_california_housing

def load_data():
    """Load California Housing dataset"""
    california = fetch_california_housing()
    df = pd.DataFrame(california.data, columns=california.feature_names)
    df['PRICE'] = california.target
    return df

def prepare_features_target(df):
    """Separate features (X) and target (y)"""
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']
    return X, y

def split_data(X, y, test_size=0.2, random_state=42):
    """Split data into train and test sets"""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state
    )
    return X_train, X_test, y_train, y_test

def print_data_info(X_train, X_test, y_train, y_test):
    """Print dataset information"""
    print("="*50)
    print("DATA PREPARATION SUMMARY")
    print("="*50)
    print(f"Training set: {X_train.shape[0]} houses, {X_train.shape[1]} features")
    print(f"Test set: {X_test.shape[0]} houses, {X_test.shape[1]} features")
    print(f"Train/Test ratio: {len(X_train)/len(X_test):.2f}:1")