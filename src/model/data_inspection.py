import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def inspect_data(X_property, X_user, y):
    logging.info("Data Inspection:")

    # Check if inputs are NumPy arrays or pandas DataFrames
    if isinstance(X_property, np.ndarray):
        X_property = pd.DataFrame(X_property)
    if isinstance(X_user, np.ndarray):
        X_user = pd.DataFrame(X_user)

    # Check for NaN values
    logging.info(f"NaN values in property data: {X_property.isna().sum().sum()}")
    logging.info(f"NaN values in user data: {X_user.isna().sum().sum()}")

    # Inspect property data
    logging.info("Property data inspection:")
    for column in X_property.columns:
        if X_property[column].dtype in [np.float64, np.int64]:
            logging.info(
                f"Column {column} - min: {X_property[column].min():.2f}, max: {X_property[column].max():.2f}, mean: {X_property[column].mean():.2f}, std: {X_property[column].std():.2f}")
        else:
            value_counts = X_property[column].value_counts()
            logging.info(
                f"Column {column} - unique values: {len(value_counts)}, top value: {value_counts.index[0]}, top count: {value_counts.iloc[0]}")

    # Inspect user data
    logging.info("User data inspection:")
    for column in X_user.columns:
        if X_user[column].dtype in [np.float64, np.int64]:
            logging.info(
                f"Column {column} - min: {X_user[column].min():.2f}, max: {X_user[column].max():.2f}, mean: {X_user[column].mean():.2f}, std: {X_user[column].std():.2f}")
        else:
            value_counts = X_user[column].value_counts()
            logging.info(
                f"Column {column} - unique values: {len(value_counts)}, top value: {value_counts.index[0]}, top count: {value_counts.iloc[0]}")

    # Check target variable
    logging.info(f"Target variable distribution: {np.bincount(y)}")
    logging.info(f"Target variable ratio: {y.mean():.2f}")
