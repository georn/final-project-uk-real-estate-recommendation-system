import logging
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.database.database import SessionLocal
from src.database.models.processed_property import ProcessedProperty

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
            logging.info(f"Column {column} - min: {X_property[column].min():.2f}, max: {X_property[column].max():.2f}, mean: {X_property[column].mean():.2f}, std: {X_property[column].std():.2f}")
        else:
            value_counts = X_property[column].value_counts()
            logging.info(f"Column {column} - unique values: {len(value_counts)}, top value: {value_counts.index[0]}, top count: {value_counts.iloc[0]}")

    # Inspect user data
    logging.info("User data inspection:")
    for column in X_user.columns:
        if X_user[column].dtype in [np.float64, np.int64]:
            logging.info(f"Column {column} - min: {X_user[column].min():.2f}, max: {X_user[column].max():.2f}, mean: {X_user[column].mean():.2f}, std: {X_user[column].std():.2f}")
        else:
            value_counts = X_user[column].value_counts()
            logging.info(f"Column {column} - unique values: {len(value_counts)}, top value: {value_counts.index[0]}, top count: {value_counts.iloc[0]}")

    # Check target variable
    logging.info(f"Target variable distribution: {np.bincount(y)}")
    logging.info(f"Target variable ratio: {y.mean():.2f}")

def handle_nan_values(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = df[col].fillna(df[col].median())
        else:
            # Note: This will leave all-NaN columns as NaN
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
    return df

def load_and_preprocess_data(user_file_path, sample_size=None, pairs_per_user=10):
    start_time = time.time()

    logging.info("Loading property data from database")
    db = SessionLocal()
    try:
        query = db.query(ProcessedProperty)
        property_df = pd.read_sql(query.statement, db.bind)
    finally:
        db.close()

    if sample_size:
        property_df = property_df.sample(n=min(sample_size, len(property_df)), random_state=42)
    logging.info(f"Property data shape: {property_df.shape}")

    logging.info(f"Loading user data from: {user_file_path}")
    user_df = pd.read_csv(user_file_path)
    if sample_size:
        user_df = user_df.sample(n=min(sample_size, len(user_df)), random_state=42)
    logging.info(f"User data shape: {user_df.shape}")

    logging.info("Creating property-user pairs")
    pairs = create_property_user_pairs(property_df, user_df, pairs_per_user)
    logging.info(f"Pairs created. Shape: {pairs.shape}")

    # Define property_features based on available columns
    property_features = ['price', 'size_sq_ft', 'year', 'month', 'day_of_week',
                         'has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural',
                         'latitude', 'longitude', 'epc_rating_encoded']
    property_type_features = [col for col in pairs.columns if col.startswith('Property Type_')]
    property_features.extend(property_type_features)

    user_features = ['Income', 'Savings', 'MaxCommuteTime', 'FamilySize']

    all_features = property_features + user_features
    missing_features = [f for f in all_features if f not in pairs.columns]
    if missing_features:
        logging.warning(f"Missing features in the data: {missing_features}")
        for feature in missing_features:
            pairs[feature] = 0  # Add missing features with default value 0
            logging.warning(f"Added missing feature '{feature}' with default value 0")

    X_property = pairs[property_features]
    X_user = pairs[user_features]

    logging.info("Handling NaN values")
    X_property = handle_nan_values(X_property)
    X_user = handle_nan_values(X_user)

    logging.info("Calculating affordability features")
    X_property = X_property.assign(
        price_to_income_ratio = X_property['price'] / pairs['Income'],
        price_to_savings_ratio = X_property['price'] / pairs['Savings'],
        affordability_score = (pairs['Income'] * 4 + pairs['Savings']) / X_property['price']
    )

    logging.info("Creating target variable")
    # Adjust this based on your specific criteria for a suitable property
    y = (X_property['affordability_score'] >= 1)

    logging.info(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
    return X_property, X_user, y

def create_property_user_pairs(property_df, user_df, pairs_per_user=10):
    pairs = []
    for _, user in user_df.iterrows():
        user_properties = property_df.sample(n=min(pairs_per_user, len(property_df)), replace=False)
        user_dict = user.to_dict()
        for _, prop in user_properties.iterrows():
            pair = {**prop, **user_dict}
            pairs.append(pair)
    result = pd.DataFrame(pairs)
    return result

def load_data(user_file_path, sample_size=None, pairs_per_user=10):
    X_property, X_user, y = load_and_preprocess_data(user_file_path, sample_size, pairs_per_user)

    X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test = train_test_split(
        X_property, X_user, y, test_size=0.2, random_state=42, stratify=y)

    return X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test

if __name__ == "__main__":
    user_file_path = '../../data/synthetic_user_profiles/synthetic_user_profiles.csv'
    sample_size = 1000  # Adjust this value as needed
    pairs_per_user = 10  # Adjust this value as needed

    try:
        logging.info("Starting data loading process")
        X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test = load_data(user_file_path, sample_size, pairs_per_user)

        # Display information about the loaded data
        logging.info("\nTraining set shapes:")
        logging.info(f"X_property_train: {X_property_train.shape}")
        logging.info(f"X_user_train: {X_user_train.shape}")
        logging.info(f"y_train: {y_train.shape}")

        # Inspect the data
        inspect_data(X_property_train, X_user_train, y_train)

    except Exception as e:
        logging.error("An error occurred during data loading:", exc_info=True)
