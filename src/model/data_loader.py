import logging
import os
import time

import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.database.database import SessionLocal
from src.database.models.processed_property import ProcessedProperty
from src.database.models.synthetic_user import SyntheticUser
from src.model.data_inspection import inspect_data
from src.model.data_preprocessing import prepare_features, create_property_user_pairs, create_target_variable

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_and_preprocess_data(sample_size=None, pairs_per_user=10):
    start_time = time.time()

    try:
        db = SessionLocal()
        logging.info("Loading property data from database")
        property_query = db.query(ProcessedProperty)
        if sample_size:
            property_query = property_query.limit(sample_size)
        property_df = pd.read_sql(property_query.statement, db.bind)

        logging.info("Loading user data from database")
        user_query = db.query(SyntheticUser)
        if sample_size:
            user_query = user_query.limit(sample_size)
        user_df = pd.read_sql(user_query.statement, db.bind)
    finally:
        db.close()

    logging.info(f"Property data shape: {property_df.shape}")
    logging.info(f"User data shape: {user_df.shape}")

    if property_df.empty or user_df.empty:
        logging.error("No data loaded from the database. Aborting preprocessing.")
        return None, None, None

    logging.info("Creating property-user pairs")
    pairs = create_property_user_pairs(property_df, user_df, pairs_per_user)
    logging.info(f"Pairs created. Shape: {pairs.shape}")

    property_features = ['price', 'size_sq_ft', 'year', 'month', 'day_of_week',
                         'price_to_income_ratio', 'price_to_savings_ratio', 'affordability_score',
                         'has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural',
                         'latitude', 'longitude', 'epc_rating_encoded',
                         'property_type_Detached', 'property_type_Semi_Detached',
                         'property_type_Terraced', 'property_type_Flat_Maisonette', 'property_type_Other',
                         'bedrooms', 'bathrooms', 'tenure', 'price_relative_to_county_avg']

    county_columns = [col for col in pairs.columns if col.startswith('county_')]
    property_features.extend(county_columns)

    user_features = ['income', 'savings', 'max_commute_time', 'family_size', 'tenure_preference']

    logging.info("Preparing features")
    X_property, X_user = prepare_features(pairs[property_features].copy(), pairs[user_features].copy())

    logging.info("Creating target variable")
    y = create_target_variable(X_property, X_user)

    logging.info(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
    return X_property, X_user, y


def load_data(sample_size=None, pairs_per_user=10):
    os.makedirs('../../models/', exist_ok=True)
    X_property, X_user, y = load_and_preprocess_data(sample_size, pairs_per_user)

    if X_property is None or X_user is None or y is None:
        logging.error("Data preprocessing failed. Unable to proceed with train-test split.")
        return None, None, None, None, None, None

    X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test = train_test_split(
        X_property, X_user, y, test_size=0.2, random_state=42, stratify=y)

    scaler_property = StandardScaler()
    X_property_train = scaler_property.fit_transform(X_property_train)
    X_property_test = scaler_property.transform(X_property_test)

    scaler_user = StandardScaler()
    X_user_train = scaler_user.fit_transform(X_user_train)
    X_user_test = scaler_user.transform(X_user_test)

    # Save the scalers
    try:
        joblib.dump(scaler_property, '../../models/scaler_property.joblib')
        joblib.dump(scaler_user, '../../models/scaler_user.joblib')
        logging.info("Scalers saved successfully.")
    except Exception as e:
        logging.error(f"Failed to save scalers: {str(e)}")

    return X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test


if __name__ == "__main__":
    sample_size = 1000
    pairs_per_user = 10

    try:
        logging.info("Starting data loading process")
        # First, check if we have user data
        db = SessionLocal()
        try:
            user_count = db.query(SyntheticUser).count()
            logging.info(f"Total synthetic users in database: {user_count}")
            if user_count == 0:
                logging.error("No synthetic user data found in the database. Please generate synthetic users first.")
                exit(1)
        finally:
            db.close()

        result = load_data(sample_size, pairs_per_user)

        if result[0] is not None:
            X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test = result

            # Display information about the loaded data
            logging.info("\nTraining set shapes:")
            logging.info(f"X_property_train: {X_property_train.shape}")
            logging.info(f"X_user_train: {X_user_train.shape}")
            logging.info(f"y_train: {y_train.shape}")

            # Inspect the data
            inspect_data(X_property_train, X_user_train, y_train)
        else:
            logging.error("Data loading failed. Unable to proceed with data inspection.")

    except Exception as e:
        logging.error("An error occurred during data loading:", exc_info=True)
