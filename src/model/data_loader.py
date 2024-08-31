import logging
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.database.database import SessionLocal
from src.database.models.processed_property import ProcessedProperty, EncodedTenure
from src.database.models.synthetic_user import SyntheticUser, TenurePreference

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


def handle_nan_values(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
    return df


def normalize_features(X_property, X_user):
    scaler = MinMaxScaler()

    numerical_features = ['price', 'size_sq_ft', 'latitude', 'longitude', 'epc_rating_encoded',
                          'bedrooms', 'bathrooms', 'price_to_income_ratio', 'price_to_savings_ratio',
                          'affordability_score', 'price_per_bedroom', 'price_per_bathroom', 'tenure']

    # Check if X_property is not empty before normalizing
    if not X_property.empty and all(feature in X_property.columns for feature in numerical_features):
        # Convert to numeric, replacing non-numeric values with NaN
        X_property[numerical_features] = X_property[numerical_features].apply(pd.to_numeric, errors='coerce')

        # Fill NaN with median of each column
        for feature in numerical_features:
            X_property[feature] = X_property[feature].fillna(X_property[feature].median())

        X_property[numerical_features] = scaler.fit_transform(X_property[numerical_features])
    else:
        logging.warning("X_property is empty or missing required features. Skipping normalization.")

    user_numerical_features = ['income', 'savings', 'max_commute_time', 'family_size']  # Updated column names

    # Check if X_user is not empty before normalizing
    if not X_user.empty and all(feature in X_user.columns for feature in user_numerical_features):
        # Convert to numeric, replacing non-numeric values with NaN
        X_user[user_numerical_features] = X_user[user_numerical_features].apply(pd.to_numeric, errors='coerce')

        # Fill NaN with median of each column
        for feature in user_numerical_features:
            X_user[feature] = X_user[feature].fillna(X_user[feature].median())

        X_user[user_numerical_features] = scaler.fit_transform(X_user[user_numerical_features])
    else:
        logging.warning("X_user is empty or missing required features. Skipping normalization.")

    # Handle epc_rating_encoded separately
    if 'epc_rating_encoded' in X_property.columns:
        X_property['epc_rating_encoded'] = X_property['epc_rating_encoded'].astype(float)
        X_property['epc_rating_encoded'] = (X_property['epc_rating_encoded'] - X_property['epc_rating_encoded'].min()) / (X_property['epc_rating_encoded'].max() - X_property['epc_rating_encoded'].min())

    if 'tenure_preference' in X_user.columns:
        X_user['tenure_preference'] = X_user['tenure_preference'].map(
            {TenurePreference.FREEHOLD: 0, TenurePreference.LEASEHOLD: 1, TenurePreference.NO_PREFERENCE: 2})

    return X_property, X_user


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
                         'bedrooms', 'bathrooms', 'tenure']

    user_features = ['income', 'savings', 'max_commute_time', 'family_size', 'tenure_preference']

    X_property = pairs[property_features]
    X_user = pairs[user_features]

    logging.info("Handling NaN values")
    X_property = handle_nan_values(X_property)
    X_user = handle_nan_values(X_user)

    logging.info("Normalizing features")
    X_property, X_user = normalize_features(X_property, X_user)

    logging.info("Creating target variable")
    y = ((X_property['affordability_score'] >= 0.5) &  # Relaxed threshold
         (X_property['bedrooms'] >= X_user['family_size'] * 0.5) &  # Allow for smaller properties
         (X_property['price_to_income_ratio'] <= 5) &  # Relaxed income requirement
         (X_property['size_sq_ft'] >= (X_user['family_size'] * 100)) &  # Reduced size requirement
         ((X_property['tenure'] == X_user['tenure_preference']) | (X_user['tenure_preference'] == TenurePreference.NO_PREFERENCE.value)))

    positive_ratio = y.mean()
    logging.info(f"Positive samples ratio: {positive_ratio:.2%}")

    if positive_ratio < 0.01:  # If less than 1% positive samples
        logging.warning("Very few positive samples. Consider relaxing matching criteria further.")

    logging.info(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
    return X_property, X_user, y


def create_property_user_pairs(property_df, user_df, pairs_per_user=10):
    pairs = []
    for _, user in user_df.iterrows():
        # Filter properties based on user's tenure preference
        if user['tenure_preference'] == TenurePreference.FREEHOLD:
            filtered_properties = property_df[property_df['tenure'] == EncodedTenure.FREEHOLD]
        elif user['tenure_preference'] == TenurePreference.LEASEHOLD:
            filtered_properties = property_df[property_df['tenure'] == EncodedTenure.LEASEHOLD]
        else:  # NO_PREFERENCE
            filtered_properties = property_df

        # If no properties match the preference, use all properties
        if filtered_properties.empty:
            filtered_properties = property_df

        user_properties = filtered_properties.sample(n=min(pairs_per_user, len(filtered_properties)), replace=False)
        user_dict = user.to_dict()
        for _, prop in user_properties.iterrows():
            pair = {**prop.to_dict(), **user_dict}
            pairs.append(pair)
    result = pd.DataFrame(pairs)
    return result


def load_data(sample_size=None, pairs_per_user=10):
    X_property, X_user, y = load_and_preprocess_data(sample_size, pairs_per_user)

    if X_property is None or X_user is None or y is None:
        logging.error("Data preprocessing failed. Unable to proceed with train-test split.")
        return None, None, None, None, None, None

    X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test = train_test_split(
        X_property, X_user, y, test_size=0.2, random_state=42, stratify=y)

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
