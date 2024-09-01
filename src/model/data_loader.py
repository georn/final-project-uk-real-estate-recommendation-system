import logging
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.database.database import SessionLocal
from src.database.models.processed_property import ProcessedProperty
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

    # Handle tenure preference for users
def tenure_preference_to_int(x):
    if isinstance(x, TenurePreference):
        return x.value
    elif isinstance(x, str):
        return {'FREEHOLD': 0, 'LEASEHOLD': 1, 'NO PREFERENCE': 2}.get(x.upper(), 2)
    elif pd.isna(x):
        return 2  # Default to NO_PREFERENCE for NaN values
    elif isinstance(x, (int, float)):
        return int(x)
    else:
        return 2  # Default to NO_PREFERENCE


def prepare_features(X_property, X_user):
    # List of features we expect in our data
    property_features = ['price', 'size_sq_ft', 'year', 'month', 'day_of_week',
                         'price_to_income_ratio', 'price_to_savings_ratio', 'affordability_score',
                         'has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural',
                         'latitude', 'longitude', 'epc_rating_encoded',
                         'property_type_Detached', 'property_type_Semi_Detached', 'property_type_Terraced',
                         'property_type_Flat_Maisonette', 'property_type_Other',
                         'bedrooms', 'bathrooms', 'tenure']

    user_features = ['income', 'savings', 'max_commute_time', 'family_size', 'tenure_preference']

    # Ensure all expected features are present
    for feature in property_features:
        if feature not in X_property.columns:
            logging.warning(f"Expected feature '{feature}' not found in property data. Adding with default value 0.")
            X_property[feature] = 0

    for feature in user_features:
        if feature not in X_user.columns:
            logging.warning(f"Expected feature '{feature}' not found in user data. Adding with default value 0.")
            X_user[feature] = 0

    # Handle NaN values and convert to appropriate data types
    numeric_property_features = [col for col in property_features if
                                 col not in ['has_garden', 'has_parking', 'location_Urban', 'location_Suburban',
                                             'location_Rural', 'property_type_Detached', 'property_type_Semi_Detached',
                                             'property_type_Terraced', 'property_type_Flat_Maisonette',
                                             'property_type_Other']]
    boolean_property_features = ['has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural',
                                 'property_type_Detached', 'property_type_Semi_Detached', 'property_type_Terraced',
                                 'property_type_Flat_Maisonette', 'property_type_Other']

    # Handle NaN values for size_sq_ft
    X_property['size_sq_ft'] = X_property['size_sq_ft'].fillna(X_property['size_sq_ft'].median())

    for col in numeric_property_features:
        X_property[col] = pd.to_numeric(X_property[col], errors='coerce')
        X_property[col] = X_property[col].fillna(X_property[col].median())

    for col in boolean_property_features:
        X_property[col] = X_property[col].fillna(False).astype(bool)

    # Avoid divide by zero
    X_property['price_to_income_ratio'] = X_property['price'] / (X_user['income'] + 1)
    X_property['price_to_savings_ratio'] = X_property['price'] / (X_user['savings'] + 1)
    X_property['affordability_score'] = (X_user['income'] * 4 + X_user['savings']) / (X_property['price'] + 1)

    # Log transform for price and size
    X_property['log_price'] = np.log1p(X_property['price'])
    X_property['log_size'] = np.log1p(X_property['size_sq_ft'])

    # Handle tenure for properties
    X_property['tenure'] = X_property['tenure'].astype(int)
    logging.info(f"Unique property tenure values: {X_property['tenure'].unique()}")

    # Log tenure preference values
    logging.info(f"Unique tenure_preference values: {X_user['tenure_preference'].unique()}")

    X_user['tenure_preference'] = X_user['tenure_preference'].apply(tenure_preference_to_int)
    logging.info(f"Unique user tenure preference values: {X_user['tenure_preference'].unique()}")

    # Convert all columns to numeric
    for col in X_property.columns:
        if col not in boolean_property_features:
            X_property[col] = pd.to_numeric(X_property[col], errors='coerce')
            X_property[col] = X_property[col].fillna(X_property[col].median())

    for col in X_user.columns:
        X_user[col] = pd.to_numeric(X_user[col], errors='coerce')
        X_user[col] = X_user[col].fillna(X_user[col].median())

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

    logging.info("Sample of property data:")
    logging.info(property_df[['price', 'size_sq_ft', 'tenure']].head())

    logging.info("Sample of user data:")
    logging.info(user_df[['income', 'savings', 'tenure_preference']].head())

    logging.info("Creating property-user pairs")
    pairs = create_property_user_pairs(property_df, user_df, pairs_per_user)
    logging.info(f"Pairs created. Shape: {pairs.shape}")

    logging.info("Sample of pairs data:")
    logging.info(pairs[['price', 'size_sq_ft', 'tenure', 'income', 'savings', 'tenure_preference']].head())

    property_features = ['price', 'size_sq_ft', 'year', 'month', 'day_of_week',
                         'price_to_income_ratio', 'price_to_savings_ratio', 'affordability_score',
                         'has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural',
                         'latitude', 'longitude', 'epc_rating_encoded',
                         'property_type_Detached', 'property_type_Semi_Detached',
                         'property_type_Terraced', 'property_type_Flat_Maisonette', 'property_type_Other',
                         'bedrooms', 'bathrooms', 'tenure']

    user_features = ['income', 'savings', 'max_commute_time', 'family_size', 'tenure_preference']

    logging.info("Preparing features")
    X_property, X_user = prepare_features(pairs[property_features].copy(), pairs[user_features].copy())

    logging.info("Feature statistics:")
    for col in X_property.columns:
        logging.info(f"{col}: min={X_property[col].min()}, max={X_property[col].max()}, mean={X_property[col].mean()}")
    for col in X_user.columns:
        logging.info(f"{col}: min={X_user[col].min()}, max={X_user[col].max()}, mean={X_user[col].mean()}")

    logging.info("Creating target variable")
    affordability_condition = X_property['affordability_score'] >= 0.3
    bedroom_condition = X_property['bedrooms'] >= X_user['family_size'] * 0.3
    price_income_condition = X_property['price_to_income_ratio'] <= 7
    size_condition = X_property['size_sq_ft'] >= (X_user['family_size'] * 50)
    tenure_condition = (X_property['tenure'] == X_user['tenure_preference']) | (X_user['tenure_preference'] == 2)

    logging.info(f"Tenure condition breakdown:")
    logging.info(f"Matching tenure: {(X_property['tenure'] == X_user['tenure_preference']).mean():.2%}")
    logging.info(f"No preference: {(X_user['tenure_preference'] == 2).mean():.2%}")

    # Log the unique values and their counts for tenure and tenure_preference
    logging.info(f"Unique tenure values: {X_property['tenure'].value_counts().to_dict()}")
    logging.info(f"Unique tenure_preference values: {X_user['tenure_preference'].value_counts().to_dict()}")

    y = affordability_condition & bedroom_condition & price_income_condition & size_condition & tenure_condition

    logging.info(f"Affordability condition met: {affordability_condition.mean():.2%}")
    logging.info(f"Bedroom condition met: {bedroom_condition.mean():.2%}")
    logging.info(f"Price-to-income condition met: {price_income_condition.mean():.2%}")
    logging.info(f"Size condition met: {size_condition.mean():.2%}")
    logging.info(f"Tenure condition met: {tenure_condition.mean():.2%}")

    positive_ratio = y.mean()
    logging.info(f"Positive samples ratio: {positive_ratio:.2%}")

    if positive_ratio < 0.01:
        logging.warning("Very few positive samples. Consider relaxing matching criteria further.")

    if positive_ratio == 0:
        logging.error("No positive samples found. Cannot proceed with training.")
        return None, None, None

    logging.info(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
    return X_property, X_user, y


def create_property_user_pairs(property_df, user_df, pairs_per_user=10):
    pairs = []
    for _, user in user_df.iterrows():
        user_tenure_pref = tenure_preference_to_int(user['tenure_preference'])

        if user_tenure_pref == 0:  # FREEHOLD
            filtered_properties = property_df[property_df['tenure'] == 0]
        elif user_tenure_pref == 1:  # LEASEHOLD
            filtered_properties = property_df[property_df['tenure'] == 1]
        else:  # NO_PREFERENCE
            filtered_properties = property_df

        if filtered_properties.empty:
            filtered_properties = property_df
            logging.warning(f"No matching properties found for tenure preference {user_tenure_pref}, using all properties")

        user_properties = filtered_properties.sample(n=min(pairs_per_user, len(filtered_properties)), replace=False)
        user_dict = user.to_dict()
        user_dict['tenure_preference'] = user_tenure_pref  # Store as integer
        for _, prop in user_properties.iterrows():
            pair = {**prop.to_dict(), **user_dict}
            pairs.append(pair)

    result = pd.DataFrame(pairs)
    logging.info(f"Created pairs shape: {result.shape}")
    logging.info(f"Sample of created pairs:\n{result[['tenure', 'tenure_preference']].head()}")
    return result


def load_data(sample_size=None, pairs_per_user=10):
    X_property, X_user, y = load_and_preprocess_data(sample_size, pairs_per_user)

    if X_property is None or X_user is None or y is None:
        logging.error("Data preprocessing failed. Unable to proceed with train-test split.")
        return None, None, None, None, None, None

    X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test = train_test_split(
        X_property, X_user, y, test_size=0.2, random_state=42, stratify=y)

    # Log data types
    logging.info("X_property_train data types:")
    logging.info(X_property_train.dtypes)
    logging.info("X_user_train data types:")
    logging.info(X_user_train.dtypes)

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
