import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

import numpy as np
import pandas as pd

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
    for i, column in enumerate(X_property.columns):
        if X_property[column].dtype in [np.float64, np.int64]:
            logging.info(f"Column {i} (Numeric) - min: {X_property[column].min():.2f}, max: {X_property[column].max():.2f}, mean: {X_property[column].mean():.2f}, std: {X_property[column].std():.2f}")
        else:
            value_counts = X_property[column].value_counts()
            logging.info(f"Column {i} (Categorical) - unique values: {len(value_counts)}, top value: {value_counts.index[0]}, top count: {value_counts.iloc[0]}")

    # Inspect user data
    logging.info("User data inspection:")
    for i, column in enumerate(X_user.columns):
        if X_user[column].dtype in [np.float64, np.int64]:
            logging.info(f"Column {i} (Numeric) - min: {X_user[column].min():.2f}, max: {X_user[column].max():.2f}, mean: {X_user[column].mean():.2f}, std: {X_user[column].std():.2f}")
        else:
            value_counts = X_user[column].value_counts()
            logging.info(f"Column {i} (Categorical) - unique values: {len(value_counts)}, top value: {value_counts.index[0]}, top count: {value_counts.iloc[0]}")

    # Check target variable
    logging.info(f"Target variable distribution: {np.bincount(y)}")
    logging.info(f"Target variable ratio: {y.mean():.2f}")

def handle_nan_values(df):
    # Create a copy of the dataframe to avoid SettingWithCopyWarning
    df = df.copy()

    # For numeric columns, fill NaN with median
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        df[col].fillna(df[col].median(), inplace=True)

    # For categorical columns, fill NaN with mode (most frequent value)
    categorical_columns = df.select_dtypes(exclude=[np.number]).columns
    for col in categorical_columns:
        mode_value = df[col].mode()
        if not mode_value.empty:
            df[col].fillna(mode_value.iloc[0], inplace=True)
        else:
            # If no mode found (all values are NaN), fill with a placeholder
            df[col].fillna('Unknown', inplace=True)

    return df

def load_and_preprocess_data(property_file_path, user_file_path, sample_size=None, pairs_per_user=10):
    start_time = time.time()

    logging.info(f"Loading property data from: {property_file_path}")
    property_df = pd.read_csv(property_file_path)
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


    # Ensure all required features are present
    all_features = property_features + user_features
    missing_features = [f for f in all_features if f not in pairs.columns]
    if missing_features:
        raise ValueError(f"Missing features in the data: {missing_features}")

    X_property = pairs[property_features]
    X_user = pairs[user_features]

    logging.info("Handling NaN values in property data")
    X_property = handle_nan_values(X_property)
    logging.info("Handling NaN values in user data")
    X_user = handle_nan_values(X_user)

    logging.info("Calculating affordability ratio")
    pairs['affordability_ratio'] = (pairs['Income'] * 4 + pairs['Savings']) / pairs['price']

    logging.info("Creating target variable")
    pairs['target'] = (pairs['affordability_ratio'] >= 1) & (pairs.apply(lambda row: row[f'location_{row["PreferredLocation"]}'] == 1, axis=1))
    y = pairs['target']

    # Check for infinite values only in numeric columns
    logging.info("Checking for infinite values in numeric columns")
    numeric_columns_property = X_property.select_dtypes(include=[np.number]).columns
    numeric_columns_user = X_user.select_dtypes(include=[np.number]).columns

    if np.isinf(X_property[numeric_columns_property].values).any():
        raise ValueError("Infinite values found in property data. Please handle these values before proceeding.")
    if np.isinf(X_user[numeric_columns_user].values).any():
        raise ValueError("Infinite values found in user data. Please handle these values before proceeding.")

    # Log data types for each DataFrame
    logging.info("Property data types:")
    logging.info(X_property.dtypes)
    logging.info("User data types:")
    logging.info(X_user.dtypes)

    # Log some statistics about the target variable
    logging.info(f"Target variable distribution: {y.value_counts(normalize=True)}")

    logging.info(f"Data preprocessing completed in {time.time() - start_time:.2f} seconds")
    return X_property, X_user, y

def create_property_user_pairs(property_df, user_df, pairs_per_user=10):
    pairs = []
    for _, user in user_df.iterrows():
        user_properties = property_df.sample(n=min(pairs_per_user, len(property_df)), replace=False)
        user_pairs = user_properties.assign(**user.to_dict())
        pairs.append(user_pairs)
    result = pd.concat(pairs, ignore_index=True)
    return result

def split_and_scale_data(X_property, X_user, y):
    X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test = train_test_split(
        X_property, X_user, y, test_size=0.2, random_state=42, stratify=y)

    scaler_property = StandardScaler()
    X_property_train = scaler_property.fit_transform(X_property_train)
    X_property_test = scaler_property.transform(X_property_test)

    scaler_user = StandardScaler()
    X_user_train = scaler_user.fit_transform(X_user_train)
    X_user_test = scaler_user.transform(X_user_test)

    return X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test, scaler_property, scaler_user

def load_data(property_file_path, user_file_path, sample_size=None, pairs_per_user=10):
    X_property, X_user, y = load_and_preprocess_data(property_file_path, user_file_path, sample_size, pairs_per_user)

    X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test = train_test_split(
        X_property, X_user, y, test_size=0.2, random_state=42, stratify=y)

    scaler_property = StandardScaler()
    X_property_train = pd.DataFrame(scaler_property.fit_transform(X_property_train), columns=X_property.columns)
    X_property_test = pd.DataFrame(scaler_property.transform(X_property_test), columns=X_property.columns)

    scaler_user = StandardScaler()
    X_user_train = pd.DataFrame(scaler_user.fit_transform(X_user_train), columns=X_user.columns)
    X_user_test = pd.DataFrame(scaler_user.transform(X_user_test), columns=X_user.columns)

    return X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test, scaler_property, scaler_user

if __name__ == "__main__":
    property_file_path = '../../data/ml-ready-data/ml_ready_data.csv'
    user_file_path = '../../data/synthetic_user_profiles/synthetic_user_profiles.csv'
    sample_size = 1000  # Adjust this value as needed
    pairs_per_user = 10  # Adjust this value as needed

    try:
        logging.info("Starting data loading process")
        X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test, scaler_property, scaler_user = load_data(property_file_path, user_file_path, sample_size, pairs_per_user)

        # Display information about the loaded data
        logging.info("\nTraining set shapes:")
        logging.info(f"X_property_train: {X_property_train.shape}")
        logging.info(f"X_user_train: {X_user_train.shape}")
        logging.info(f"y_train: {y_train.shape}")

    except Exception as e:
        logging.error("An error occurred during data loading:", exc_info=True)
