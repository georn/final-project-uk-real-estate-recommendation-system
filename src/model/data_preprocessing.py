import logging
import numpy as np
import pandas as pd
from src.database.models.synthetic_user import TenurePreference

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def handle_nan_values(df):
    df = df.copy()
    for col in df.columns:
        if df[col].dtype in [np.float64, np.int64]:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown')
    return df


def tenure_preference_to_int(x):
    if isinstance(x, TenurePreference):
        return {'Freehold': 0, 'Leasehold': 1, 'No Preference': 2}[x.value]
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
    missing_count = X_property['size_sq_ft'].isnull().sum()
    if missing_count > 0:
        # Use property type for more accurate imputation
        X_property['size_sq_ft'] = X_property['size_sq_ft'].fillna(
            X_property.groupby('property_type')['size_sq_ft'].transform('median'))
        still_missing = X_property['size_sq_ft'].isnull().sum()
        if still_missing > 0:
            overall_median = X_property['size_sq_ft'].median()
            X_property['size_sq_ft'] = X_property['size_sq_ft'].fillna(overall_median)
            logging.info(
                f"Imputed {still_missing} remaining missing size_sq_ft values with overall median: {overall_median}")
        logging.info(f"Imputed {missing_count} missing size_sq_ft values based on property type medians")

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

    # Handle county columns
    county_columns = [col for col in X_property.columns if col.startswith('county_')]
    for col in county_columns:
        X_property[col] = X_property[col].fillna(False).astype(bool)

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
            logging.warning(
                f"No matching properties found for tenure preference {user_tenure_pref}, using all properties")

        user_properties = filtered_properties.sample(n=min(pairs_per_user, len(filtered_properties)), replace=False)
        user_dict = user.to_dict()
        user_dict['tenure_preference'] = user_tenure_pref  # Store as integer
        for _, prop in user_properties.iterrows():
            pair = {**prop.to_dict(), **user_dict}
            pairs.append(pair)

    result = pd.DataFrame(pairs)

    # Log statistics about the pairs
    logging.info(f"Created pairs shape: {result.shape}")
    logging.info(f"Unique property count: {result['id'].nunique()}")
    logging.info(f"Unique user count: {result['income'].nunique()}")
    logging.info(f"Sample of created pairs:\n{result[['tenure', 'tenure_preference', 'size_sq_ft']].head()}")

    return result


def create_target_variable(X_property, X_user):
    affordability_condition = X_property['affordability_score'] >= 0.3
    bedroom_condition = X_property['bedrooms'] >= X_user['family_size'] * 0.3
    price_income_condition = X_property['price_to_income_ratio'] <= 7
    size_condition = X_property['size_sq_ft'] >= (X_user['family_size'] * 50)
    tenure_condition = (X_property['tenure'] == X_user['tenure_preference']) | (X_user['tenure_preference'] == 2)

    y = affordability_condition & bedroom_condition & price_income_condition & size_condition & tenure_condition

    logging.info(f"Affordability condition met: {affordability_condition.mean():.2%}")
    logging.info(f"Bedroom condition met: {bedroom_condition.mean():.2%}")
    logging.info(f"Price-to-income condition met: {price_income_condition.mean():.2%}")
    logging.info(f"Size condition met: {size_condition.mean():.2%}")
    logging.info(f"Tenure condition met: {tenure_condition.mean():.2%}")

    positive_ratio = y.mean()
    logging.info(f"Positive samples ratio: {positive_ratio:.2%}")

    return y
