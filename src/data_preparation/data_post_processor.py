import logging
import re

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.data_preparation.location_classifier import classify_location
from src.database.database import SessionLocal
from src.database.models.merged_property import MergedProperty, Tenure
from src.database.models.processed_property import ProcessedProperty, EncodedTenure
from src.database.models.synthetic_user import SyntheticUser

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def process_data():
    """Main function to process the data."""
    logging.info("Loading data from MergedProperty table")

    db = SessionLocal()
    try:
        query = db.query(MergedProperty)
        df = pd.read_sql(query.statement, db.bind)

        # Load synthetic user data
        user_query = db.query(SyntheticUser)
        user_df = pd.read_sql(user_query.statement, db.bind)

        logging.info("Initial dataframe info:")
        logging.info(df.info())

        df = handle_missing_values(df)
        logging.info(f"After handle_missing_values - Non-null size_sq_ft: {df['size_sq_ft'].notnull().sum()}")

        df = engineer_features(df)
        logging.info(f"After engineer_features - Non-null size_sq_ft: {df['size_sq_ft'].notnull().sum()}")

        df = calculate_affordability_metrics(df, user_df)
        logging.info(f"After calculate_affordability_metrics - Non-null size_sq_ft: {df['size_sq_ft'].notnull().sum()}")

        df = encode_categorical_variables(df)
        logging.info(f"After encode_categorical_variables - Non-null size_sq_ft: {df['size_sq_ft'].notnull().sum()}")

        df = scale_selected_features(df)
        logging.info(f"After scale_selected_features - Non-null size_sq_ft: {df['size_sq_ft'].notnull().sum()}")

        final_features = [
            'id', 'price', 'size_sq_ft', 'year', 'month', 'day_of_week',
            'price_to_income_ratio', 'price_to_savings_ratio', 'affordability_score',
            'has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural',
            'latitude', 'longitude', 'epc_rating_encoded',
            'property_type_Detached', 'property_type_Semi-Detached', 'property_type_Terraced',
            'property_type_Flat/Maisonette', 'property_type_Other',
            'bedrooms', 'bathrooms', 'tenure', 'price_relative_to_county_avg'
        ]

        # Add county-specific columns to final_features
        county_columns = [col for col in df.columns if col.startswith('county_')]
        final_features.extend(county_columns)

        df_final = df[final_features]

        logging.info("Final dataframe info:")
        logging.info(df_final.info())
        logging.info(f"Number of rows with non-null size_sq_ft: {df_final['size_sq_ft'].notnull().sum()}")
        logging.info(f"size_sq_ft statistics: min={df_final['size_sq_ft'].min()}, max={df_final['size_sq_ft'].max()}, mean={df_final['size_sq_ft'].mean()}, median={df_final['size_sq_ft'].median()}")

        # Store processed data back to database
        store_processed_data(df_final, db)

        return df_final

    finally:
        db.close()


def calculate_affordability_metrics(df, user_df):
    """Calculate affordability metrics using synthetic user data."""
    logging.info("Calculating affordability metrics...")

    # Calculate average income and savings from synthetic user data
    avg_income = user_df['income'].mean()
    avg_savings = user_df['savings'].mean()

    df['price_to_income_ratio'] = df['price'] / avg_income
    df['price_to_savings_ratio'] = df['price'] / avg_savings
    df['affordability_score'] = (avg_income * 4 + avg_savings) / df['price']

    return df


def impute_numeric_columns(df, exclude_columns):
    """Impute missing values in numeric columns using the median."""
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_features:
        if col not in exclude_columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                logging.info(f"Imputed {missing_count} missing values in {col} with median: {median_value}")
    return df


def impute_categorical_columns(df, exclude_columns):
    """Impute missing values in categorical columns with custom handling."""
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        if col not in exclude_columns:
            missing_count = df[col].isnull().sum() + (df[col] == 'Size info not available').sum()
            if missing_count > 0:
                df = impute_categorical_column(df, col, missing_count)
    return df


def impute_categorical_column(df, col, missing_count):
    """Helper function to handle individual categorical columns."""
    if col == 'size':
        df.loc[df['size'] == 'Size info not available', 'size'] = np.nan
    elif col == 'features':
        df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
        logging.info(f"Replaced {missing_count} missing values in {col} with empty list")
    else:
        mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
        df[col] = df[col].fillna(mode_value)
        logging.info(f"Imputed {missing_count} missing values in {col} with mode: {mode_value}")
    return df


def extract_sq_ft(size_str):
    """Extract square footage from a string."""
    if pd.isna(size_str) or size_str == 'Size info not available':
        return np.nan
    match = re.search(r'(\d+(?:\.\d+)?)\s*sq\s*ft', str(size_str))
    result = float(match.group(1)) if match else np.nan
    logging.debug(f"Extracted size: {result} from input: {size_str}")
    return result


def impute_size_and_create_sq_ft(df):
    """Create and impute the 'size_sq_ft' column."""
    logging.info("Starting impute_size_and_create_sq_ft function")
    df['size_sq_ft'] = df['size'].apply(extract_sq_ft)
    missing_count = df['size_sq_ft'].isnull().sum()
    logging.info(f"Initial missing count: {missing_count}")

    if missing_count > 0:
        logging.info("Imputing missing values...")
        # Group by property type and bedrooms, and impute with median
        df['size_sq_ft'] = df.groupby(['property_type', 'bedrooms'])['size_sq_ft'].transform(lambda x: x.fillna(x.median()))

        still_missing = df['size_sq_ft'].isnull().sum()
        logging.info(f"Missing after property type and bedrooms imputation: {still_missing}")

        if still_missing > 0:
            # If there are still missing values, use property type median
            df['size_sq_ft'] = df.groupby('property_type')['size_sq_ft'].transform(lambda x: x.fillna(x.median()))

            final_missing = df['size_sq_ft'].isnull().sum()
            logging.info(f"Missing after property type imputation: {final_missing}")

            if final_missing > 0:
                # If there are still missing values, use overall median
                overall_median = df['size_sq_ft'].median()
                if pd.notna(overall_median):
                    df['size_sq_ft'] = df['size_sq_ft'].fillna(overall_median)
                    logging.info(f"Imputed {final_missing} remaining missing size_sq_ft values with overall median: {overall_median}")
                else:
                    logging.warning("Unable to impute size_sq_ft values. All values are missing or invalid.")

        logging.info(f"Extracted and imputed size_sq_ft values. {missing_count} were initially missing, {df['size_sq_ft'].isnull().sum()} remain missing.")

    logging.info(f"Final size_sq_ft statistics: min={df['size_sq_ft'].min()}, max={df['size_sq_ft'].max()}, mean={df['size_sq_ft'].mean()}, median={df['size_sq_ft'].median()}")
    return df


def impute_specific_columns(df, columns):
    """Impute missing values in specific columns (like bedrooms, bathrooms)."""
    for col in columns:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                logging.info(f"Imputed {missing_count} missing values in {col} with median: {median_value}")
    return df


def impute_lat_long(df, columns):
    """Convert latitude and longitude to float and handle missing values."""
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            logging.info(f"Imputed {missing_count} missing values in {col} with median: {median_value}")
    return df


def handle_missing_values(df):
    """Main function to handle missing values in the dataframe."""
    logging.info("Handling missing values...")

    # Define columns to exclude from imputation
    exclude_columns = ['id', 'listing_id', 'historical_id']

    # Handle numeric columns
    df = impute_numeric_columns(df, exclude_columns)

    # Handle categorical columns
    df = impute_categorical_columns(df, exclude_columns)

    # Handle size and create size_sq_ft
    df = impute_size_and_create_sq_ft(df)

    # Handle specific columns
    df = impute_specific_columns(df, ['bedrooms', 'bathrooms'])

    # Handle latitude and longitude
    df = impute_lat_long(df, ['latitude', 'longitude'])

    logging.info("Missing values handling completed.")
    logging.info(f"Columns with missing values: {df.columns[df.isnull().any()].tolist()}")
    logging.info(f"Missing value counts:\n{df.isnull().sum()}")

    return df


def engineer_features(df):
    """Create new features from existing data."""
    logging.info("Engineering features...")

    df['date'] = pd.to_datetime(df['date'])
    current_date = pd.Timestamp.now()

    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek

    # Calculate days since the property was listed/sold
    df['days_since_date'] = (current_date - df['date']).dt.days

    # Create a categorical feature for listing recency
    df['listing_recency'] = pd.cut(df['days_since_date'],
                                   bins=[0, 1, 7, 14, 30, 60, np.inf],
                                   labels=['Today', 'Last Week', 'Last 2 Weeks', 'Last Month', 'Last 2 Months', 'Older'])

    # Handle EPC rating
    epc_order = ['G', 'F', 'E', 'D', 'C', 'B', 'A']
    df['epc_rating'] = df['epc_rating'].fillna('Unknown')
    df['epc_rating_encoded'] = df['epc_rating'].map({rating: idx for idx, rating in enumerate(epc_order, start=1)})
    df['epc_rating_encoded'] = df['epc_rating_encoded'].fillna(0)
    df['epc_rating_encoded'] = pd.to_numeric(df['epc_rating_encoded'], errors='coerce').astype('Int64')

    # Create binary features for garden and parking
    df['has_garden'] = df['features'].apply(
        lambda x: 1 if isinstance(x, list) and any('garden' in feat.lower() for feat in x) else 0)
    df['has_parking'] = df['features'].apply(
        lambda x: 1 if isinstance(x, list) and any('parking' in feat.lower() for feat in x) else 0)

    # Handle bedrooms and bathrooms
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')

    # Handle missing county information
    if 'county' not in df.columns:
        logging.warning("'county' column not found. Attempting to create it from 'district'.")
        if 'district' in df.columns:
            df['county'] = df['district'].fillna('Unknown')
        else:
            logging.warning("'district' column not found. Setting 'county' to 'Unknown'.")
            df['county'] = 'Unknown'

    # Classify locations
    df[['location_Urban', 'location_Suburban', 'location_Rural']] = df.apply(
        lambda row: pd.Series(classify_location(
            row['latitude'] if pd.notnull(row['latitude']) else None,
            row['longitude'] if pd.notnull(row['longitude']) else None,
            row['county']
        )),
        axis=1
    )

    # One-hot encode the county
    county_dummies = pd.get_dummies(df['county'], prefix='county')
    df = pd.concat([df, county_dummies], axis=1)

    # Calculate county-specific statistics
    df['price_relative_to_county_avg'] = df.groupby('county')['price'].transform(lambda x: x / x.mean())

    logging.info(f"Engineered features: {df.columns.tolist()}")
    return df


def encode_categorical_variables(df):
    """Encode categorical variables for ML models."""
    logging.info("Encoding categorical variables...")

    # One-hot encoding for Property Type
    property_types = ['Detached', 'Semi-Detached', 'Terraced', 'Flat/Maisonette', 'Other']

    # Create columns for each property type, initialized to 0
    for pt in property_types:
        df[f'property_type_{pt}'] = 0

    # Set the appropriate column to 1 based on the property_type
    for pt in property_types:
        if pt == 'Flat/Maisonette':
            df.loc[df['property_type'].isin(['Flat', 'Flat/Maisonette']), f'property_type_{pt}'] = 1
        else:
            df.loc[df['property_type'] == pt, f'property_type_{pt}'] = 1

    # Set 'Other' for any unmatched property types
    df.loc[df[['property_type_' + pt for pt in property_types]].sum(axis=1) == 0, 'property_type_Other'] = 1

    # Create location type columns if they don't exist
    if 'location_Urban' not in df.columns:
        df['location_Urban'] = 0
    if 'location_Suburban' not in df.columns:
        df['location_Suburban'] = 0
    if 'location_Rural' not in df.columns:
        df['location_Rural'] = 0

    # Encode tenure
    tenure_mapping = {
        Tenure.FREEHOLD: EncodedTenure.FREEHOLD,
        Tenure.LEASEHOLD: EncodedTenure.LEASEHOLD,
        Tenure.UNKNOWN: EncodedTenure.UNKNOWN
    }
    df['tenure'] = df['tenure'].map(lambda x: tenure_mapping.get(x, EncodedTenure.UNKNOWN))

    return df


def scale_selected_features(df):
    """Scale only selected numerical features."""
    logging.info("Scaling selected numerical features...")

    features_to_scale = ['year', 'month', 'day_of_week']

    if features_to_scale:
        scaler = StandardScaler()
        scaled_features = scaler.fit_transform(df[features_to_scale])

        # Replace the original values with the scaled values
        for i, feature in enumerate(features_to_scale):
            df[feature] = scaled_features[:, i]

    return df


def store_processed_data(df, db):
    logging.info("Storing processed data in ProcessedProperty table")
    logging.info(f"Number of rows with non-null size_sq_ft: {df['size_sq_ft'].notnull().sum()}")
    logging.info(f"size_sq_ft statistics: min={df['size_sq_ft'].min()}, max={df['size_sq_ft'].max()}, mean={df['size_sq_ft'].mean()}, median={df['size_sq_ft'].median()}")

    # Clear existing data
    db.query(ProcessedProperty).delete()

    for _, row in df.iterrows():
        processed_property = ProcessedProperty(
            original_id=int(row['id']),
            price=float(row['price']) if pd.notnull(row['price']) else None,
            size_sq_ft=float(row['size_sq_ft']) if pd.notnull(row['size_sq_ft']) else None,
            year=int(row['year']) if pd.notnull(row['year']) else None,
            month=int(row['month']) if pd.notnull(row['month']) else None,
            day_of_week=int(row['day_of_week']) if pd.notnull(row['day_of_week']) else None,
            price_to_income_ratio=float(row['price_to_income_ratio']) if pd.notnull(row['price_to_income_ratio']) else None,
            price_to_savings_ratio=float(row['price_to_savings_ratio']) if pd.notnull(row['price_to_savings_ratio']) else None,
            affordability_score=float(row['affordability_score']) if pd.notnull(row['affordability_score']) else None,
            has_garden=bool(row['has_garden']),
            has_parking=bool(row['has_parking']),
            location_Urban=bool(row['location_Urban']),
            location_Suburban=bool(row['location_Suburban']),
            location_Rural=bool(row['location_Rural']),
            latitude=float(row['latitude']) if pd.notnull(row['latitude']) else None,
            longitude=float(row['longitude']) if pd.notnull(row['longitude']) else None,
            epc_rating_encoded=int(row['epc_rating_encoded']) if pd.notnull(row['epc_rating_encoded']) else None,
            property_type_Detached=bool(row['property_type_Detached']),
            property_type_Semi_Detached=bool(row['property_type_Semi-Detached']),
            property_type_Terraced=bool(row['property_type_Terraced']),
            property_type_Flat_Maisonette=bool(row['property_type_Flat/Maisonette']),
            property_type_Other=bool(row['property_type_Other']),
            bedrooms=int(row['bedrooms']) if pd.notnull(row['bedrooms']) else None,
            bathrooms=int(row['bathrooms']) if pd.notnull(row['bathrooms']) else None,
            tenure=EncodedTenure(row['tenure']) if pd.notnull(row['tenure']) else EncodedTenure.UNKNOWN,
            price_relative_to_county_avg=float(row['price_relative_to_county_avg']) if pd.notnull(row['price_relative_to_county_avg']) else None
        )

        # Add county-specific columns
        for col in df.columns:
            if col.startswith('county_'):
                setattr(processed_property, col, bool(row[col]))

        db.add(processed_property)

    try:
        db.commit()
        logging.info("Processed data stored successfully")
    except Exception as e:
        db.rollback()
        logging.error(f"Error committing to database: {str(e)}")


if __name__ == "__main__":
    processed_data = process_data()
    logging.info("Data processing completed.")
