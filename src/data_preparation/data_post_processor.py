import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.database.database import SessionLocal
from src.database.models.merged_property import MergedProperty, Tenure
from src.database.models.processed_property import ProcessedProperty, EncodedTenure
from src.database.models.synthetic_user import SyntheticUser

from src.data_preparation.location_classifier import classify_location

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
        df = engineer_features(df)
        df = calculate_affordability_metrics(df, user_df)
        df = encode_categorical_variables(df)
        df = scale_selected_features(df)

        final_features = [
            'id', 'price', 'size_sq_ft', 'year', 'month', 'day_of_week',
            'price_to_income_ratio', 'price_to_savings_ratio', 'affordability_score',
            'has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural',
            'latitude', 'longitude', 'epc_rating_encoded',
            'property_type_Detached', 'property_type_Semi-Detached', 'property_type_Terraced',
            'property_type_Flat/Maisonette', 'property_type_Other',
            'bedrooms', 'bathrooms', 'tenure'
        ]

        df_final = df[final_features]

        logging.info("Final dataframe info:")
        logging.info(df_final.info())

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

#TODO: default size value
def handle_missing_values(df):
    """Impute missing values in the dataframe."""
    logging.info("Handling missing values...")

    # Columns to exclude from imputation
    exclude_columns = ['id', 'listing_id', 'historical_id']

    # Handle numeric columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_features:
        if col not in exclude_columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                logging.info(f"Imputed {missing_count} missing values in {col} with median: {median_value}")

    # Handle categorical columns
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        if col not in exclude_columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                if col == 'size':
                    df.loc[df['size'] == 'Size info not available', 'size'] = np.nan
                    df.loc[df['size'] == 'Size info not available', 'size'] = np.nan
                elif col == 'features':
                    df[col] = df[col].apply(lambda x: x if isinstance(x, list) else [])
                    logging.info(f"Replaced {missing_count} missing values in {col} with empty list")
                else:
                    mode_value = df[col].mode().iloc[0] if not df[col].mode().empty else 'Unknown'
                    df[col] = df[col].fillna(mode_value)
                    logging.info(f"Imputed {missing_count} missing values in {col} with mode: {mode_value}")

    # Handle size_sq_ft
    df['size_sq_ft'] = df['size'].str.extract('(\d+)').astype(float)
    missing_count = df['size_sq_ft'].isnull().sum()
    if missing_count > 0:
        median_size = df['size_sq_ft'].median()
        df['size_sq_ft'] = df['size_sq_ft'].fillna(median_size)
        logging.info(f"Imputed {missing_count} missing size_sq_ft values with median: {median_size}")

    # If there are still NaN values (in case median was NaN), use a default value
    if df['size_sq_ft'].isnull().sum() > 0:
        default_size = 1000  # You can adjust this value based on your data
        df['size_sq_ft'] = df['size_sq_ft'].fillna(default_size)
        logging.info(f"Imputed remaining missing size_sq_ft values with default: {default_size}")

    # Handle bedrooms and bathrooms separately
    for col in ['bedrooms', 'bathrooms']:
        if col in df.columns:
            missing_count = df[col].isnull().sum()
            if missing_count > 0:
                median_value = df[col].median()
                df[col] = df[col].fillna(median_value)
                logging.info(f"Imputed {missing_count} missing values in {col} with median: {median_value}")

    # Convert latitude and longitude to float and handle missing values
    for col in ['latitude', 'longitude']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        missing_count = df[col].isnull().sum()
        if missing_count > 0:
            median_value = df[col].median()
            df[col] = df[col].fillna(median_value)
            logging.info(f"Imputed {missing_count} missing values in {col} with median: {median_value}")

    return df

def engineer_features(df):
    """Create new features from existing data."""
    logging.info("Engineering features...")

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek

    # Extract numerical value from 'size' column
    df['size_sq_ft'] = df['size'].str.extract('(\d+)').astype(float)

    # Handle EPC rating
    epc_order = ['G', 'F', 'E', 'D', 'C', 'B', 'A']
    df['epc_rating'] = df['epc_rating'].fillna('Unknown')
    df['epc_rating_encoded'] = df['epc_rating'].map({rating: idx for idx, rating in enumerate(epc_order, start=1)})
    df['epc_rating_encoded'] = df['epc_rating_encoded'].fillna(0)
    df['epc_rating_encoded'] = pd.to_numeric(df['epc_rating_encoded'], errors='coerce').astype('Int64')

    # Create binary features for garden and parking
    df['has_garden'] = df['features'].apply(lambda x: 1 if isinstance(x, list) and any('garden' in feat.lower() for feat in x) else 0)
    df['has_parking'] = df['features'].apply(lambda x: 1 if isinstance(x, list) and any('parking' in feat.lower() for feat in x) else 0)

    # Handle bedrooms and bathrooms
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')

    # Classify locations
    df[['location_Urban', 'location_Suburban', 'location_Rural']] = df.apply(
        lambda row: pd.Series(classify_location(row['latitude'], row['longitude'])),
        axis=1
    )

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
            tenure=EncodedTenure(row['tenure']) if pd.notnull(row['tenure']) else EncodedTenure.UNKNOWN
        )
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
