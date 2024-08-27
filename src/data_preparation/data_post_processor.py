import logging

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.database.database import SessionLocal
from src.database.models.merged_property import MergedProperty
from src.database.models.processed_property import ProcessedProperty

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data():
    """Main function to process the data."""
    logging.info("Loading data from MergedProperty table")

    db = SessionLocal()
    try:
        query = db.query(MergedProperty)
        df = pd.read_sql(query.statement, db.bind)

        logging.info("Initial dataframe info:")
        logging.info(df.info())

        df = handle_missing_values(df)
        df = engineer_features(df)
        df = encode_categorical_variables(df)
        df = scale_selected_features(df)

        final_features = [
            'id', 'price', 'size_sq_ft', 'year', 'month', 'day_of_week',
            'price_to_income_ratio', 'price_to_savings_ratio', 'affordability_score',
            'has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural',
            'latitude', 'longitude', 'epc_rating_encoded',
            'property_type_Detached', 'property_type_Semi-Detached', 'property_type_Terraced',
            'property_type_Flat/Maisonette', 'property_type_Other',
            'bedrooms', 'bathrooms'
        ]

        final_features = [f for f in final_features if f in df.columns]

        logging.info(f"Final features selected: {final_features}")

        df_final = df[final_features]

        logging.info("Final dataframe info:")
        logging.info(df_final.info())

        # Store processed data back to database
        store_processed_data(df_final, db)

        return df_final

    finally:
        db.close()

def handle_missing_values(df):
    """Impute missing values in the dataframe."""
    logging.info("Handling missing values...")

    # Handle numeric columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_features:
        if df[col].isnull().sum() > 0:
            logging.info(f"Imputing missing values in {col} with median")
            df[col] = df[col].fillna(df[col].median())

    # Handle categorical columns
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        if df[col].isnull().sum() > 0:
            logging.info(f"Imputing missing values in {col} with 'Unknown'")
            df[col] = df[col].fillna('Unknown')

    # Handle bedrooms and bathrooms separately
    if 'bedrooms' in df.columns:
        df['bedrooms'] = df['bedrooms'].fillna(df['bedrooms'].median())
    if 'bathrooms' in df.columns:
        df['bathrooms'] = df['bathrooms'].fillna(df['bathrooms'].median())

    # Convert latitude and longitude to float
    df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
    df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

    return df

def engineer_features(df):
    """Create new features from existing data."""
    logging.info("Engineering features...")

    df['date'] = pd.to_datetime(df['date'])
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day_of_week'] = df['date'].dt.dayofweek

    # Add affordability features (these will be used later when combining with user data)
    df['price_to_income_ratio'] = np.nan
    df['price_to_savings_ratio'] = np.nan
    df['affordability_score'] = np.nan

    # Extract numerical value from 'size' column
    df['size_sq_ft'] = df['size'].str.extract('(\d+)').astype(float)

    # Handle EPC rating
    epc_order = ['G', 'F', 'E', 'D', 'C', 'B', 'A']
    df['epc_rating_encoded'] = df['epc_rating'].map({rating: idx for idx, rating in enumerate(epc_order)})
    df['epc_rating_encoded'] = pd.to_numeric(df['epc_rating_encoded'], errors='coerce').astype('Int64')

    # Create binary features for garden and parking
    df['has_garden'] = df['features'].apply(lambda x: 1 if isinstance(x, list) and any('garden' in feat.lower() for feat in x) else 0)
    df['has_parking'] = df['features'].apply(lambda x: 1 if isinstance(x, list) and any('parking' in feat.lower() for feat in x) else 0)

    # Handle bedrooms and bathrooms
    df['bedrooms'] = pd.to_numeric(df['bedrooms'], errors='coerce')
    df['bathrooms'] = pd.to_numeric(df['bathrooms'], errors='coerce')

    return df

def encode_categorical_variables(df):
    """Encode categorical variables for ML models."""
    logging.info("Encoding categorical variables...")

    # One-hot encoding for Property Type
    property_types = ['Detached', 'Semi-Detached', 'Terraced', 'Flat/Maisonette', 'Other']
    for pt in property_types:
        df[f'property_type_{pt}'] = (df['property_type'] == pt).astype(int)

    # Create location type columns if they don't exist
    if 'location_Urban' not in df.columns:
        df['location_Urban'] = 0
    if 'location_Suburban' not in df.columns:
        df['location_Suburban'] = 0
    if 'location_Rural' not in df.columns:
        df['location_Rural'] = 0

    return df

def scale_selected_features(df):
    """Scale only selected numerical features."""
    logging.info("Scaling selected numerical features...")

    features_to_scale = ['year', 'month', 'day_of_week']

    if features_to_scale:
        scaler = StandardScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    return df

def store_processed_data(df, db):
    logging.info("Storing processed data in ProcessedProperty table")

    # Clear existing data
    db.query(ProcessedProperty).delete()

    for _, row in df.iterrows():
        try:
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
                bathrooms=int(row['bathrooms']) if pd.notnull(row['bathrooms']) else None
            )
            db.add(processed_property)
        except Exception as e:
            logging.error(f"Error processing row {row['id']}: {str(e)}")
            logging.error(f"Problematic row data: {row.to_dict()}")
            continue

    try:
        db.commit()
        logging.info("Processed data stored successfully")
    except Exception as e:
        db.rollback()
        logging.error(f"Error committing to database: {str(e)}")
        logging.error("Detailed error information:", exc_info=True)

if __name__ == "__main__":
    processed_data = process_data()
    logging.info("Data processing completed.")
