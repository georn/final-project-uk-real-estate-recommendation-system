import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def process_data(input_file, output_file):
    """Main function to process the data."""
    logging.info(f"Loading data from {input_file}")
    df = pd.read_csv(input_file)

    logging.info("Initial dataframe info:")
    logging.info(df.info())

    # Handle missing values
    df = handle_missing_values(df)

    # Engineer features
    df = engineer_features(df)

    # Encode categorical variables
    df = encode_categorical_variables(df)

    # Scale only specific numerical features
    df = scale_selected_features(df)

    # Select final features for ML
    final_features = [
        'id', 'price', 'size_sq_ft', 'year', 'month', 'day_of_week',
        'price_to_income_ratio', 'price_to_savings_ratio', 'affordability_score',
        'has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural'
    ]

    if 'latitude' in df.columns:
        final_features.append('latitude')
    if 'longitude' in df.columns:
        final_features.append('longitude')
    if 'epc_rating_encoded' in df.columns:
        final_features.append('epc_rating_encoded')

    property_type_columns = [col for col in df.columns if col.startswith('Property Type_')]
    final_features.extend(property_type_columns)

    # Ensure all final_features are actually in df.columns
    final_features = [f for f in final_features if f in df.columns]

    logging.info(f"Final features selected: {final_features}")

    df_final = df[final_features]

    logging.info("Final dataframe info:")
    logging.info(df_final.info())

    df_final.to_csv(output_file, index=False)
    logging.info(f"Processed data saved to {output_file}")

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

    return df

def engineer_features(df):
    """Create new features from existing data."""
    logging.info("Engineering features...")

    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek

    # Add affordability features (these will be used later when combining with user data)
    df['price_to_income_ratio'] = np.nan
    df['price_to_savings_ratio'] = np.nan
    df['affordability_score'] = np.nan

    if 'size' in df.columns:
        df['size_sq_ft'] = df['size'].str.extract('(\d+)').astype(float)
    else:
        logging.warning("'size' column not found. 'size_sq_ft' feature not created.")

    if 'epc_rating' in df.columns:
        epc_order = ['G', 'F', 'E', 'D', 'C', 'B', 'A']
        df['epc_rating_encoded'] = df['epc_rating'].map({rating: idx for idx, rating in enumerate(epc_order)})
    else:
        logging.warning("'epc_rating' column not found. 'epc_rating_encoded' feature not created.")

    return df

def encode_categorical_variables(df):
    """Encode categorical variables for ML models."""
    logging.info("Encoding categorical variables...")

    # One-hot encoding for Property Type
    if 'Property Type' in df.columns:
        df = pd.get_dummies(df, columns=['Property Type'], prefix='Property Type')
    else:
        logging.warning("'Property Type' not found in dataframe. Skipping one-hot encoding for this feature.")

    # Encode location type
    if 'location_type' in df.columns:
        df = pd.get_dummies(df, columns=['location_type'], prefix='location')
    else:
        logging.warning("'location_type' column not found. Skipping encoding for this feature.")

    return df

def scale_selected_features(df):
    """Scale only selected numerical features."""
    logging.info("Scaling selected numerical features...")

    features_to_scale = ['year', 'month', 'day_of_week']

    if features_to_scale:
        scaler = StandardScaler()
        df[features_to_scale] = scaler.fit_transform(df[features_to_scale])

    return df

if __name__ == "__main__":
    input_file = '../../data/preprocessed-data/preprocessed.csv'
    output_file = '../../data/ml-ready-data/ml_ready_data.csv'
    process_data(input_file, output_file)
    logging.info(f"Data processing completed. Output saved to {output_file}")