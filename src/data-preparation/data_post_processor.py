import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, RobustScaler

def load_data(file_path):
    """Load the preprocessed data from CSV."""
    return pd.read_csv(file_path)

def print_column_info(df):
    """Print information about each column in the dataframe."""
    for col in df.columns:
        non_null_count = df[col].count()
        dtype = df[col].dtype
        unique_count = df[col].nunique()
        print(f"Column: {col}, Non-null count: {non_null_count}, dtype: {dtype}, Unique values: {unique_count}")

def handle_missing_values(df):
    """Impute missing values in the dataframe."""
    print("Handling missing values...")

    # Handle numeric columns
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_features:
        if df[col].isnull().all():
            print(f"Column {col} is entirely null. Dropping this column.")
            df = df.drop(columns=[col])
        elif df[col].isnull().any():
            print(f"Imputing missing values in {col} with median")
            df[col] = df[col].fillna(df[col].median())

    # Handle categorical columns
    categorical_features = df.select_dtypes(include=['object']).columns
    for col in categorical_features:
        if df[col].isnull().any():
            print(f"Imputing missing values in {col} with 'Unknown'")
            df[col] = df[col].fillna('Unknown')

    return df

def engineer_features(df):
    """Create new features from existing data."""
    print("Engineering features...")
    df['Date'] = pd.to_datetime(df['Date'])
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day_of_week'] = df['Date'].dt.dayofweek

    if 'size' in df.columns:
        df['size_sq_ft'] = df['size'].str.extract('(\d+)').astype(float)
    else:
        print("Warning: 'size' column not found. Skipping size_sq_ft feature.")

    if 'epc_rating' in df.columns:
        df['has_epc_rating'] = df['epc_rating'].notna().astype(int)
    else:
        print("Warning: 'epc_rating' column not found. Skipping has_epc_rating feature.")

    # Add features to capture "outlier" nature without removing data
    if 'price' in df.columns:
        df['price_percentile'] = df['price'].rank(pct=True)
    if 'size_sq_ft' in df.columns:
        df['size_percentile'] = df['size_sq_ft'].rank(pct=True)

    return df

def encode_categorical_variables(df):
    """Encode categorical variables for ML models."""
    print("Encoding categorical variables...")

    # One-hot encoding for Property Type
    if 'Property Type' in df.columns:
        df = pd.get_dummies(df, columns=['Property Type'], prefix='Property Type')
    else:
        print("Warning: 'Property Type' not found in dataframe. Skipping one-hot encoding for this feature.")

    # Binary encoding for source
    if 'source' in df.columns:
        print("\nDebugging 'source' column:")
        print(f"Unique values in 'source': {df['source'].unique()}")
        print(f"Value counts of 'source':\n{df['source'].value_counts(normalize=True)}")

        df['is_scraped'] = df['source'] == 'scraped'

        print("\nAfter encoding:")
        print(f"Unique values in 'is_scraped': {df['is_scraped'].unique()}")
        print(f"Distribution of 'is_scraped':\n{df['is_scraped'].value_counts(normalize=True)}")
        print(f"First few rows of 'is_scraped':\n{df[['source', 'is_scraped']].head(10)}")

        df = df.drop(columns=['source'])
    else:
        print("Warning: 'source' column not found. Skipping encoding for this feature.")

    # Ordinal encoding for EPC rating
    if 'epc_rating' in df.columns:
        epc_order = ['G', 'F', 'E', 'D', 'C', 'B', 'A']
        df['epc_rating_encoded'] = df['epc_rating'].map({rating: idx for idx, rating in enumerate(epc_order)})
    else:
        print("Warning: 'epc_rating' column not found. Skipping ordinal encoding for EPC rating.")

    return df

def scale_numerical_features(df):
    """Scale numerical features using robust methods."""
    print("Scaling numerical features...")

    # Use RobustScaler for all numerical features
    num_features = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    num_features = [f for f in num_features if f not in ['id', 'year', 'month', 'day_of_week']]

    if num_features:
        robust_scaler = RobustScaler()
        df[num_features] = robust_scaler.fit_transform(df[num_features])

    return df

def process_data(input_file, output_file):
    """Main function to process the data."""
    print(f"Loading data from {input_file}")
    df = load_data(input_file)

    print("Initial dataframe info:")
    print(df.info())
    print_column_info(df)

    df = handle_missing_values(df)
    df = engineer_features(df)
    df = encode_categorical_variables(df)
    df = scale_numerical_features(df)

    # Select final features for ML
    final_features = ['id', 'price', 'year', 'month', 'day_of_week', 'price_percentile']
    if 'size_sq_ft' in df.columns:
        final_features.extend(['size_sq_ft', 'size_percentile'])
    if 'latitude' in df.columns:
        final_features.append('latitude')
    if 'longitude' in df.columns:
        final_features.append('longitude')
    if 'has_epc_rating' in df.columns:
        final_features.append('has_epc_rating')
    if 'epc_rating_encoded' in df.columns:
        final_features.append('epc_rating_encoded')

    property_type_columns = [col for col in df.columns if col.startswith('Property Type_')]
    final_features.extend(property_type_columns)

    if 'is_scraped' in df.columns:
        final_features.append('is_scraped')

    df_final = df[final_features]

    print("Final dataframe info:")
    print(df_final.info())
    print_column_info(df_final)

    df_final.to_csv(output_file, index=False)
    print(f"Processed data saved to {output_file}")

    # Verify the data
    df_verification = pd.read_csv(output_file)
    print("\nVerification of saved data types:")
    print(df_verification.dtypes)
    print("\nDistribution of 'is_scraped' in saved data:")
    print(df_verification['is_scraped'].value_counts(normalize=True))

if __name__ == "__main__":
    input_file = '../../data/preprocessed-data/preprocessed.csv'
    output_file = '../../data/ml-ready-data/ml_ready_data.csv'
    process_data(input_file, output_file)