import pandas as pd
import json
import os

def explore_preprocessed_data(file_path):
    # Load the preprocessed CSV file
    data = pd.read_csv(file_path)

    # Set display options to show all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    # Print column names
    print("Column Names:\n")
    print(data.columns)
    print("\n")

    # Print the first 5 rows
    print("First 5 Rows:\n")
    print(data.head())
    print("\n")

    # Print the last 5 rows
    print("Last 5 Rows:\n")
    print(data.tail())
    print("\n")

    # Reset display options to default
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')

def explore_cleaned_data(file_path):
    # Load the cleaned CSV file
    data = pd.read_csv(file_path)

    # Set display options to show all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)

    # Print column names
    print("Column Names:\n")
    print(data.columns)
    print("\n")

    # Print the first 5 rows
    print("First 5 Rows:\n")
    print(data.head())
    print("\n")

    # Print data types of each column
    print("Data Types:\n")
    print(data.dtypes)
    print("\n")

    # Print summary statistics
    print("Summary Statistics:\n")
    print(data.describe(include='all'))
    print("\n")

    # Print unique values in categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        print(f"Unique values in {col}:")
        print(data[col].unique())
        print("\n")

    # Reset display options to default
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')
    pd.reset_option('display.width')

def explore_scraped_data(file_path):
    # Determine file type and load accordingly
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = pd.DataFrame(json.load(f))
    else:
        raise ValueError("Unsupported file format. Please use CSV or JSON.")

    # Set display options to show all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    pd.set_option('display.width', 1000)

    # Print column names
    print("Column Names:\n")
    print(data.columns)
    print("\n")

    # Print the first 5 rows
    print("First 5 Rows:\n")
    print(data.head())
    print("\n")

    # Print data types of each column
    print("Data Types:\n")
    print(data.dtypes)
    print("\n")

    # Print summary statistics
    print("Summary Statistics:\n")
    print(data.describe(include='all'))
    print("\n")

    # Print unique values in categorical columns
    categorical_columns = data.select_dtypes(include=['object']).columns
    for col in categorical_columns:
        print(f"Unique values in {col}:")
        unique_values = data[col].unique()
        print(unique_values[:10] if len(unique_values) > 10 else unique_values)  # Limit to first 10 if there are many
        print(f"Total unique values: {len(unique_values)}")
        print("\n")

    # Reset display options to default
    pd.reset_option('display.max_columns')
    pd.reset_option('display.max_rows')
    pd.reset_option('display.width')

if __name__ == "__main__":
    # file_path = '../../data/preprocessed-data/preprocessed.csv'
    # file_path = '../../data/ml-ready-data/ml_ready_data.csv'
    # file_path = '../../data/historical-data/buckinghamshire_2023_cleaned_data.csv'
    file_path = '../../data/property_data_650000.json'
    if os.path.exists(file_path):
        explore_scraped_data(file_path)
    else:
        print(f"File not found: {file_path}")
