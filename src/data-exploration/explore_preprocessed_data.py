import pandas as pd

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

if __name__ == "__main__":
    file_path = '../../data/preprocessed-data/preprocessed.csv'
    # file_path = '../../data/ml-ready-data/ml_ready_data.csv'
    explore_preprocessed_data(file_path)
