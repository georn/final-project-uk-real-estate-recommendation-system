import pandas as pd
import os

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the data directory
data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'historical-data'))

def cleanse_data(input_file, output_file):
    # Define the headers based on the provided breakdown
    headers = ["Unique Transaction Identifier", "Price", "Date of Transaction",
               "Postal Code", "Property Type", "Old/New", "Duration",
               "PAON", "SAON", "Street", "Locality", "Town/City",
               "District", "County", "PPD Category Type", "Record Status"]

    # Construct full paths for input and output files
    input_path = os.path.join(data_dir, input_file)
    output_path = os.path.join(data_dir, output_file)

    print(f"Loading data from: {input_path}")
    # Load the CSV file without headers
    data = pd.read_csv(input_path, header=None, names=headers)

    # Convert Date of Transaction to datetime for filtering
    print("Converting dates and filtering data...")
    data['Date of Transaction'] = pd.to_datetime(data['Date of Transaction'])

    # Filter for properties in Buckinghamshire and from the year 2023
    filtered_data = data[(data['County'].str.upper() == 'BUCKINGHAMSHIRE') &
                         (data['Date of Transaction'].dt.year == 2023)]

    print(f"Number of records after filtering: {len(filtered_data)}")

    print(f"Saving cleaned data to: {output_path}")
    # Save the cleaned data to a new CSV file
    filtered_data.to_csv(output_path, index=False)
    print("Data cleansing process completed successfully.")

if __name__ == "__main__":
    input_csv = 'pp-monthly-update-new-version.csv'
    output_csv = 'buckinghamshire_2023_cleaned_data.csv'
    cleanse_data(input_csv, output_csv)
    print("Data cleansing completed when run as main script.")
