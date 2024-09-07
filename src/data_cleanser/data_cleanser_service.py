import os

import pandas as pd


# Paths and File Handling
def get_full_path(file_name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.abspath(os.path.join(current_dir, '..', '..', 'data', 'historical-data'))
    return os.path.join(data_dir, file_name)


def load_data(input_file, headers):
    input_path = get_full_path(input_file)
    print(f"Loading data from: {input_path}")
    return pd.read_csv(input_path, header=None, names=headers)


def save_data(data, output_file):
    output_path = get_full_path(output_file)
    print(f"Saving cleaned data to: {output_path}")
    data.to_csv(output_path, index=False)
    print("Data saving completed successfully.")


# Standardization
def standardize_county(county):
    county = county.upper()
    if county.startswith('NORTH ') or county.startswith('WEST '):
        return county.split(' ', 1)[1]
    return county


# Cleansing and Filtering Logic
def filter_by_shires_and_year(data, target_shires, year):
    print("Filtering data by shires and year...")
    data['County'] = data['County'].apply(standardize_county)
    filtered_data = data[
        (data['County'].isin([standardize_county(shire) for shire in target_shires])) &
        (data['Date of Transaction'].dt.year == year)
        ]
    return filtered_data


# Date Handling
def process_dates(data):
    print("Converting dates...")
    data['Date of Transaction'] = pd.to_datetime(data['Date of Transaction'])
    return data


# Main Cleansing Function
def cleanse_data(input_file, output_file, target_shires):
    headers = [
        "Unique Transaction Identifier", "Price", "Date of Transaction",
        "Postal Code", "Property Type", "Old/New", "Duration",
        "PAON", "SAON", "Street", "Locality", "Town/City",
        "District", "County", "PPD Category Type", "Record Status"
    ]

    data = load_data(input_file, headers)
    data = process_dates(data)
    filtered_data = filter_by_shires_and_year(data, target_shires, 2023)
    print(f"Number of records after filtering: {len(filtered_data)}")
    save_data(filtered_data, output_file)

    # Print summary of records for each county
    print("\nNumber of records per county:")
    county_counts = filtered_data['County'].value_counts()
    print(county_counts)


if __name__ == "__main__":
    input_csv = 'pp-2023.csv'
    output_csv = 'multi_county_2023_cleaned_data.csv'
    target_shires = [
        'Buckinghamshire', 'Bedford', 'Oxfordshire',
        'North Northamptonshire', 'West Northamptonshire',
        'Hertfordshire', 'West Berkshire'
    ]
    cleanse_data(input_csv, output_csv, target_shires)
    print("Data cleansing completed when run as main script.")
