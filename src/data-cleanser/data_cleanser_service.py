import pandas as pd
# import csv


# def extract_rows(input_file, num_rows):
#     with open(input_file, mode='r', newline='', encoding='utf-8') as infile:
#         reader = csv.reader(infile)
#         data = [next(reader) for _ in range(num_rows)]  # Extract first 15 rows
#     return data
#
#
# def write_to_new_csv(data, output_file):
#     with open(output_file, mode='w', newline='', encoding='utf-8') as outfile:
#         writer = csv.writer(outfile)
#         writer.writerows(data)


def cleanse_data(input_file, output_file):
    # Define the headers based on the provided breakdown
    headers = ["Unique Transaction Identifier", "Price", "Date of Transaction",
               "Postal Code", "Property Type", "Old/New", "Duration",
               "PAON", "SAON", "Street", "Locality", "Town/City",
               "District", "County", "PPD Category Type", "Record Status"]

    # Load the CSV file without headers
    data = pd.read_csv(input_file, header=None, names=headers)

    # Convert Date of Transaction to datetime for filtering
    print("Converting dates and filtering data...")
    data['Date of Transaction'] = pd.to_datetime(data['Date of Transaction'])

    # Filter for properties in Buckinghamshire and from the year 2023
    data['Date of Transaction'] = pd.to_datetime(data['Date of Transaction'])
    filtered_data = data[(data['County'].str.upper() == 'BUCKINGHAMSHIRE') &
                         (data['Date of Transaction'].dt.year == 2023)]

    print(f"Number of records after filtering: {len(filtered_data)}")

    print("Saving cleaned data to CSV file...")
    # Save the cleaned data to a new CSV file
    filtered_data.to_csv(output_file, index=False)
    print("Data cleansing process completed successfully.")


# File paths
input_csv = '../../data/historical-data/pp-monthly-update-new-version.csv'  # Update with actual path
output_csv = '../../data/historical-data/buckinghamshire_2023_cleaned_data.csv'  # Update with desired output path

cleanse_data(input_csv, output_csv)
