import os
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.data_cleanser.data_cleanser_service import (
    cleanse_data, get_full_path, load_data, process_dates, filter_by_shires_and_year
)


class TestDataCleanserService(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing, without column names
        self.test_data = pd.DataFrame([
            ["A1", 200000, "2023-01-15", "HP1 1AA", "D", "Y", "F", "1", "", "High St", "Town1", "Aylesbury",
             "Aylesbury Vale", "BUCKINGHAMSHIRE", "A", "A"],
            ["B2", 300000, "2022-06-30", "RG1 1BB", "S", "N", "L", "2", "Flat 1", "Main Rd", "Town2", "Reading",
             "Reading", "BERKSHIRE", "B", "A"],
            ["C3", 400000, "2023-12-01", "HP2 2CC", "T", "Y", "F", "3", "", "Park Ave", "Town3", "High Wycombe",
             "Wycombe", "BUCKINGHAMSHIRE", "A", "A"],
            ["D4", 500000, "2023-03-01", "OX1 1DD", "D", "N", "L", "4", "", "Church St", "Town4", "Oxford", "Oxford",
             "OXFORDSHIRE", "B", "A"]
        ])

    def test_get_full_path(self):
        with patch('src.data_cleanser.data_cleanser_service.os.path.abspath') as mock_abspath:
            mock_abspath.return_value = '/mocked/full/path'
            result = get_full_path('file.csv')
            self.assertEqual(result, '/mocked/full/path/file.csv')

    def test_load_data(self):
        headers = [
            "Unique Transaction Identifier", "Price", "Date of Transaction",
            "Postal Code", "Property Type", "Old/New", "Duration",
            "PAON", "SAON", "Street", "Locality", "Town/City",
            "District", "County", "PPD Category Type", "Record Status"
        ]

        with tempfile.TemporaryDirectory() as tmpdirname:
            input_file = os.path.join(tmpdirname, "input.csv")
            self.test_data.to_csv(input_file, index=False, header=False)

            loaded_data = load_data(input_file, headers)
            self.assertEqual(loaded_data.shape, self.test_data.shape)

    def test_process_dates(self):
        # Add headers to the test data to match the expected DataFrame structure
        headers = [
            "Unique Transaction Identifier", "Price", "Date of Transaction",
            "Postal Code", "Property Type", "Old/New", "Duration",
            "PAON", "SAON", "Street", "Locality", "Town/City",
            "District", "County", "PPD Category Type", "Record Status"
        ]

        # Assign headers to the test data
        self.test_data.columns = headers

        # Convert dates to datetime in the test data
        processed_data = process_dates(self.test_data)
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(processed_data['Date of Transaction']))

    def test_filter_by_shires_and_year(self):
        # Add headers to the test data to match the expected DataFrame structure
        headers = [
            "Unique Transaction Identifier", "Price", "Date of Transaction",
            "Postal Code", "Property Type", "Old/New", "Duration",
            "PAON", "SAON", "Street", "Locality", "Town/City",
            "District", "County", "PPD Category Type", "Record Status"
        ]

        # Assign headers to the test data
        self.test_data.columns = headers

        # Ensure the 'Date of Transaction' is processed before filtering
        processed_data = process_dates(self.test_data)

        # Apply filtering logic
        target_shires = ['Buckinghamshire', 'Oxfordshire']
        filtered_data = filter_by_shires_and_year(processed_data, target_shires, 2023)

        # Verify the filtered data
        self.assertEqual(len(filtered_data), 3)  # Adjusting to expect 3 records
        self.assertTrue(all(filtered_data['County'].isin(['BUCKINGHAMSHIRE', 'OXFORDSHIRE'])))

    def test_cleanse_data(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            input_file = os.path.join(tmpdirname, "input.csv")
            output_file = os.path.join(tmpdirname, "output.csv")

            # Save test data to the input file without headers
            self.test_data.to_csv(input_file, index=False, header=False)

            # Run the cleanse_data function
            target_shires = ['Buckinghamshire', 'Oxfordshire']
            cleanse_data(input_file, output_file, target_shires)

            # Read the output file
            output_data = pd.read_csv(output_file)

            # Test assertions
            self.assertEqual(len(output_data),
                             3)  # Should expect 3 rows (2 from Buckinghamshire and 1 from Oxfordshire)
            self.assertTrue(all(output_data['County'].isin(['BUCKINGHAMSHIRE', 'OXFORDSHIRE'])))


if __name__ == '__main__':
    unittest.main()
