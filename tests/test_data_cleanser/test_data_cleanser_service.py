import unittest
import pandas as pd
import os
import tempfile
from src.data_cleanser.data_cleanser_service import cleanse_data

class TestDataCleanserService(unittest.TestCase):

    def setUp(self):
        # Create a sample DataFrame for testing, without column names
        self.test_data = pd.DataFrame([
            ["A1", 200000, "2023-01-15", "HP1 1AA", "D", "Y", "F", "1", "", "High St", "Town1", "Aylesbury", "Aylesbury Vale", "BUCKINGHAMSHIRE", "A", "A"],
            ["B2", 300000, "2022-06-30", "RG1 1BB", "S", "N", "L", "2", "Flat 1", "Main Rd", "Town2", "Reading", "Reading", "BERKSHIRE", "B", "A"],
            ["C3", 400000, "2023-12-01", "HP2 2CC", "T", "Y", "F", "3", "", "Park Ave", "Town3", "High Wycombe", "Wycombe", "BUCKINGHAMSHIRE", "A", "A"],
            ["D4", 500000, "2023-03-01", "OX1 1DD", "D", "N", "L", "4", "", "Church St", "Town4", "Oxford", "Oxford", "OXFORDSHIRE", "B", "A"]
        ])

    def test_cleanse_data(self):
        with tempfile.TemporaryDirectory() as tmpdirname:
            # Create temporary input and output files
            input_file = os.path.join(tmpdirname, "input.csv")
            output_file = os.path.join(tmpdirname, "output.csv")

            # Save test data to the input file without headers
            self.test_data.to_csv(input_file, index=False, header=False)

            # Run the cleanse_data function
            cleanse_data(input_file, output_file)

            # Read the output file
            output_data = pd.read_csv(output_file)

            # Test assertions
            self.assertEqual(len(output_data), 2)  # Should only have 2 rows (Buckinghamshire in 2023)
            self.assertTrue(all(output_data['County'] == 'BUCKINGHAMSHIRE'))
            self.assertTrue(all(pd.to_datetime(output_data['Date of Transaction']).dt.year == 2023))

            # Check if all expected columns are present
            expected_columns = [
                "Unique Transaction Identifier", "Price", "Date of Transaction",
                "Postal Code", "Property Type", "Old/New", "Duration",
                "PAON", "SAON", "Street", "Locality", "Town/City",
                "District", "County", "PPD Category Type", "Record Status"
            ]
            self.assertTrue(all(column in output_data.columns for column in expected_columns))

if __name__ == '__main__':
    unittest.main()
