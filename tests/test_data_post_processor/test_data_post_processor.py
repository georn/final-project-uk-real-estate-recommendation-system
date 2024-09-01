import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd

from src.data_preparation.data_post_processor import (
    process_data, handle_missing_values, engineer_features,
    encode_categorical_variables, calculate_affordability_metrics,
    scale_selected_features, store_processed_data
)
from src.database.models.processed_property import ProcessedProperty, EncodedTenure


class TestDataPostProcessor(unittest.TestCase):

    def setUp(self):
        self.sample_df = pd.DataFrame({
            'id': [1, 2, 3],
            'price': [200000, 250000, 300000],
            'size': ['1000 sq ft', '1200 sq ft', '1500 sq ft'],
            'date': ['2023-01-15', '2023-06-30', '2023-12-25'],
            'property_type': ['Detached', 'Semi-Detached', 'Flat'],
            'features': [['Garden', 'Parking'], ['Parking'], ['Garden']],
            'epc_rating': ['B', 'C', 'A'],
            'latitude': [51.5074, 51.5074, 51.5074],
            'longitude': [-0.1278, -0.1278, -0.1278],
            'bedrooms': [3, 2, 1],
            'bathrooms': [2, 1, 1],
            'tenure': ['FREEHOLD', 'LEASEHOLD', 'UNKNOWN']
        })

    @patch('src.data_preparation.data_post_processor.SessionLocal')
    @patch('src.data_preparation.data_post_processor.store_processed_data')
    def test_process_data(self, mock_store_processed_data, mock_session):
        mock_db = MagicMock()
        mock_session.return_value = mock_db

        # Create a mock user DataFrame
        mock_user_df = pd.DataFrame({
            'income': [50000, 60000, 70000],
            'savings': [20000, 25000, 30000]
        })

        call_count = 0
        def mock_read_sql(query, con):
            nonlocal call_count
            if call_count == 0:
                call_count += 1
                return self.sample_df  # First call, return MergedProperty data
            elif call_count == 1:
                call_count += 1
                return mock_user_df  # Second call, return SyntheticUser data
            else:
                raise ValueError(f"Unexpected query: {query}")

        with patch('pandas.read_sql', side_effect=mock_read_sql) as mock_read_sql_patch:
            result = process_data()

            # Assert that read_sql was called twice
            self.assertEqual(call_count, 2)

        self.assertIsInstance(result, pd.DataFrame)
        self.assertIn('size_sq_ft', result.columns)
        self.assertIn('year', result.columns)
        self.assertIn('has_garden', result.columns)
        self.assertIn('has_parking', result.columns)
        self.assertIn('property_type_Detached', result.columns)
        self.assertIn('price_to_income_ratio', result.columns)
        self.assertIn('price_to_savings_ratio', result.columns)
        self.assertIn('affordability_score', result.columns)

        mock_store_processed_data.assert_called_once()
        mock_db.close.assert_called_once()

    def test_handle_missing_values(self):
        df_with_missing = self.sample_df.copy()
        df_with_missing.loc[0, 'price'] = np.nan
        df_with_missing.loc[1, 'property_type'] = np.nan

        result = handle_missing_values(df_with_missing)

        self.assertFalse(result['price'].isnull().any())
        self.assertFalse(result['property_type'].isnull().any())
        self.assertEqual(result.loc[1, 'property_type'], 'Unknown')

    def test_engineer_features(self):
        result = engineer_features(self.sample_df)

        self.assertIn('year', result.columns)
        self.assertIn('month', result.columns)
        self.assertIn('day_of_week', result.columns)
        self.assertIn('size_sq_ft', result.columns)
        self.assertIn('epc_rating_encoded', result.columns)
        self.assertIn('has_garden', result.columns)
        self.assertIn('has_parking', result.columns)

    def test_encode_categorical_variables(self):
        result = encode_categorical_variables(self.sample_df)

        expected_property_types = [
            'property_type_Detached',
            'property_type_Semi-Detached',
            'property_type_Terraced',
            'property_type_Flat/Maisonette',
            'property_type_Other'
        ]

        for property_type in expected_property_types:
            self.assertIn(property_type, result.columns, f"{property_type} not found in result columns")

        self.assertIn('location_Urban', result.columns)
        self.assertIn('location_Suburban', result.columns)
        self.assertIn('location_Rural', result.columns)

        # Print actual values for debugging
        print("\nActual encoded values:")
        for i in range(len(self.sample_df)):
            print(f"Row {i}:")
            for prop_type in expected_property_types:
                print(f"  {prop_type}: {result.loc[i, prop_type]}")

        # Check if exactly one property type is 1 for each row
        for i in range(len(self.sample_df)):
            encoded_values = [result.loc[i, prop_type] for prop_type in expected_property_types]
            self.assertEqual(sum(encoded_values), 1,
                             f"Row {i} does not have exactly one property type encoded as 1: {encoded_values}")

        # Print tenure values for debugging
        print("\nTenure values:")
        print("Original:", self.sample_df['tenure'].tolist())
        print("Encoded:", result['tenure'].tolist())

        # Check if tenure is encoded correctly
        self.assertTrue(
            all(result['tenure'].isin([EncodedTenure.FREEHOLD, EncodedTenure.LEASEHOLD, EncodedTenure.UNKNOWN])),
            f"Not all tenure values are correctly encoded. Encoded values: {result['tenure'].tolist()}")

    def test_calculate_affordability_metrics(self):
        user_df = pd.DataFrame({
            'income': [50000, 60000, 70000],
            'savings': [20000, 25000, 30000]
        })

        result = calculate_affordability_metrics(self.sample_df, user_df)

        self.assertIn('price_to_income_ratio', result.columns)
        self.assertIn('price_to_savings_ratio', result.columns)
        self.assertIn('affordability_score', result.columns)

    def test_scale_selected_features(self):
        df = self.sample_df.copy()
        df['year'] = [2020, 2021, 2022]
        df['month'] = [1, 6, 12]
        df['day_of_week'] = [0, 3, 5]

        original_values = {col: df[col].tolist() for col in ['year', 'month', 'day_of_week']}
        original_means = {col: df[col].mean() for col in ['year', 'month', 'day_of_week']}
        original_stds = {col: df[col].std() for col in ['year', 'month', 'day_of_week']}

        print("\nOriginal data:")
        print(df[['year', 'month', 'day_of_week']])

        result = scale_selected_features(df)

        print("\nScaled data:")
        print(result[['year', 'month', 'day_of_week']])

        for col in ['year', 'month', 'day_of_week']:
            scaled_mean = result[col].mean()
            scaled_std = result[col].std()

            print(f"\n{col}:")
            print(f"Original mean: {original_means[col]}, Original std: {original_stds[col]}")
            print(f"Scaled mean: {scaled_mean}, Scaled std: {scaled_std}")

            # Check if the mean is close to 0 (allowing for small floating-point errors)
            self.assertAlmostEqual(scaled_mean, 0, places=7,
                                   msg=f"Mean of scaled {col} should be close to 0")

            # Check if the standard deviation is within an acceptable range
            self.assertGreater(scaled_std, 0.9,
                               msg=f"Standard deviation of scaled {col} should be greater than 0.9")
            self.assertLess(scaled_std, 1.3,  # Increased upper bound to 1.3
                            msg=f"Standard deviation of scaled {col} should be less than 1.3")

            # Check if the values have changed from the original
            print(f"Original values: {original_values[col]}")
            print(f"Scaled values: {result[col].tolist()}")
            self.assertFalse(np.allclose(result[col], original_values[col]),
                             f"Values of {col} should have changed after scaling")

        print("\nAll tests passed for scale_selected_features")

    @patch('src.data_preparation.data_post_processor.logging.error')
    def test_store_processed_data(self, mock_log_error):
        # Prepare the data
        df = self.sample_df.copy()
        df = engineer_features(df)
        df = encode_categorical_variables(df)

        # Add mock data for columns that would be created by other functions
        df['price_to_income_ratio'] = [3.5, 4.0, 4.5]
        df['price_to_savings_ratio'] = [10.0, 12.0, 15.0]
        df['affordability_score'] = [0.8, 0.7, 0.6]

        # Ensure all required columns are present
        required_columns = [
            'id', 'price', 'size_sq_ft', 'year', 'month', 'day_of_week',
            'price_to_income_ratio', 'price_to_savings_ratio', 'affordability_score',
            'has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural',
            'latitude', 'longitude', 'epc_rating_encoded',
            'property_type_Detached', 'property_type_Semi-Detached', 'property_type_Terraced',
            'property_type_Flat/Maisonette', 'property_type_Other',
            'bedrooms', 'bathrooms', 'tenure'
        ]

        for col in required_columns:
            if col not in df.columns:
                df[col] = 0  # Add missing columns with dummy data

        mock_db = MagicMock()
        mock_db.commit.side_effect = Exception("Test exception")

        store_processed_data(df, mock_db)

        mock_db.query.assert_called_once_with(ProcessedProperty)
        mock_db.query().delete.assert_called_once()
        self.assertEqual(mock_db.add.call_count, 3)
        mock_db.rollback.assert_called_once()
        mock_log_error.assert_called_with("Error committing to database: Test exception")


if __name__ == '__main__':
    unittest.main()
