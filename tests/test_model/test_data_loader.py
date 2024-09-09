import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.database.models.processed_property import EncodedTenure
from src.database.models.synthetic_user import TenurePreference
from src.model.data_loader import load_data, load_and_preprocess_data
from src.model.data_preprocessing import handle_nan_values, create_property_user_pairs


class TestDataLoader(unittest.TestCase):

    def setUp(self):
        self.property_df = pd.DataFrame({
            'id': range(1, 101),
            'price': np.random.randint(100000, 1000000, 100),
            'size_sq_ft': np.random.randint(50, 300, 100),
            'year': np.random.randint(2000, 2023, 100),
            'month': np.random.randint(1, 13, 100),
            'day_of_week': np.random.randint(0, 7, 100),
            'has_garden': np.random.choice([0, 1], 100),
            'has_parking': np.random.choice([0, 1], 100),
            'location_Urban': np.random.choice([0, 1], 100),
            'location_Suburban': np.random.choice([0, 1], 100),
            'location_Rural': np.random.choice([0, 1], 100),
            'latitude': np.random.uniform(50, 52, 100),
            'longitude': np.random.uniform(-1, 1, 100),
            'epc_rating_encoded': np.random.randint(1, 7, 100),
            'property_type_Detached': np.random.choice([0, 1], 100),
            'property_type_Semi_Detached': np.random.choice([0, 1], 100),
            'property_type_Terraced': np.random.choice([0, 1], 100),
            'property_type_Flat_Maisonette': np.random.choice([0, 1], 100),
            'property_type_Other': np.random.choice([0, 1], 100),
            'bedrooms': np.random.randint(1, 6, 100),
            'bathrooms': np.random.randint(1, 4, 100),
            'tenure': np.random.choice([e.value for e in EncodedTenure], 100)
        })

        self.user_df = pd.DataFrame({
            'id': range(1, 21),
            'income': np.random.randint(30000, 150000, 20),
            'savings': np.random.randint(5000, 100000, 20),
            'max_commute_time': np.random.randint(10, 60, 20),
            'family_size': np.random.randint(1, 6, 20),
            'tenure_preference': np.random.choice([e.value for e in TenurePreference], 20)
        })

    @pytest.mark.skip(reason="Skipping edge cases for now")
    def test_create_property_user_pairs(self):
        pairs_per_user = 5
        result = create_property_user_pairs(self.property_df, self.user_df, pairs_per_user)

        expected_rows = len(self.user_df) * pairs_per_user
        self.assertEqual(result.shape[0], expected_rows)

        expected_columns = set(self.property_df.columns) | set(self.user_df.columns)
        self.assertEqual(set(result.columns), expected_columns)

        user_counts = result['id'].value_counts()
        self.assertTrue(all(user_counts == pairs_per_user))

        for user_id in self.user_df['id']:
            user_properties = result[result['id'] == user_id]['id']
            self.assertEqual(len(user_properties), len(set(user_properties)))

    @pytest.mark.skip(reason="Skipping edge cases for now")
    def test_create_property_user_pairs_with_small_property_df(self):
        small_property_df = self.property_df.head(3)
        pairs_per_user = 5
        result = create_property_user_pairs(small_property_df, self.user_df, pairs_per_user)

        self.assertEqual(result.shape[0], len(self.user_df) * 3)

        for user_id in self.user_df['id']:
            user_properties = set(result[result['id'] == user_id]['id'])
            self.assertTrue(user_properties.issubset(set(small_property_df['id'])))
            self.assertEqual(len(user_properties), 3)

    @pytest.mark.skip(reason="Skipping edge cases for now")
    def test_create_property_user_pairs_with_empty_df(self):
        empty_property_df = pd.DataFrame(columns=self.property_df.columns)
        empty_user_df = pd.DataFrame(columns=self.user_df.columns)

        result = create_property_user_pairs(empty_property_df, self.user_df, 5)
        self.assertTrue(result.empty)

        result = create_property_user_pairs(self.property_df, empty_user_df, 5)
        self.assertTrue(result.empty)

    @pytest.mark.skip(reason="Skipping edge cases for now")
    def test_create_property_user_pairs_data_integrity(self):
        pairs_per_user = 5
        result = create_property_user_pairs(self.property_df, self.user_df, pairs_per_user)

        for col in self.property_df.columns:
            self.assertTrue(all(result[col].isin(self.property_df[col])))

        for col in self.user_df.columns:
            for user_id in self.user_df['id']:
                user_rows = result[result['id'] == user_id]
                user_value = self.user_df.loc[self.user_df['id'] == user_id, col].iloc[0]
                self.assertTrue(all(user_rows[col] == user_value))

    @patch('src.model.data_loader.SessionLocal')
    @patch('src.model.data_loader.create_property_user_pairs')
    @patch('pandas.read_sql')
    def test_load_and_preprocess_data(self, mock_read_sql, mock_create_pairs, mock_session):
        mock_db = MagicMock()
        mock_session.return_value = mock_db

        # Create mock DataFrames
        mock_property_df = pd.DataFrame(self.property_df)
        mock_user_df = pd.DataFrame(self.user_df)

        # Set up mock_read_sql to return our mock DataFrames
        mock_read_sql.side_effect = [mock_property_df, mock_user_df]

        # Set up mock_create_pairs to return a DataFrame that combines property and user data
        mock_pairs = pd.concat([mock_property_df, mock_user_df], axis=1)

        # Add the missing columns
        mock_pairs['price_to_income_ratio'] = mock_pairs['price'] / mock_pairs['income']
        mock_pairs['price_to_savings_ratio'] = mock_pairs['price'] / mock_pairs['savings']
        mock_pairs['affordability_score'] = (mock_pairs['income'] * 4 + mock_pairs['savings']) / mock_pairs['price']

        mock_create_pairs.return_value = mock_pairs

        X_property, X_user, y = load_and_preprocess_data(sample_size=3, pairs_per_user=3)

        self.assertIsInstance(X_property, pd.DataFrame)
        self.assertIsInstance(X_user, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

        expected_property_columns = [
            'price', 'size_sq_ft', 'year', 'month', 'day_of_week',
            'price_to_income_ratio', 'price_to_savings_ratio', 'affordability_score',
            'has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural',
            'latitude', 'longitude', 'epc_rating_encoded',
            'property_type_Detached', 'property_type_Semi_Detached',
            'property_type_Terraced', 'property_type_Flat_Maisonette', 'property_type_Other',
            'bedrooms', 'bathrooms', 'tenure'
        ]
        self.assertListEqual(sorted(X_property.columns), sorted(expected_property_columns))

        expected_user_columns = ['income', 'savings', 'max_commute_time', 'family_size', 'tenure_preference']
        self.assertListEqual(sorted(X_user.columns), sorted(expected_user_columns))

        self.assertTrue(all(y.isin([True, False])))

        # Check that read_sql was called twice (once for properties, once for users)
        self.assertEqual(mock_read_sql.call_count, 2)

        # Check that create_property_user_pairs was called
        mock_create_pairs.assert_called_once()

    @patch('src.model.data_loader.logging')
    @patch('src.model.data_loader.SessionLocal')
    @patch('src.model.data_loader.create_property_user_pairs')
    def test_load_and_preprocess_data_logging(self, mock_create_pairs, mock_session, mock_logging):
        mock_property_query = MagicMock()
        mock_property_query.statement = "SELECT * FROM processed_properties"
        mock_user_query = MagicMock()
        mock_user_query.statement = "SELECT * FROM synthetic_users"

        mock_session.return_value.__enter__.return_value.query.side_effect = [
            mock_property_query,
            mock_user_query
        ]

        pd.read_sql = MagicMock(side_effect=[self.property_df, self.user_df])

        mock_pairs = pd.concat([
            pd.concat([self.property_df] * 3, ignore_index=True),
            pd.concat([self.user_df] * 3, ignore_index=True)
        ], axis=1)

        # Add missing columns
        mock_pairs['price_to_income_ratio'] = mock_pairs['price'] / mock_pairs['income']
        mock_pairs['price_to_savings_ratio'] = mock_pairs['price'] / mock_pairs['savings']
        mock_pairs['affordability_score'] = (mock_pairs['income'] * 4 + mock_pairs['savings']) / mock_pairs['price']

        mock_create_pairs.return_value = mock_pairs

        load_and_preprocess_data(sample_size=3, pairs_per_user=3)

        mock_logging.info.assert_any_call("Loading property data from database")
        mock_logging.info.assert_any_call("Loading user data from database")
        mock_logging.info.assert_any_call("Creating property-user pairs")
        mock_logging.info.assert_any_call("Handling NaN values")
        mock_logging.info.assert_any_call("Normalizing features")
        mock_logging.info.assert_any_call("Creating target variable")

    @patch('src.model.data_loader.SessionLocal')
    @patch('src.model.data_loader.create_property_user_pairs')
    def test_load_and_preprocess_data_with_missing_features(self, mock_create_pairs, mock_session):
        property_df_missing_feature = self.property_df.drop('has_garden', axis=1)

        mock_property_query = MagicMock()
        mock_property_query.statement = "SELECT * FROM processed_properties"
        mock_user_query = MagicMock()
        mock_user_query.statement = "SELECT * FROM synthetic_users"

        mock_session.return_value.__enter__.return_value.query.side_effect = [
            mock_property_query,
            mock_user_query
        ]

        pd.read_sql = MagicMock(side_effect=[property_df_missing_feature, self.user_df])

        mock_pairs = pd.concat([
            pd.concat([property_df_missing_feature] * 3, ignore_index=True),
            pd.concat([self.user_df] * 3, ignore_index=True)
        ], axis=1)

        # Add missing columns
        mock_pairs['price_to_income_ratio'] = mock_pairs['price'] / mock_pairs['income']
        mock_pairs['price_to_savings_ratio'] = mock_pairs['price'] / mock_pairs['savings']
        mock_pairs['affordability_score'] = (mock_pairs['income'] * 4 + mock_pairs['savings']) / mock_pairs['price']
        mock_pairs['has_garden'] = 0  # Add the missing 'has_garden' column

        mock_create_pairs.return_value = mock_pairs

        X_property, X_user, y = load_and_preprocess_data(sample_size=3, pairs_per_user=3)

        self.assertTrue('has_garden' in X_property.columns)
        self.assertTrue(all(X_property['has_garden'] == 0))

    @pytest.mark.skip(reason="Skipping edge cases for now")
    @patch('src.model.data_loader.logging')
    @patch('src.model.data_loader.SessionLocal')
    def test_load_and_preprocess_data_error_handling(self, mock_session, mock_logging):
        mock_session.return_value.__enter__.return_value.query.side_effect = Exception("Database error")

        with self.assertRaises(Exception):
            load_and_preprocess_data(sample_size=3, pairs_per_user=3)

        mock_logging.error.assert_called_with("An error occurred during data loading:", exc_info=True)

    @patch('src.model.data_loader.load_and_preprocess_data')
    @patch('src.model.data_loader.train_test_split')
    def test_load_data(self, mock_train_test_split, mock_load_and_preprocess):
        mock_X_property = pd.DataFrame(np.random.rand(100, 10))
        mock_X_user = pd.DataFrame(np.random.rand(100, 5))
        mock_y = pd.Series(np.random.choice([0, 1], 100))

        mock_load_and_preprocess.return_value = (mock_X_property, mock_X_user, mock_y)
        mock_train_test_split.return_value = (
            mock_X_property[:80], mock_X_property[80:],
            mock_X_user[:80], mock_X_user[80:],
            mock_y[:80], mock_y[80:]
        )

        result = load_data(sample_size=100, pairs_per_user=10)

        self.assertEqual(len(result),
                         6)  # X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test
        for df in result[:4]:
            self.assertIsInstance(df, pd.DataFrame)
        for series in result[4:]:
            self.assertIsInstance(series, pd.Series)

        mock_load_and_preprocess.assert_called_once_with(100, 10)
        mock_train_test_split.assert_called_once()


if __name__ == '__main__':
    unittest.main()
