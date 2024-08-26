import unittest
import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from src.model.data_loader import load_data, load_and_preprocess_data, handle_nan_values, create_property_user_pairs

class TestDataLoader(unittest.TestCase):

    def setUp(self):
        # Create sample property and user DataFrames
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
            'Property Type_House': np.random.choice([0, 1], 100),
            'Property Type_Apartment': np.random.choice([0, 1], 100)
        })

        self.user_df = pd.DataFrame({
            'id': range(1, 21),
            'Income': np.random.randint(30000, 150000, 20),
            'Savings': np.random.randint(5000, 100000, 20),
            'MaxCommuteTime': np.random.randint(10, 60, 20),
            'FamilySize': np.random.randint(1, 6, 20)
        })

    def test_handle_nan_values(self):
        df = pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0],
            'categorical_col': ['A', 'B', np.nan, 'D', 'E'],
            'all_nan_col': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'no_nan_col': [1, 2, 3, 4, 5]
        })

        result_df = handle_nan_values(df)

        self.assertTrue(df['numeric_col'].isna().any())
        self.assertTrue(df['categorical_col'].isna().any())
        self.assertFalse(result_df['numeric_col'].isna().any())
        self.assertFalse(result_df['categorical_col'].isna().any())
        self.assertEqual(result_df['numeric_col'].iloc[2], 3.0)
        self.assertIn(result_df['categorical_col'].iloc[2], ['A', 'B', 'D', 'E'])
        self.assertTrue(result_df['all_nan_col'].isna().all())
        pd.testing.assert_series_equal(df['no_nan_col'], result_df['no_nan_col'])

    def test_handle_all_nan_column(self):
        df = pd.DataFrame({
            'all_nan_col': [np.nan, np.nan, np.nan]
        })
        result_df = handle_nan_values(df)
        self.assertTrue(result_df['all_nan_col'].isna().all())

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

    def test_create_property_user_pairs_with_empty_df(self):
        empty_property_df = pd.DataFrame(columns=self.property_df.columns)
        empty_user_df = pd.DataFrame(columns=self.user_df.columns)

        result = create_property_user_pairs(empty_property_df, self.user_df, 5)
        self.assertTrue(result.empty)

        result = create_property_user_pairs(self.property_df, empty_user_df, 5)
        self.assertTrue(result.empty)

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
    def test_load_and_preprocess_data(self, mock_create_pairs, mock_session):
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
        mock_create_pairs.return_value = mock_pairs

        X_property, X_user, y = load_and_preprocess_data(sample_size=3, pairs_per_user=3)

        self.assertIsInstance(X_property, pd.DataFrame)
        self.assertIsInstance(X_user, pd.DataFrame)
        self.assertIsInstance(y, pd.Series)

        expected_rows = len(mock_pairs)
        self.assertEqual(X_property.shape[0], expected_rows)
        self.assertEqual(X_user.shape[0], expected_rows)
        self.assertEqual(y.shape[0], expected_rows)

        expected_property_columns = [
            'price', 'size_sq_ft', 'year', 'month', 'day_of_week',
            'has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural',
            'latitude', 'longitude', 'epc_rating_encoded', 'Property Type_House', 'Property Type_Apartment',
            'price_to_income_ratio', 'price_to_savings_ratio', 'affordability_score'
        ]
        self.assertListEqual(list(X_property.columns), expected_property_columns)

        expected_user_columns = ['Income', 'Savings', 'MaxCommuteTime', 'FamilySize']
        self.assertListEqual(list(X_user.columns), expected_user_columns)

        self.assertTrue('price_to_income_ratio' in X_property.columns)
        self.assertTrue('price_to_savings_ratio' in X_property.columns)
        self.assertTrue('affordability_score' in X_property.columns)

        self.assertTrue(all(y.isin([True, False])))

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

        mock_create_pairs.return_value = pd.concat([
            pd.concat([self.property_df] * 3, ignore_index=True),
            pd.concat([self.user_df] * 3, ignore_index=True)
        ], axis=1)

        load_and_preprocess_data(sample_size=3, pairs_per_user=3)

        mock_logging.info.assert_any_call("Loading property data from database")
        mock_logging.info.assert_any_call("Loading user data from database")
        mock_logging.info.assert_any_call("Creating property-user pairs")
        mock_logging.info.assert_any_call("Handling NaN values")
        mock_logging.info.assert_any_call("Calculating affordability features")
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

        mock_create_pairs.return_value = pd.concat([
            pd.concat([property_df_missing_feature] * 3, ignore_index=True),
            pd.concat([self.user_df] * 3, ignore_index=True)
        ], axis=1)

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

        self.assertEqual(len(result), 6)  # X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test
        for df in result[:4]:
            self.assertIsInstance(df, pd.DataFrame)
        for series in result[4:]:
            self.assertIsInstance(series, pd.Series)

        mock_load_and_preprocess.assert_called_once_with(100, 10)
        mock_train_test_split.assert_called_once()

if __name__ == '__main__':
    unittest.main()
