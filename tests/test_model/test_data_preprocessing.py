import unittest

import numpy as np
import pandas as pd
from pandas.testing import assert_series_equal

from src.database.models.synthetic_user import TenurePreference
from src.model.data_preprocessing import (
    handle_nan_values, tenure_preference_to_int, prepare_features,
    create_property_user_pairs, create_target_variable
)


class TestDataPreprocessing(unittest.TestCase):

    def setUp(self):
        # Set up test data
        self.property_df = pd.DataFrame({
            'id': [1, 2, 3],
            'price': [200000, 300000, 250000],
            'size_sq_ft': [1000, np.nan, 1200],
            'year': [2020, 2021, 2022],
            'month': [1, 6, 12],
            'day_of_week': [0, 3, 5],
            'has_garden': [True, False, True],
            'has_parking': [True, True, False],
            'location_Urban': [True, False, False],
            'location_Suburban': [False, True, False],
            'location_Rural': [False, False, True],
            'latitude': [51.5074, 51.5074, 51.5074],
            'longitude': [-0.1278, -0.1278, -0.1278],
            'epc_rating_encoded': [1, 2, 3],
            'property_type_Detached': [True, False, False],
            'property_type_Semi_Detached': [False, True, False],
            'property_type_Terraced': [False, False, True],
            'property_type_Flat_Maisonette': [False, False, False],
            'property_type_Other': [False, False, False],
            'bedrooms': [3, 2, 1],
            'bathrooms': [2, 1, 1],
            'tenure': [0, 1, 2]
        })

        self.user_df = pd.DataFrame({
            'income': [50000, 60000, 70000],
            'savings': [20000, 25000, 30000],
            'max_commute_time': [30, 45, 60],
            'family_size': [2, 3, 4],
            'tenure_preference': [TenurePreference.FREEHOLD, TenurePreference.LEASEHOLD, TenurePreference.NO_PREFERENCE]
        })

    def test_handle_nan_values(self):
        df = pd.DataFrame({
            'numeric_col': [1.0, 2.0, np.nan, 4.0, 5.0],
            'categorical_col': ['A', 'B', np.nan, 'D', 'E'],
            'all_nan_col': [np.nan, np.nan, np.nan, np.nan, np.nan],
            'no_nan_col': [1, 2, 3, 4, 5]
        })

        result_df = handle_nan_values(df)

        # Check if original dataframe has NaNs
        self.assertTrue(df['numeric_col'].isna().any())
        self.assertTrue(df['categorical_col'].isna().any())

        # Check if NaNs are handled in result dataframe
        self.assertFalse(result_df['numeric_col'].isna().any())
        self.assertFalse(result_df['categorical_col'].isna().any())

        # Check if numeric NaN is replaced with median
        self.assertEqual(result_df['numeric_col'].iloc[2], 3.0)

        # Check if categorical NaN is replaced with a valid category
        self.assertIn(result_df['categorical_col'].iloc[2], ['A', 'B', 'D', 'E'])

        # Check if all-NaN column remains all-NaN
        self.assertTrue(result_df['all_nan_col'].isna().all())

        # Check if no-NaN column remains unchanged
        assert_series_equal(df['no_nan_col'], result_df['no_nan_col'])

    def test_handle_all_nan_column(self):
        df = pd.DataFrame({
            'all_nan_col': [np.nan, np.nan, np.nan]
        })
        result_df = handle_nan_values(df)
        self.assertTrue(result_df['all_nan_col'].isna().all())

    def test_tenure_preference_to_int(self):
        self.assertEqual(tenure_preference_to_int(TenurePreference.FREEHOLD), 0)
        self.assertEqual(tenure_preference_to_int(TenurePreference.LEASEHOLD), 1)
        self.assertEqual(tenure_preference_to_int(TenurePreference.NO_PREFERENCE), 2)
        self.assertEqual(tenure_preference_to_int('FREEHOLD'), 0)
        self.assertEqual(tenure_preference_to_int('LEASEHOLD'), 1)
        self.assertEqual(tenure_preference_to_int('NO PREFERENCE'), 2)
        self.assertEqual(tenure_preference_to_int(np.nan), 2)
        self.assertEqual(tenure_preference_to_int(0), 0)
        self.assertEqual(tenure_preference_to_int(1), 1)
        self.assertEqual(tenure_preference_to_int(2), 2)
        self.assertEqual(tenure_preference_to_int(None), 2)
        self.assertEqual(tenure_preference_to_int([]), 2)
        self.assertEqual(tenure_preference_to_int({}), 2)

    def test_prepare_features(self):
        X_property, X_user = prepare_features(self.property_df, self.user_df)

        # Check if all expected features are present
        expected_property_features = [
            'price', 'size_sq_ft', 'year', 'month', 'day_of_week',
            'price_to_income_ratio', 'price_to_savings_ratio', 'affordability_score',
            'has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural',
            'latitude', 'longitude', 'epc_rating_encoded',
            'property_type_Detached', 'property_type_Semi_Detached', 'property_type_Terraced',
            'property_type_Flat_Maisonette', 'property_type_Other',
            'bedrooms', 'bathrooms', 'tenure', 'log_price', 'log_size'
        ]
        for feature in expected_property_features:
            self.assertIn(feature, X_property.columns)

        expected_user_features = ['income', 'savings', 'max_commute_time', 'family_size', 'tenure_preference']
        for feature in expected_user_features:
            self.assertIn(feature, X_user.columns)

        # Check if NaN values are handled
        self.assertFalse(X_property.isnull().any().any())
        self.assertFalse(X_user.isnull().any().any())

        # Check if derived features are correctly calculated
        self.assertTrue((X_property['price_to_income_ratio'] == X_property['price'] / (X_user['income'] + 1)).all())
        self.assertTrue((X_property['price_to_savings_ratio'] == X_property['price'] / (X_user['savings'] + 1)).all())
        self.assertTrue((X_property['affordability_score'] == (X_user['income'] * 4 + X_user['savings']) / (
                X_property['price'] + 1)).all())

        # Check if log transforms are applied
        self.assertTrue((X_property['log_price'] == np.log1p(X_property['price'])).all())
        self.assertTrue((X_property['log_size'] == np.log1p(X_property['size_sq_ft'])).all())

    def test_create_property_user_pairs(self):
        pairs = create_property_user_pairs(self.property_df, self.user_df, pairs_per_user=2)

        self.assertEqual(len(pairs), 6)  # 3 users * 2 pairs each
        self.assertIn('id', pairs.columns)
        self.assertIn('income', pairs.columns)
        self.assertIn('tenure', pairs.columns)
        self.assertIn('tenure_preference', pairs.columns)

    def test_create_target_variable(self):
        X_property, X_user = prepare_features(self.property_df, self.user_df)
        y = create_target_variable(X_property, X_user)

        self.assertIsInstance(y, pd.Series)
        self.assertEqual(len(y), len(X_property))
        self.assertTrue(y.dtype == bool)


if __name__ == '__main__':
    unittest.main()
