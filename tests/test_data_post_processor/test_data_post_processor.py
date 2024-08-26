from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from src.data_preparation.data_post_processor import process_data, handle_missing_values, encode_categorical_variables, \
    engineer_features, store_processed_data
from src.database.models.processed_property import ProcessedProperty

# Constants
SAMPLE_DATE = '2023-01-15'
SAMPLE_SIZE = '1000 sq ft'
SAMPLE_LATITUDE = 51.5074
SAMPLE_LONGITUDE = -0.1278

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'numeric_col': [1, 2, np.nan, 4, 5],
        'categorical_col': ['A', 'B', np.nan, 'D', 'E'],
        'latitude': [SAMPLE_LATITUDE, np.nan, SAMPLE_LATITUDE, SAMPLE_LATITUDE, np.nan],
        'longitude': [SAMPLE_LONGITUDE, SAMPLE_LONGITUDE, np.nan, SAMPLE_LONGITUDE, np.nan]
    })

class TestHandleMissingValues:
    def test_handle_missing_values(self, sample_df):
        result_df = handle_missing_values(sample_df)

        assert result_df.isnull().sum().sum() == 0, "There should be no missing values in the result"
        assert result_df['numeric_col'].iloc[2] == 3, "Numeric column should be filled with median"
        assert result_df['categorical_col'].iloc[2] == 'Unknown', "Categorical column should be filled with 'Unknown'"
        assert result_df['latitude'].dtype == float, "Latitude should be converted to float"
        assert result_df['longitude'].dtype == float, "Longitude should be converted to float"
        assert not pd.isna(result_df['latitude'].iloc[1]), "Missing latitude should be filled"
        assert not pd.isna(result_df['longitude'].iloc[2]), "Missing longitude should be filled"

    @pytest.mark.skip(reason="Skipping edge cases for now")
    @pytest.mark.parametrize("input_data, expected", [
        ({'numeric_col': [1, np.nan, 3]}, {'numeric_col': [1, 2, 3]}),
        ({'categorical_col': ['A', np.nan, 'C']}, {'categorical_col': ['A', 'Unknown', 'C']}),
    ])
    def test_handle_missing_values_edge_cases(self, input_data, expected):
        df = pd.DataFrame(input_data)
        result = handle_missing_values(df)
        pd.testing.assert_frame_equal(result, pd.DataFrame(expected))

class TestEncodeCategoricalVariables:
    def test_encode_categorical_variables(self):
        df = pd.DataFrame({
            'property_type': ['Detached', 'Semi-detached', 'Flat', 'Detached', 'Terraced'],
            'other_col': [1, 2, 3, 4, 5]
        })

        result_df = encode_categorical_variables(df)

        assert 'Property Type_Detached' in result_df.columns, "Detached should be one-hot encoded"
        assert 'Property Type_Semi-detached' in result_df.columns, "Semi-detached should be one-hot encoded"
        assert 'Property Type_Flat' in result_df.columns, "Flat should be one-hot encoded"
        assert 'Property Type_Terraced' in result_df.columns, "Terraced should be one-hot encoded"
        assert 'property_type' not in result_df.columns, "Original property_type column should be removed"
        assert 'other_col' in result_df.columns, "Other columns should be untouched"
        assert 'location_Urban' in result_df.columns, "Urban location column should be created"
        assert 'location_Suburban' in result_df.columns, "Suburban location column should be created"
        assert 'location_Rural' in result_df.columns, "Rural location column should be created"

class TestEngineerFeatures:
    def test_engineer_features(self):
        df = pd.DataFrame({
            'date': [SAMPLE_DATE, '2023-06-30', '2023-12-25'],
            'size': [SAMPLE_SIZE, '1500 sq ft', '800 sq ft'],
            'epc_rating': ['B', 'D', 'A'],
            'features': ['Garden, Parking', 'Parking', 'Garden, Pool']
        })

        result_df = engineer_features(df)

        assert all(col in result_df.columns for col in ['year', 'month', 'day_of_week']), "Date-related columns should be created"
        assert result_df['year'].tolist() == [2023, 2023, 2023], "Year should be correctly extracted"
        assert result_df['month'].tolist() == [1, 6, 12], "Month should be correctly extracted"
        assert result_df['day_of_week'].tolist() == [6, 4, 0], "Day of week should be correctly calculated"
        assert all(col in result_df.columns for col in ['price_to_income_ratio', 'price_to_savings_ratio', 'affordability_score']), "Affordability features should be added"
        assert result_df['size_sq_ft'].tolist() == [1000.0, 1500.0, 800.0], "Size should be correctly extracted"
        assert result_df['epc_rating_encoded'].tolist() == [5.0, 3.0, 6.0], "EPC rating should be correctly encoded"
        assert result_df['has_garden'].tolist() == [1, 0, 1], "Garden feature should be correctly encoded"
        assert result_df['has_parking'].tolist() == [1, 1, 0], "Parking feature should be correctly encoded"

@patch('src.data_preparation.data_post_processor.SessionLocal')
@patch('src.data_preparation.data_post_processor.store_processed_data')
class TestProcessData:
    def test_process_data(self, mock_store_processed_data, mock_session):
        mock_db = MagicMock()
        mock_session.return_value = mock_db

        mock_data = pd.DataFrame({
            'id': [1, 2, 3],
            'price': [200000, 250000, 300000],
            'size': [SAMPLE_SIZE, '1200 sq ft', '1500 sq ft'],
            'date': [SAMPLE_DATE, '2023-06-30', '2023-12-25'],
            'property_type': ['Detached', 'Semi-detached', 'Flat'],
            'features': ['Garden, Parking', 'Parking', 'Garden'],
            'epc_rating': ['B', 'C', 'A'],
            'latitude': [SAMPLE_LATITUDE, SAMPLE_LATITUDE, SAMPLE_LATITUDE],
            'longitude': [SAMPLE_LONGITUDE, SAMPLE_LONGITUDE, SAMPLE_LONGITUDE]
        })

        mock_db.query.return_value.statement = "SELECT * FROM MergedProperty"
        mock_db.bind = None

        with patch('pandas.read_sql', return_value=mock_data):
            result = process_data()

        assert isinstance(result, pd.DataFrame), "Result should be a DataFrame"
        assert all(col in result.columns for col in ['id', 'price', 'size_sq_ft', 'year', 'month', 'day_of_week', 'has_garden', 'has_parking', 'Property Type_Detached', 'Property Type_Semi-detached', 'Property Type_Flat']), "All expected columns should be present"
        np.testing.assert_almost_equal(result['size_sq_ft'].tolist(), [1000.0, 1200.0, 1500.0], err_msg="Size should be correctly processed")
        assert result['has_garden'].tolist() == [1, 0, 1], "Garden feature should be correctly processed"
        assert result['has_parking'].tolist() == [1, 1, 0], "Parking feature should be correctly processed"
        assert result['Property Type_Detached'].tolist() == [True, False, False], "Property type should be correctly one-hot encoded"
        assert result['Property Type_Semi-detached'].tolist() == [False, True, False], "Property type should be correctly one-hot encoded"
        assert result['Property Type_Flat'].tolist() == [False, False, True], "Property type should be correctly one-hot encoded"

        mock_store_processed_data.assert_called_once()
        mock_db.close.assert_called_once()

class TestStoreProcessedData:
    def test_store_processed_data(self):
        mock_db = MagicMock()

        df = pd.DataFrame({
            'id': [1, 2],
            'price': [200000.0, 250000.0],
            'size_sq_ft': [1000.0, 1200.0],
            'year': [2023, 2023],
            'month': [1, 6],
            'day_of_week': [2, 5],
            'price_to_income_ratio': [np.nan, 4.5],
            'price_to_savings_ratio': [10.0, np.nan],
            'affordability_score': [0.8, 0.7],
            'has_garden': [True, False],
            'has_parking': [False, True],
            'location_Urban': [True, False],
            'location_Suburban': [False, True],
            'location_Rural': [False, False],
            'latitude': [SAMPLE_LATITUDE, SAMPLE_LATITUDE],
            'longitude': [SAMPLE_LONGITUDE, SAMPLE_LONGITUDE],
            'epc_rating_encoded': [5.0, 3.0],
            'Property Type_Detached': [True, False],
            'Property Type_Semi-detached': [False, True]
        })

        store_processed_data(df, mock_db)

        mock_db.query.assert_called_once_with(ProcessedProperty)
        mock_db.query().delete.assert_called_once()
        assert mock_db.add.call_count == 2, "Two rows should be added"

        first_call_args = mock_db.add.call_args_list[0][0][0]
        assert isinstance(first_call_args, ProcessedProperty), "Added object should be a ProcessedProperty"
        assert first_call_args.original_id == 1, "ID should be correctly set"
        assert first_call_args.price == 200000.0, "Price should be correctly set"
        assert first_call_args.size_sq_ft == 1000.0, "Size should be correctly set"
        assert first_call_args.year == 2023, "Year should be correctly set"
        assert first_call_args.price_to_income_ratio is None, "NaN should be converted to None"
        assert first_call_args.price_to_savings_ratio == 10.0, "Price to savings ratio should be correctly set"
        assert first_call_args.has_garden is True, "Garden feature should be correctly set"
        assert first_call_args.has_parking is False, "Parking feature should be correctly set"
        assert first_call_args.location_Urban is True, "Urban location should be correctly set"
        assert first_call_args.latitude == SAMPLE_LATITUDE, "Latitude should be correctly set"
        assert first_call_args.longitude == SAMPLE_LONGITUDE, "Longitude should be correctly set"
        assert first_call_args.epc_rating_encoded == 5.0, "EPC rating should be correctly set"
        assert first_call_args.property_type == 'Unknown', "Property type should be set to Unknown"
        assert first_call_args.additional_features == {'Property Type_Detached': True, 'Property Type_Semi-detached': False}, "Additional features should be correctly set"

        mock_db.commit.assert_called_once()

    def test_store_processed_data_error_handling(self):
        mock_db = MagicMock()
        mock_db.commit.side_effect = Exception("Test exception")

        df = pd.DataFrame({'id': [1]})  # Minimal DataFrame for testing

        with patch('src.data_preparation.data_post_processor.logging.error') as mock_log_error:
            store_processed_data(df, mock_db)

            mock_db.rollback.assert_called_once()
            assert mock_log_error.call_count >= 2, f"At least two error logs should be made, got {mock_log_error.call_count}"
            mock_log_error.assert_any_call("Error committing to database: Test exception")

            # Print out all the error log calls for debugging
            print("All error log calls:")
            for call in mock_log_error.call_args_list:
                print(call)

if __name__ == '__main__':
    pytest.main()
