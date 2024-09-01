import pytest
from unittest.mock import patch, MagicMock
import pandas as pd
from src.data_standardiser.geocoding import geocode_address, check_preprocessed_file

@patch('src.data_standardiser.geocoding.geocode')
@patch('src.data_standardiser.geocoding.requests.get')
def test_geocode_address_nominatim_success(mock_requests_get, mock_geocode):
    # Mock successful Nominatim geocoding
    mock_location = MagicMock()
    mock_location.latitude = 51.5074
    mock_location.longitude = -0.1278
    mock_geocode.return_value = mock_location

    result = geocode_address("10 Downing St, London, UK")

    assert result == (51.5074, -0.1278)
    mock_geocode.assert_called_once()
    mock_requests_get.assert_not_called()

@patch('src.data_standardiser.geocoding.geocode')
@patch('src.data_standardiser.geocoding.requests.get')
@patch('src.data_standardiser.geocoding.clean_address')
@pytest.mark.skip(reason="Skipping edge cases for now")
def test_geocode_address_arcgis_fallback(mock_clean_address, mock_requests_get, mock_geocode):
    # Mock clean_address function
    mock_clean_address.return_value = "10 Downing St, London, UK"

    # Mock Nominatim failure
    mock_geocode.return_value = None

    # Mock ArcGIS success
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"candidates": [{"location": {"x": -0.1278, "y": 51.5074}}]}'
    mock_requests_get.return_value = mock_response

    result = geocode_address("10 Downing St, London, UK")

    assert result == (51.5074, -0.1278)
    mock_clean_address.assert_called_once()
    assert mock_geocode.call_count > 0, "Expected 'geocode' to have been called at least once"
    mock_requests_get.assert_called_once()

@patch('src.data_standardiser.geocoding.geocode')
@patch('src.data_standardiser.geocoding.requests.get')
def test_geocode_address_both_fail(mock_requests_get, mock_geocode):
    # Mock both Nominatim and ArcGIS failure
    mock_geocode.return_value = None

    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = '{"candidates": []}'
    mock_requests_get.return_value = mock_response

    result = geocode_address("Invalid Address")

    assert result == (None, None)
    mock_geocode.assert_called()
    mock_requests_get.assert_called()

@patch('os.path.exists')
@patch('pandas.read_csv')
def test_check_preprocessed_file_exists(mock_read_csv, mock_exists):
    mock_exists.return_value = True
    mock_df = pd.DataFrame({
        'latitude': [51.5074, 51.5075],
        'longitude': [-0.1278, -0.1279]
    })
    mock_read_csv.return_value = mock_df

    result = check_preprocessed_file('dummy_path.csv')

    assert result == True
    mock_exists.assert_called_once()
    mock_read_csv.assert_called_once()

@patch('os.path.exists')
def test_check_preprocessed_file_not_exists(mock_exists):
    mock_exists.return_value = False

    result = check_preprocessed_file('dummy_path.csv')

    assert result == False
    mock_exists.assert_called_once()
