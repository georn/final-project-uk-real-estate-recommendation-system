import pytest
from unittest.mock import patch, MagicMock, mock_open
from src.data_standardiser.database_operations import (
    update_geocoding_data,
    process_and_insert_historical_data,
    process_and_insert_listing_data,
    merge_data_in_database
)
from src.database.models.listing_property import ListingProperty
from src.database.models.merged_property import MergedProperty
from src.database.models.historical_property import HistoricalProperty, PropertyDuration

@pytest.fixture
def mock_db_session():
    with patch('src.data_standardiser.database_operations.SessionLocal') as mock_session:
        yield mock_session()

def test_update_geocoding_data(mock_db_session):
    mock_listing = MagicMock(spec=ListingProperty)
    mock_listing.id = 1
    mock_listing.address = "123 Test St, Test City"
    mock_db_session.query.return_value.filter.return_value.all.return_value = [mock_listing]

    with patch('src.data_standardiser.database_operations.geocode_address', return_value=(51.5074, -0.1278)):
        update_geocoding_data(batch_size=1)

    mock_db_session.commit.assert_called()
    assert mock_listing.latitude == 51.5074
    assert mock_listing.longitude == -0.1278

@patch('pandas.read_csv')
def test_process_and_insert_historical_data(mock_read_csv, mock_db_session):
    mock_df = MagicMock()
    mock_df.__len__.return_value = 1
    mock_df.iterrows.return_value = iter([(0, {
        'Unique Transaction Identifier': '123ABC',
        'Price': 300000,
        'Date of Transaction': '2023-01-01',
        'Postal Code': 'SW1A 1AA',
        'Property Type': 'D',
        'Old/New': 'Y',
        'Duration': 'F',
        'PAON': '10',
        'SAON': '',
        'Street': 'Downing Street',
        'Locality': 'Westminster',
        'Town/City': 'London',
        'District': 'City of Westminster',
        'PPD Category Type': 'A',
        'Record Status': 'A'
    })])
    mock_read_csv.return_value = mock_df

    process_and_insert_historical_data('dummy_path.csv')

    mock_db_session.execute.assert_called()
    mock_db_session.commit.assert_called()

@patch('json.load')
@patch('builtins.open', new_callable=mock_open)
def test_process_and_insert_listing_data(mock_file, mock_json_load, mock_db_session):
    mock_json_load.return_value = [{
        'property_url': 'http://example.com/property/1',
        'title': 'Test Property',
        'address': '123 Test St, Test City',
        'price': '£300,000',
        'pricing_qualifier': 'Guide Price',
        'listing_time': '3 days ago',
        'property_type': 'Detached house',
        'bedrooms': '3',
        'bathrooms': '2',
        'epc_rating': 'C',
        'size': '1500 sq ft',
        'features': ['Garden', 'Parking']
    }]

    with patch('src.data_standardiser.database_operations.geocode_address', return_value=(51.5074, -0.1278)):
        with patch('src.data_standardiser.database_operations.extract_tenure', return_value='FREEHOLD'):
            process_and_insert_listing_data('dummy_path.json')

    mock_db_session.execute.assert_called()
    mock_db_session.commit.assert_called()

def test_merge_data_in_database(mock_db_session):
    mock_historical = MagicMock(spec=HistoricalProperty)
    mock_historical.duration = PropertyDuration.FREEHOLD
    mock_historical.postal_code = "SW1A 1AA"
    mock_listing = MagicMock(spec=ListingProperty)
    mock_listing.id = 1
    mock_listing.address = "SW1A 1AA"

    mock_db_session.query.return_value.all.return_value = [mock_historical]
    mock_db_session.query.return_value.filter_by.return_value.first.return_value = mock_listing

    # Mock the MergedProperty.id to be None to simulate no existing merged property
    mock_db_session.query.return_value.outerjoin.return_value.filter.return_value.all.return_value = []

    merge_data_in_database()

    mock_db_session.add.assert_called()
    mock_db_session.commit.assert_called()
