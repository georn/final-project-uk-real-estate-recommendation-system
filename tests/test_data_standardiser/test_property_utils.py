import pytest
from src.data_standardiser.property_utils import standardize_property_type, extract_tenure
from src.database.models.listing_property import Tenure

def test_standardize_property_type():
    # Test registry mapping
    assert standardize_property_type('D') == 'Detached'
    assert standardize_property_type('S') == 'Semi-Detached'
    assert standardize_property_type('T') == 'Terraced'
    assert standardize_property_type('F') == 'Flat/Maisonette'
    assert standardize_property_type('O') == 'Other'

    # Test scraped mapping
    assert standardize_property_type('Apartment') == 'Flat/Maisonette'
    assert standardize_property_type('Bungalow') == 'Detached'
    assert standardize_property_type('Semi-detached house') == 'Semi-Detached'
    assert standardize_property_type('Terraced house') == 'Terraced'
    assert standardize_property_type('Barn') == 'Other'

    # Test unknown property type
    assert standardize_property_type('Unknown Type') == 'Other'

def test_extract_tenure():
    # Test freehold
    features = ["Tenure: Freehold", "Garden", "Parking"]
    assert extract_tenure(features) == Tenure.FREEHOLD

    # Test leasehold
    features = ["Garden", "Tenure: Leasehold", "Parking"]
    assert extract_tenure(features) == Tenure.LEASEHOLD

    # Test unknown
    features = ["Garden", "Parking"]
    assert extract_tenure(features) == Tenure.UNKNOWN

    # Test None features
    assert extract_tenure(None) == Tenure.UNKNOWN

    # Test case insensitivity
    features = ["TENURE: FREEHOLD", "Garden"]
    assert extract_tenure(features) == Tenure.FREEHOLD
