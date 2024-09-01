import pytest
from src.data_standardiser.address_utils import clean_address, normalize_address_scraped, normalize_address_land_registry

def test_clean_address():
    # Test basic address cleaning
    assert clean_address("  123 Main St., City, @UK!!  ") == "123 Main St, City, UK"

    # Test address that already ends with ', UK'
    assert clean_address("123 Main St, City, UK") == "123 Main St, City, UK"

    # Test address that ends with 'United Kingdom'
    assert clean_address("456 Another Rd, Town, United Kingdom") == "456 Another Rd, Town, United Kingdom"

    # Test address without 'UK' or 'United Kingdom'
    assert clean_address("789 Street Name, City") == "789 Street Name, City, UK"

def test_normalize_address_scraped():
    # Test address without 'Buckinghamshire'
    assert normalize_address_scraped("123 Main St, City") == "123 Main St, City, Buckinghamshire"

    # Test address with 'Buckinghamshire'
    assert normalize_address_scraped("456 Another Rd, Town, Buckinghamshire") == "456 Another Rd, Town, Buckinghamshire"

def test_normalize_address_land_registry():
    # Create a mock row dictionary
    row = {
        'Street': 'Main St',
        'Locality': 'Downtown',
        'Town/City': 'Cityville',
        'District': 'Central',
        'County': 'Buckinghamshire'
    }

    expected = "Main St, Downtown, Cityville, Central, Buckinghamshire"
    assert normalize_address_land_registry(row) == expected

    # Test with some empty fields
    row = {
        'Street': 'Main St',
        'Locality': '',
        'Town/City': 'Cityville',
        'District': '',
        'County': 'Buckinghamshire'
    }

    expected = "Main St, Cityville, Buckinghamshire"
    assert normalize_address_land_registry(row) == expected
