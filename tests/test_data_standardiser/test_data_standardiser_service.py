# import sys
# from unittest.mock import MagicMock
#
# import pytest
#
# from src.data_standardiser.data_standardiser_service import clean_address
#
# # Mock geopy before importing data_standardiser_service
# sys.modules['geopy'] = MagicMock()
# sys.modules['geopy.geocoders'] = MagicMock()
# sys.modules['geopy.exc'] = MagicMock()
#
# @pytest.fixture
# def mock_json_data():
#     return [
#         {
#             "property_url": "https://example.com/property1",
#             "title": "Beautiful House",
#             "address": "123 Main St, City, UK",
#             "price": "£300,000",
#             "pricing_qualifier": "Guide Price",
#             "listing_time": "3 days ago",
#             "property_type": "Detached house",
#             "bedrooms": "3 bed",
#             "bathrooms": "2 bath",
#             "epc_rating": "C",
#             "size": "1500 sq ft",
#             "features": ["Garden", "Parking"]
#         }
#     ]
#
# def test_clean_address():
#     # Test 1: Basic address cleaning
#     address = "  123 Main St., City, @UK!!  "
#     expected = "123 Main St City, UK"
#     assert clean_address(address) == expected
#
#     # Test 2: Address that already ends with ', UK'
#     address = "123 Main St, City, UK"
#     expected = "123 Main St City, UK"
#     assert clean_address(address) == expected
#
#     # Test 3: Address that ends with 'United Kingdom'
#     address = "456 Another Rd, Town, United Kingdom"
#     expected = "456 Another Rd Town, United Kingdom"
#     assert clean_address(address) == expected
#
#     # Test 4: Address without 'UK' or 'United Kingdom'
#     address = "789 Street Name, City"
#     expected = "789 Street Name City, UK"
#     assert clean_address(address) == expected
#
#
# if __name__ == "__main__":
#     pytest.main([__file__])
