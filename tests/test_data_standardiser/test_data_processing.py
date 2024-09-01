import pytest
import pandas as pd
from datetime import datetime
from src.data_standardiser.data_processing import (
    standardise_price,
    standardize_epc_rating,
    extract_number,
    standardize_size,
    update_date_column
)

def test_standardise_price():
    assert standardise_price("£300,000") == 300000
    assert standardise_price("€250,000.50") == 250000.50
    assert standardise_price(300000) == 300000
    assert standardise_price("Invalid") is None

def test_standardize_epc_rating():
    assert standardize_epc_rating("A") == "A"
    assert standardize_epc_rating("b") == "B"
    assert standardize_epc_rating("EPC Rating: C") == "C"
    assert standardize_epc_rating("Not Rated") is None
    assert standardize_epc_rating(None) is None

def test_extract_number():
    assert extract_number("3 bedrooms") == 3
    assert extract_number("Bath: 2") == 2
    assert extract_number("No numbers here") is None
    assert extract_number(None) is None

def test_standardize_size():
    assert standardize_size("1500 sq ft") == "1500 sq ft / 139 sq m"
    assert standardize_size("150 sq m") == "1615 sq ft / 150 sq m"
    assert standardize_size("1,500 sq ft / 139.35 sq m") == "1500 sq ft / 139 sq m"
    assert standardize_size("2,000 sq ft") == "2000 sq ft / 186 sq m"
    assert standardize_size("1,234.56 sq ft") == "1235 sq ft / 115 sq m"
    assert standardize_size("1,000,000 sq ft") == "1000000 sq ft / 92903 sq m"
    assert standardize_size("Invalid size") == "Invalid size"
    assert standardize_size(None) is None

def test_update_date_column():
    df = pd.DataFrame({
        'source': ['registry', 'scraped', 'registry'],
        'old_date': ['2023-01-01', '2023-02-01', '2023-03-01']
    })
    new_date = datetime(2023, 5, 1)

    result = update_date_column(df, 'old_date', new_date)

    assert result['Date'].iloc[0] == pd.Timestamp('2023-01-01')
    assert result['Date'].iloc[1] == pd.Timestamp('2023-05-01')
    assert result['Date'].iloc[2] == pd.Timestamp('2023-03-01')

    # Test when 'source' column is missing
    df_no_source = pd.DataFrame({'old_date': ['2023-01-01', '2023-02-01']})
    result_no_source = update_date_column(df_no_source, 'old_date', new_date)
    assert (result_no_source['Date'] == pd.Timestamp('2023-05-01')).all()

    # Test when source_column is missing
    result_no_column = update_date_column(df, 'non_existent_column', new_date)
    assert (result_no_column['Date'] == pd.Timestamp('2023-05-01')).all()
