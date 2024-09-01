import logging
import re

import pandas as pd

logger = logging.getLogger(__name__)


def standardise_price(price):
    if not isinstance(price, str):
        return price

    price = price.replace('£', '').replace(',', '').replace('€', '').strip()

    try:
        return float(price) if '.' in price else int(price)
    except ValueError:
        logger.warning(f"Could not convert price '{price}' to a number.")
        return None


def standardize_epc_rating(rating):
    if rating is None:
        return None
    match = re.search(r'\b[A-G]\b', str(rating), re.IGNORECASE)
    return match.group().upper() if match else None


def extract_number(value):
    if value is None:
        return None
    match = re.search(r'\d+', str(value))
    return int(match.group()) if match else None


def standardize_size(size):
    if size is None:
        return None

    # Remove any thousands separators and extra whitespace
    size = re.sub(r'\s+', ' ', str(size)).strip()

    # Try to extract both sq ft and sq m values
    match = re.search(r'([\d,]+(?:\.\d+)?)\s*sq\s*ft\s*/\s*([\d,]+(?:\.\d+)?)\s*sq\s*m', size, re.IGNORECASE)

    if match:
        sq_ft, sq_m = map(lambda x: float(x.replace(',', '')), match.groups())
    else:
        # If we don't have both, try to find just one and calculate the other
        match = re.search(r'([\d,]+(?:\.\d+)?)\s*(sq\s*ft|sq\s*m)', size, re.IGNORECASE)
        if match:
            value, unit = float(match.group(1).replace(',', '')), match.group(2).lower()
            if 'sq ft' in unit:
                sq_ft, sq_m = value, value / 10.7639
            else:
                sq_m, sq_ft = value, value * 10.7639
        else:
            return size  # If we can't parse it, return the original string

    # Ensure sq_ft is always the larger number
    if sq_m > sq_ft:
        sq_ft, sq_m = sq_m * 10.7639, sq_m

    return f"{round(sq_ft)} sq ft / {round(sq_m)} sq m"


def update_date_column(df, source_column, new_date):
    """
    Update 'Date' column in the DataFrame based on source.
    """
    df['Date'] = pd.NaT
    if 'source' in df.columns:
        if source_column in df.columns:
            df.loc[df['source'] == 'registry', 'Date'] = pd.to_datetime(df[source_column], errors='coerce')
        else:
            print(f"Warning: '{source_column}' not found in the dataframe. Using default date for all rows.")
            df['Date'] = new_date
        df.loc[df['source'] == 'scraped', 'Date'] = new_date
    else:
        print("Warning: 'source' column not found. Using default date for all rows.")
        df['Date'] = new_date
    return df
