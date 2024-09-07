COUNTY_MAPPING = {
    'VALE OF WHITE HORSE': 'oxfordshire',
    'BEDFORD': 'bedfordshire',
    'HERTSMERE': 'hertfordshire',
    'NORTH HERTFORDSHIRE': 'hertfordshire',
    'CHERWELL': 'oxfordshire',
    'THREE RIVERS': 'hertfordshire',
    'DACORUM': 'hertfordshire',
    'ST ALBANS': 'hertfordshire',
    'WATFORD': 'hertfordshire',
    'EAST HERTFORDSHIRE': 'hertfordshire',
    'WEST BERKSHIRE': 'berkshire',
    'BROXBOURNE': 'hertfordshire',
    'OXFORD': 'oxfordshire',
    'STEVENAGE': 'hertfordshire',
    'WEST NORTHAMPTONSHIRE': 'northamptonshire',
    'WEST OXFORDSHIRE': 'oxfordshire',
    'WELWYN HATFIELD': 'hertfordshire',
    'NORTH NORTHAMPTONSHIRE': 'northamptonshire',
    'SOUTH OXFORDSHIRE': 'oxfordshire',
    'BUCKINGHAMSHIRE': 'buckinghamshire',
}


def standardize_county(county):
    """
    Standardize county names to a consistent set of shires.
    """
    if county is None:
        return None

    county = county.upper()
    return COUNTY_MAPPING.get(county, county.lower())
