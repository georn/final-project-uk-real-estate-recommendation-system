import re

def clean_address(address):
    # Remove any special characters except letters, numbers, spaces, and commas
    cleaned = re.sub(r'[^\w\s,]', '', address)
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Remove any stray commas
    cleaned = re.sub(r',+', ',', cleaned)
    # Remove redundant city names
    cleaned = re.sub(r'(\w+)\s+\1', r'\1', cleaned)
    # Ensure there's a space after each comma
    cleaned = re.sub(r',(?=[^\s])', ', ', cleaned)
    # Add UK if not present
    if not cleaned.lower().endswith(('uk', 'united kingdom')):
        cleaned += ', UK'
    # Final clean-up of any stray spaces
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def normalize_address_scraped(address):
    """
    Normalize addresses from the scraped data.
    """
    # Assuming the county is always 'Buckinghamshire' if not specified
    if 'Buckinghamshire' not in address:
        address += ', Buckinghamshire'
    return address.strip()

def normalize_address_land_registry(row):
    # Convert each component to a string to avoid TypeError
    components = [
        str(row['Street']),
        str(row['Locality']),
        str(row['Town/City']),
        str(row['District']),
        str(row['County'])
    ]
    # Join the non-empty components
    return ', '.join(filter(None, components))
