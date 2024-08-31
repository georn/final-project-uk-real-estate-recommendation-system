import re

def clean_address(address):
    # Remove any special characters except letters, numbers, spaces, and commas
    cleaned = re.sub(r'[^\w\s,]', '', address)
    # Replace multiple spaces with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    # Remove any stray commas (e.g., multiple commas in a row)
    cleaned = re.sub(r',+', ',', cleaned)
    # Ensure there's a space after each comma
    cleaned = re.sub(r',(?=[^\s])', ', ', cleaned)
    # Remove commas that are not followed by "UK" or "United Kingdom"
    cleaned = re.sub(r',(?=\s+[^UK]|[^United Kingdom])', '', cleaned)
    # Ensure that "UK" is treated as a single unit and not split
    if cleaned.lower().endswith('uk'):
        if not cleaned.endswith(', UK'):
            cleaned = cleaned[:-2].rstrip() + ', UK'
    elif cleaned.lower().endswith('united kingdom'):
        if not cleaned.endswith(', United Kingdom'):
            cleaned = cleaned[:-14].rstrip() + ', United Kingdom'
    else:
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
