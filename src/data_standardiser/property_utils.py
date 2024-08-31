from src.database.models.listing_property import Tenure


def standardize_property_type(property_type):
    registry_mapping = {
        'D': 'Detached',
        'S': 'Semi-Detached',
        'T': 'Terraced',
        'F': 'Flat/Maisonette',
        'O': 'Other'
    }

    scraped_mapping = {
        'Apartment': 'Flat/Maisonette',
        'Barn': 'Other',
        'Barn conversion': 'Other',
        'Block of apartments': 'Flat/Maisonette',
        'Bungalow': 'Detached',
        'Chalet': 'Other',
        'Character property': 'Other',
        'Cluster house': 'Other',
        'Coach house': 'Flat/Maisonette',
        'Cottage': 'Detached',
        'Detached bungalow': 'Detached',
        'Detached house': 'Detached',
        'Duplex': 'Flat/Maisonette',
        'End of terrace house': 'Terraced',
        'Equestrian property': 'Other',
        'Farm house': 'Other',
        'Flat': 'Flat/Maisonette',
        'Ground floor flat': 'Flat/Maisonette',
        'Ground floor maisonette': 'Flat/Maisonette',
        'House': 'Detached',
        'Houseboat': 'Other',
        'Link detached house': 'Detached',
        'Lodge': 'Other',
        'Log cabin': 'Other',
        'Maisonette': 'Flat/Maisonette',
        'Mews': 'Other',
        'Penthouse': 'Flat/Maisonette',
        'Semi-detached bungalow': 'Semi-Detached',
        'Semi-detached house': 'Semi-Detached',
        'Studio': 'Flat/Maisonette',
        'Terraced house': 'Terraced',
        'Townhouse': 'Terraced'
    }

    # First, check if it's a registry type
    if property_type in registry_mapping:
        return registry_mapping[property_type]

    # Then, check if it's a scraped type
    elif property_type in scraped_mapping:
        return scraped_mapping[property_type]

    # If it's neither, return 'Other'
    else:
        return 'Other'


def extract_tenure(features):
    if features is None:
        return Tenure.UNKNOWN
    for feature in features:
        if feature.lower().startswith("tenure:"):
            tenure = feature.split(":")[1].strip().lower()
            if "freehold" in tenure:
                return Tenure.FREEHOLD
            elif "leasehold" in tenure:
                return Tenure.LEASEHOLD
    return Tenure.UNKNOWN
