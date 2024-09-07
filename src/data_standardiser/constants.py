import os
import sys

# Setup path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

csv_file_path = os.path.join(project_root, 'data', 'historical-data', 'multi_county_2023_cleaned_data.csv')

json_file_paths = {
    'bedfordshire': os.path.join(project_root, 'data', 'property_data_bedfordshire_900000.json'),
    'buckinghamshire': os.path.join(project_root, 'data', 'property_data_buckinghamshire_900000.json'),
    'hertfordshire': os.path.join(project_root, 'data', 'property_data_hertfordshire_900000.json'),
    'berkshire': os.path.join(project_root, 'data', 'property_data_berkshire_900000.json'),
    'northamptonshire': os.path.join(project_root, 'data', 'property_data_northamptonshire_900000.json'),
    'oxfordshire': os.path.join(project_root, 'data', 'property_data_oxfordshire_900000.json')
}

# Mapping for converting scraped property types to registry property types
scraped_to_registry_property_type_mapping = {
    'Apartment': 'F',
    'Barn conversion': 'O',
    'Block of apartments': 'F',
    'Bungalow': 'D',
    'Character property': 'O',
    'Cluster house': 'O',
    'Coach house': 'F',
    'Cottage': 'D',
    'Detached bungalow': 'D',
    'Detached house': 'D',
    'Duplex': 'F',
    'End of terrace house': 'T',
    'Equestrian property': 'O',
    'Farm house': 'O',
    'Flat': 'F',
    'Ground floor flat': 'F',
    'Ground floor maisonette': 'F',
    'House': 'D',
    'Link detached house': 'D',
    'Lodge': 'O',
    'Maisonette': 'F',
    'Mews': 'O',
    'Penthouse': 'F',
    'Semi-detached bungalow': 'D',
    'Semi-detached house': 'S',
    'Studio': 'F',
    'Terraced house': 'T',
    'Townhouse': 'D'
}
