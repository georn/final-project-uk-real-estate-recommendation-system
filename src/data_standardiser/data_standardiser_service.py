import sys
import os
import json
import pandas as pd
from datetime import datetime
import argparse
import logging
from sqlalchemy.exc import IntegrityError

# Setup path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

csv_file_path = os.path.join(project_root, 'data', 'historical-data', 'buckinghamshire_2023_cleaned_data.csv')
json_file_path = os.path.join(project_root, 'data', 'property_data_650000.json')

from geopy.geocoders import Nominatim, ArcGIS
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderQuotaExceeded

from src.database.database import SessionLocal
from src.database.models.historical_property import HistoricalProperty, PropertyType, PropertyAge, PropertyDuration, \
    PPDCategoryType, RecordStatus
from src.database.models.listing_property import ListingProperty

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Nominatim API
geolocator = Nominatim(user_agent="StudentDataProjectScraper/1.0 (Contact: gor5@student.london.ac.uk)")
geolocator_arcgis = ArcGIS(user_agent="StudentDataProjectScraper/1.0 (Contact: gor5@student.london.ac.uk)")

# Rate limiter to avoid overloading the API
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2)
geocode_arcgis = RateLimiter(geolocator_arcgis.geocode, min_delay_seconds=2)

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


def geocode_address(address):
    try:
        location = geocode(address)
        if location:
            print(f"Geocoded '{address}': Latitude {location.latitude}, Longitude {location.longitude}")
            return location.latitude, location.longitude
        else:
            # Fallback to ArcGIS if Nominatim fails
            location = geocode_arcgis(address)
            if location:
                print(f"Geocoded '{address}': Latitude {location.latitude}, Longitude {location.longitude}")
                return location.latitude, location.longitude
            else:
                print(f"No result for '{address}'")
                return None, None
    except GeocoderQuotaExceeded:
        print("Quota exceeded for geocoding API")
        return None, None
    except GeocoderTimedOut:
        print("Geocoding API timed out")
        return None, None
    except Exception as e:
        print(f"Error geocoding {address}: {e}")
        return None, None


def check_preprocessed_file(file_path):
    """Check if the preprocessed file exists and has latitude and longitude."""
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        if 'latitude' in df.columns and 'longitude' in df.columns:
            if df[['latitude', 'longitude']].notnull().all().all():
                # File exists and latitude and longitude are filled
                return True
    return False


def standardise_price(price):
    """
    Convert a price string to a numerical value.
    Handles strings like '£275,000' and converts them to 275000.
    """
    if not isinstance(price, str):
        return price  # If it's already a number, return as-is

    # Removing currency symbols and commas
    price = price.replace('£', '').replace(',', '').replace('€', '').strip()

    try:
        # Convert to float or int
        price_value = float(price) if '.' in price else int(price)
    except ValueError:
        # Handle cases where conversion fails
        print(f"Warning: Could not convert price '{price}' to a number.")
        price_value = None

    return price_value


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


# Read JSON, standardize price, normalize address, add source column

def read_and_process_scraped_data(scraped_file_path, skip_geocoding):
    # Read the scraped data
    scraped_data = pd.read_json(scraped_file_path)

    # Ensure 'id' column exists and is an integer
    if 'id' not in scraped_data.columns:
        scraped_data['id'] = range(1, len(scraped_data) + 1)
    else:
        scraped_data['id'] = scraped_data['id'].astype(int)

    scraped_data['price'] = scraped_data['price'].apply(standardise_price)
    scraped_data['normalized_address'] = scraped_data['address'].apply(normalize_address_scraped)
    scraped_data['source'] = 'scraped'

    if not skip_geocoding:
        lat_long = scraped_data['normalized_address'].apply(geocode_address)
        scraped_data['latitude'] = lat_long.apply(lambda x: x[0] if x else None)
        scraped_data['longitude'] = lat_long.apply(lambda x: x[1] if x else None)
    else:
        # If skipping geocoding, add placeholder columns
        scraped_data['latitude'] = None
        scraped_data['longitude'] = None

    # Map the property types
    scraped_data['Property Type'] = scraped_data['property_type'].map(scraped_to_registry_property_type_mapping)

    return scraped_data
    # Read the scraped data
    scraped_data = pd.read_json(scraped_file_path)

    # Ensure 'id' column exists and is an integer
    if 'id' not in scraped_data.columns:
        scraped_data['id'] = range(1, len(scraped_data) + 1)
    else:
        scraped_data['id'] = scraped_data['id'].astype(int)

    scraped_data['price'] = scraped_data['price'].apply(standardise_price)
    scraped_data['normalized_address'] = scraped_data['address'].apply(normalize_address_scraped)
    scraped_data['source'] = 'scraped'

    if not skip_geocoding:
        lat_long = scraped_data['normalized_address'].apply(geocode_address)
        scraped_data['latitude'] = lat_long.apply(lambda x: x[0] if x else None)
        scraped_data['longitude'] = lat_long.apply(lambda x: x[1] if x else None)

    # Map the property types
    scraped_data['Property Type'] = scraped_data['property_type'].map(scraped_to_registry_property_type_mapping)

    return scraped_data


def read_and_process_registry_data(registry_file_path, skip_geocoding):
    registry_data = pd.read_csv(registry_file_path)

    registry_data['Price'] = registry_data['Price'].apply(standardise_price)
    registry_data['normalized_address'] = registry_data.apply(normalize_address_land_registry, axis=1)
    registry_data.rename(columns={'Price': 'price'}, inplace=True)

    if not skip_geocoding:
        lat_long = registry_data['normalized_address'].apply(geocode_address)
        registry_data['latitude'] = lat_long.apply(lambda x: x[0] if x else None)
        registry_data['longitude'] = lat_long.apply(lambda x: x[1] if x else None)
    else:
        registry_data['latitude'] = None
        registry_data['longitude'] = None

    return registry_data


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


def process_and_save_data(scraped_data, registry_data, output_file_path):
    """
    Process and save merged data with unique IDs.
    """
    # Reset index for both datasets to ensure unique indices
    scraped_data = scraped_data.reset_index(drop=True)
    registry_data = registry_data.reset_index(drop=True)

    # Add a temporary column to identify the source
    scraped_data['temp_source'] = 'scraped'
    registry_data['temp_source'] = 'registry'

    # Concatenate the datasets
    merged_data = pd.concat([scraped_data, registry_data], ignore_index=True)

    # Create a new unique ID column
    merged_data['id'] = range(1, len(merged_data) + 1)

    # Update the date column
    # Check if 'Date of Transfer' exists, if not use 'Date of Transaction'
    date_column = 'Date of Transfer' if 'Date of Transfer' in merged_data.columns else 'Date of Transaction'
    if date_column not in merged_data.columns:
        print(f"Warning: Neither 'Date of Transfer' nor 'Date of Transaction' found. Using current date for all rows.")
        date_column = None

    merged_data = update_date_column(merged_data, date_column, datetime(2023, 12, 31))

    # Remove the temporary source column
    merged_data = merged_data.drop(columns=['temp_source'])

    # Ensure 'id' is the first column
    cols = ['id'] + [col for col in merged_data.columns if col != 'id']
    merged_data = merged_data[cols]

    # Save merged data
    merged_data.to_csv(output_file_path, index=False)
    print(f"Merged data saved successfully to '{output_file_path}'.")

    return merged_data
    """
    Process and save merged data with unique IDs.
    """
    # Reset index for both datasets to ensure unique indices
    scraped_data = scraped_data.reset_index(drop=True)
    registry_data = registry_data.reset_index(drop=True)

    # Add a temporary column to identify the source
    scraped_data['temp_source'] = 'scraped'
    registry_data['temp_source'] = 'registry'

    # Concatenate the datasets
    merged_data = pd.concat([scraped_data, registry_data], ignore_index=True)

    # Create a new unique ID column
    merged_data['id'] = range(1, len(merged_data) + 1)

    # Update the date column
    merged_data = update_date_column(merged_data, 'Date of Transfer', datetime(2023, 12, 31))

    # Remove the temporary source column
    merged_data = merged_data.drop(columns=['temp_source'])

    # Ensure 'id' is the first column
    cols = ['id'] + [col for col in merged_data.columns if col != 'id']
    merged_data = merged_data[cols]

    # Save merged data
    merged_data.to_csv(output_file_path, index=False)
    print(f"Merged data saved successfully to '{output_file_path}'.")

    return merged_data


def merge_datasets(scraped_data, registry_data):
    # Merge the two datasets
    merged_data = pd.concat([scraped_data, registry_data], ignore_index=True)

    return merged_data


def save_merged_data(merged_data, output_file_path):
    try:
        merged_data.to_csv(output_file_path, index=False)
        print(f"Data saved successfully to {output_file_path}")
    except Exception as e:
        print(f"An error occurred while saving the data: {e}")


def process_and_insert_historical_data(registry_file_path, skip_geocoding=True):
    registry_data = read_and_process_registry_data(registry_file_path, skip_geocoding)

    db = SessionLocal()
    try:
        for _, row in registry_data.iterrows():
            historical_property = HistoricalProperty(
                id=row['Unique Transaction Identifier'],
                price=row['price'],
                date_of_transaction=pd.to_datetime(row['Date of Transaction']).date(),
                postal_code=row['Postal Code'],
                property_type=PropertyType(row['Property Type']),
                property_age=PropertyAge(row['Old/New']),
                duration=PropertyDuration(row['Duration']),
                paon=row['PAON'],
                saon=row['SAON'],
                street=row['Street'],
                locality=row['Locality'],
                town_city=row['Town/City'],
                district=row['District'],
                ppd_category_type=PPDCategoryType(row['PPD Category Type']),
                record_status=RecordStatus(row['Record Status'])
            )
            db.add(historical_property)

        db.commit()
        print(f"Inserted {len(registry_data)} historical properties into the database.")
    except Exception as e:
        db.rollback()
        print(f"An error occurred while inserting data: {e}")
    finally:
        db.close()


def process_and_insert_listing_data(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)

    db = SessionLocal()
    try:
        for item in data:
            listing = ListingProperty(
                property_url=item.get('property_url'),
                title=item.get('title'),
                address=item.get('address'),
                price=standardise_price(item.get('price')),
                pricing_qualifier=item.get('pricing_qualifier'),
                listing_time=item.get('listing_time'),
                property_type=item.get('property_type'),
                bedrooms=item.get('bedrooms'),
                bathrooms=item.get('bathrooms'),
                epc_rating=item.get('epc_rating'),
                size=item.get('size'),
                features=item.get('features')
            )
            try:
                db.add(listing)
                db.commit()
                logger.info(f"Inserted listing: {listing.title}")
            except IntegrityError:
                db.rollback()
                logger.warning(f"Duplicate listing found, skipping: {listing.property_url}")
            except Exception as e:
                db.rollback()
                logger.error(f"Error inserting listing {listing.title}: {e}")

        logger.info(f"Processed {len(data)} listings.")
    finally:
        db.close()


def main():
    parser = argparse.ArgumentParser(description="Process property data with optional geocoding.")
    parser.add_argument('--skip-geocoding', action='store_true', help='Skip the geocoding process')
    args = parser.parse_args()

    scraped_file = json_file_path
    registry_file = csv_file_path
    # output_file = '../../data/preprocessed-data/preprocessed.csv'

    # process_and_insert_historical_data(registry_file, args.skip_geocoding)
    process_and_insert_listing_data(scraped_file)

    # Always process new data
    # scraped_data = read_and_process_scraped_data(scraped_file, args.skip_geocoding)
    # registry_data = read_and_process_registry_data(registry_file, args.skip_geocoding)

    # Process and save merged data
    # merged_data = process_and_save_data(scraped_data, registry_data, output_file)

    print("Data processing completed.")

    # Optional: Print some information about the merged dataset
    # print(f"Total number of records: {len(merged_data)}")
    # print(f"Number of scraped records: {len(scraped_data)}")
    # print(f"Number of registry records: {len(registry_data)}")
    # print(f"Columns in the merged dataset: {merged_data.columns.tolist()}")


if __name__ == "__main__":
    main()
