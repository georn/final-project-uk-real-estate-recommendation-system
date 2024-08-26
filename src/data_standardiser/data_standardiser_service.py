import argparse
import json
import logging
import os
import ssl
import sys
from datetime import date

import certifi
import geopy
import pandas as pd
from sqlalchemy.dialects.postgresql import insert

# Setup path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from geopy.geocoders import Nominatim, ArcGIS
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderQuotaExceeded

from src.database.database import SessionLocal
from src.database.models.historical_property import HistoricalProperty, PropertyType, PropertyAge, PropertyDuration, \
    PPDCategoryType, RecordStatus
from src.database.models.listing_property import ListingProperty
from src.database.models.merged_property import MergedProperty

csv_file_path = os.path.join(project_root, 'data', 'historical-data', 'buckinghamshire_2023_cleaned_data.csv')
json_file_path = os.path.join(project_root, 'data', 'property_data_650000.json')

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable SSL verification (not recommended for production)
ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx

# Initialize Nominatim API and ArcGIS
geolocator = Nominatim(user_agent="StudentDataProjectScraper/1.0 (Contact: gor5@student.london.ac.uk)", scheme='http')
geolocator_arcgis = ArcGIS(user_agent="StudentDataProjectScraper/1.0 (Contact: gor5@student.london.ac.uk)", scheme='http')

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


def geocode_address(address):
    cleaned_address = clean_address(address)
    try:
        # Try Nominatim first
        location = geocode(cleaned_address)
        if location:
            logger.info(f"Nominatim geocoded '{cleaned_address}': Latitude {location.latitude}, Longitude {location.longitude}")
            return location.latitude, location.longitude
        else:
            logger.warning(f"Nominatim failed to geocode '{cleaned_address}', falling back to ArcGIS.")
            # Fallback to ArcGIS
            try:
                location = geocode_arcgis(cleaned_address)
                if location:
                    logger.info(f"ArcGIS geocoded '{cleaned_address}': Latitude {location.latitude}, Longitude {location.longitude}")
                    return location.latitude, location.longitude
                else:
                    logger.warning(f"No result for '{cleaned_address}' using ArcGIS")
                    return None, None
            except Exception as e:
                logger.error(f"ArcGIS geocoding error for '{cleaned_address}': {str(e)}")
                return None, None
    except GeocoderQuotaExceeded:
        logger.error("Quota exceeded for geocoding API")
        return None, None
    except GeocoderTimedOut:
        logger.error("Geocoding API timed out")
        return None, None
    except Exception as e:
        logger.error(f"Error geocoding {cleaned_address}: {str(e)}")
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
    if not isinstance(price, str):
        return price

    price = price.replace('£', '').replace(',', '').replace('€', '').strip()

    try:
        return float(price) if '.' in price else int(price)
    except ValueError:
        logger.warning(f"Could not convert price '{price}' to a number.")
        return None


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


def process_and_insert_historical_data(file_path):
    df = pd.read_csv(file_path)
    df['Price'] = df['Price'].apply(standardise_price)

    db = SessionLocal()
    try:
        for _, row in df.iterrows():
            stmt = insert(HistoricalProperty).values(
                id=row['Unique Transaction Identifier'],
                price=row['Price'],
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

            # This creates an "upsert" operation
            stmt = stmt.on_conflict_do_update(
                index_elements=['id'],
                set_={
                    'price': stmt.excluded.price,
                    'date_of_transaction': stmt.excluded.date_of_transaction,
                    'postal_code': stmt.excluded.postal_code,
                    'property_type': stmt.excluded.property_type,
                    'property_age': stmt.excluded.property_age,
                    'duration': stmt.excluded.duration,
                    'paon': stmt.excluded.paon,
                    'saon': stmt.excluded.saon,
                    'street': stmt.excluded.street,
                    'locality': stmt.excluded.locality,
                    'town_city': stmt.excluded.town_city,
                    'district': stmt.excluded.district,
                    'ppd_category_type': stmt.excluded.ppd_category_type,
                    'record_status': stmt.excluded.record_status
                }
            )

            db.execute(stmt)

        db.commit()
        logger.info(f"Processed {len(df)} historical properties (inserted or updated).")
    except Exception as e:
        db.rollback()
        logger.error(f"An error occurred while processing historical data: {e}")
    finally:
        db.close()


def process_and_insert_listing_data(file_path, skip_geocoding=False):
    with open(file_path, 'r') as file:
        data = json.load(file)

    db = SessionLocal()
    try:
        for item in data:
            latitude, longitude = None, None
            if not skip_geocoding:
                address = item.get('address')
                latitude, longitude = geocode_address(address)
                if latitude is None or longitude is None:
                    logger.warning(f"Failed to geocode address: {address}")

            stmt = insert(ListingProperty).values(
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
                features=item.get('features'),
                latitude=latitude,
                longitude=longitude
            )

            stmt = stmt.on_conflict_do_update(
                index_elements=['property_url'],
                set_={
                    'title': stmt.excluded.title,
                    'address': stmt.excluded.address,
                    'price': stmt.excluded.price,
                    'pricing_qualifier': stmt.excluded.pricing_qualifier,
                    'listing_time': stmt.excluded.listing_time,
                    'property_type': stmt.excluded.property_type,
                    'bedrooms': stmt.excluded.bedrooms,
                    'bathrooms': stmt.excluded.bathrooms,
                    'epc_rating': stmt.excluded.epc_rating,
                    'size': stmt.excluded.size,
                    'features': stmt.excluded.features,
                    'latitude': stmt.excluded.latitude,
                    'longitude': stmt.excluded.longitude
                }
            )

            db.execute(stmt)

        db.commit()
        logger.info(f"Processed {len(data)} listings (inserted or updated).")
    except Exception as e:
        db.rollback()
        logger.error(f"An error occurred while processing listing data: {e}")
    finally:
        db.close()

def enum_to_string(value):
    return value.value if hasattr(value, 'value') else value

def merge_data_in_database():
    db = SessionLocal()
    try:
        db.query(MergedProperty).delete()

        # fetch all historical properties
        historical_properties = db.query(HistoricalProperty).all()

        for hp in historical_properties:
            # Try to find a matching listing property
            lp = db.query(ListingProperty).filter_by(address=hp.postal_code).first()

            merged_property = MergedProperty(
                historical_id=hp.id,
                listing_id=lp.id if lp else None,
                price=hp.price or (lp.price if lp else None),
                postal_code=hp.postal_code,
                property_type=enum_to_string(hp.property_type) or (lp.property_type if lp else None),
                date=hp.date_of_transaction,
                property_age=enum_to_string(hp.property_age),
                duration=enum_to_string(hp.duration),
                bedrooms=lp.bedrooms if lp else None,
                bathrooms=lp.bathrooms if lp else None,
                epc_rating=lp.epc_rating if lp else None,
                size=lp.size if lp else None,
                features=lp.features if lp else None,
                data_source='both' if lp else 'historical',
                listing_time=lp.listing_time if lp else None,
                latitude=lp.latitude if lp else None,
                longitude=lp.longitude if lp else None,
            )
            db.add(merged_property)

        # Now let's add any listing properties that don't have a matching historical property
        listing_properties = db.query(ListingProperty).outerjoin(
            MergedProperty, ListingProperty.id == MergedProperty.listing_id
        ).filter(MergedProperty.id == None).all()

        for lp in listing_properties:
            merged_property = MergedProperty(
                listing_id=lp.id,
                price=lp.price,
                postal_code=lp.address,  # Assuming address in ListingProperty corresponds to postal_code
                property_type=lp.property_type,
                date=date.today(),  # Use current date for listing properties
                bedrooms=lp.bedrooms,
                bathrooms=lp.bathrooms,
                epc_rating=lp.epc_rating,
                size=lp.size,
                features=lp.features,
                data_source='listing',
                listing_time=lp.listing_time
            )
            db.add(merged_property)

        db.commit()
        logger.info("Merged data created in the database.")
    except Exception as e:
        db.rollback()
        logger.error(f"An error occurred while merging data: {e}")
    finally:
        db.close()

def main():
    parser = argparse.ArgumentParser(description="Process property data with optional geocoding.")
    parser.add_argument('--skip-geocoding', action='store_true', help='Skip the geocoding process')
    args = parser.parse_args()

    scraped_file = json_file_path
    registry_file = csv_file_path

    process_and_insert_historical_data(registry_file)
    process_and_insert_listing_data(scraped_file, args.skip_geocoding)  # Pass the skip_geocoding argument

    logger.info("Merging data in the database.")
    merge_data_in_database()

    print("Data processing completed.")


if __name__ == "__main__":
    main()
