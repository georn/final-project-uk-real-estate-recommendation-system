import logging
import os
import ssl
import time
import urllib.parse

import certifi
import geopy
import pandas as pd
from geopy.exc import GeocoderTimedOut, GeocoderQuotaExceeded
from geopy.extra.rate_limiter import RateLimiter
from geopy.geocoders import Nominatim, ArcGIS

from src.data_standardiser.address_utils import clean_address

# Disable SSL verification (not recommended for production)
ctx = ssl.create_default_context(cafile=certifi.where())
geopy.geocoders.options.default_ssl_context = ctx

# Initialize Nominatim API and ArcGIS
geolocator = Nominatim(user_agent="StudentDataProjectScraper/1.0 (Contact: gor5@student.london.ac.uk)", scheme='http')
geolocator_arcgis = ArcGIS(user_agent="StudentDataProjectScraper/1.0 (Contact: gor5@student.london.ac.uk)",
                           scheme='http')

# Rate limiter to avoid overloading the API
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=2)
geocode_arcgis = RateLimiter(geolocator_arcgis.geocode, min_delay_seconds=2)

logger = logging.getLogger(__name__)
geocoding_cache = {}


def geocode_address(address):
    cleaned_address = clean_address(address)
    if cleaned_address in geocoding_cache:
        return geocoding_cache[cleaned_address]
    for attempt in range(3):  # Try up to 3 times
        try:
            # Try Nominatim first
            nominatim_url = f"https://nominatim.openstreetmap.org/search?q={urllib.parse.quote(cleaned_address)}&format=json"
            logger.info(f"Attempting Nominatim geocoding (Attempt {attempt + 1}/3)")
            logger.info(f"Nominatim Request URL: {nominatim_url}")

            location = geocode(cleaned_address)
            if location:
                logger.info(
                    f"Nominatim geocoded '{cleaned_address}': Latitude {location.latitude}, Longitude {location.longitude}")
                return location.latitude, location.longitude
            else:
                logger.warning(f"Nominatim failed to geocode '{cleaned_address}', falling back to ArcGIS.")
                # Fallback to ArcGIS
                arcgis_url = f"https://geocode.arcgis.com/arcgis/rest/services/World/GeocodeServer/findAddressCandidates?f=json&singleLine={urllib.parse.quote(cleaned_address)}"
                logger.info(f"Attempting ArcGIS geocoding (Attempt {attempt + 1}/3)")
                logger.info(f"ArcGIS Request URL: {arcgis_url}")

                location = geocode_arcgis(cleaned_address)
                if location:
                    logger.info(
                        f"ArcGIS geocoded '{cleaned_address}': Latitude {location.latitude}, Longitude {location.longitude}")
                    return location.latitude, location.longitude
                else:
                    logger.warning(f"No result for '{cleaned_address}' using ArcGIS")
        except GeocoderQuotaExceeded:
            logger.error("Quota exceeded for geocoding API")
            return None, None
        except GeocoderTimedOut:
            logger.warning(f"Geocoding API timed out for '{cleaned_address}'. Retrying...")
            time.sleep(2 ** attempt)  # Exponential backoff
        except Exception as e:
            logger.error(f"Error geocoding {cleaned_address}: {str(e)}")
            time.sleep(2 ** attempt)  # Exponential backoff

    logger.error(f"Failed to geocode '{cleaned_address}' after 3 attempts")
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
