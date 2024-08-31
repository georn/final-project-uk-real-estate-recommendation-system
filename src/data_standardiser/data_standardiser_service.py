import argparse
import json
import logging
import os
import re
import ssl
import sys
import time
import urllib.parse
from datetime import date

import certifi
import geopy
import pandas as pd
from sqlalchemy.dialects.postgresql import insert



from geopy.geocoders import Nominatim, ArcGIS
from geopy.extra.rate_limiter import RateLimiter
from geopy.exc import GeocoderTimedOut, GeocoderQuotaExceeded

from src.database.database import SessionLocal
from src.database.models.historical_property import HistoricalProperty, PropertyType, PropertyAge, PropertyDuration, \
    PPDCategoryType, RecordStatus
from src.database.models.listing_property import ListingProperty, Tenure
from src.database.models.merged_property import MergedProperty, Tenure as MergedTenure





























# Read JSON, standardize price, normalize address, add source column














