import argparse
import logging
import os
import sys

# Setup path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.data_standardiser.constants import csv_file_path, json_file_path
from src.data_standardiser.database_operations import process_and_insert_historical_data, \
    process_and_insert_listing_data, merge_data_in_database

logger = logging.getLogger(__name__)


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
