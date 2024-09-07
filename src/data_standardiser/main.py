import argparse
import logging
import os
import sys

# Setup path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(project_root)

from src.data_standardiser.constants import csv_file_path, json_file_paths
from src.data_standardiser.database_operations import (
    process_and_insert_historical_data,
    process_and_insert_listing_data,
    merge_data_in_database,
    update_geocoding_data,
    load_geocoded_data_from_csv
)

# Set up logging
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Process property data with optional geocoding.")
    parser.add_argument('--skip-geocoding', action='store_true', help='Skip the geocoding process')
    parser.add_argument('--update-geocoding', action='store_true', help='Only update geocoding data')
    parser.add_argument('--load-geocoded', nargs=2, metavar=('MERGED_CSV', 'LISTING_CSV'),
                        help='Load pre-geocoded data from CSV files')
    args = parser.parse_args()

    if args.load_geocoded:
        logger.info("Loading pre-geocoded data from CSV files.")
        load_geocoded_data_from_csv(args.load_geocoded[0], args.load_geocoded[1])
    elif args.update_geocoding:
        logger.info("Updating geocoding data.")
        update_geocoding_data()
    else:
        logger.info("Starting data processing")
        process_and_insert_historical_data(csv_file_path)

        for county, file_path in json_file_paths.items():
            logger.info(f"Processing listing data for {county}")
            process_and_insert_listing_data(file_path, args.skip_geocoding, county)

        logger.info("Merging data in the database.")
        merge_data_in_database()

        if not args.skip_geocoding:
            logger.info("Updating geocoding data.")
            update_geocoding_data()
        else:
            logger.info("Skipping geocoding update as requested.")

    logger.info("Data processing completed.")


if __name__ == "__main__":
    main()
