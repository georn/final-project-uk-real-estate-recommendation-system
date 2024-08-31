import json
import logging
from datetime import date

import pandas as pd

from sqlalchemy.dialects.postgresql import insert

from src.data_standardiser.data_processing import standardise_price, extract_number, standardize_epc_rating, \
    standardize_size
from src.data_standardiser.geocoding import geocode_address
from src.data_standardiser.property_utils import extract_tenure, standardize_property_type
from src.data_standardiser.utils import enum_to_string
from src.database.database import SessionLocal
from src.database.models.historical_property import HistoricalProperty
from src.database.models.listing_property import ListingProperty, Tenure
from src.database.models.merged_property import MergedProperty, Tenure as MergedTenure
from src.database.models.historical_property import PropertyType, PropertyAge, PropertyDuration, PPDCategoryType, \
    RecordStatus

logger = logging.getLogger(__name__)

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

            tenure = extract_tenure(item.get('features'))

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
                longitude=longitude,
                tenure=tenure
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
                    'longitude': stmt.excluded.longitude,
                    'tenure': stmt.excluded.tenure
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

def merge_data_in_database():
    db = SessionLocal()
    try:
        db.query(MergedProperty).delete()

        # fetch all historical properties
        historical_properties = db.query(HistoricalProperty).all()

        for hp in historical_properties:
            # Try to find a matching listing property
            lp = db.query(ListingProperty).filter_by(address=hp.postal_code).first()

            # Determine tenure
            if hp.duration == PropertyDuration.FREEHOLD:
                tenure = MergedTenure.FREEHOLD
            elif hp.duration == PropertyDuration.LEASEHOLD:
                tenure = MergedTenure.LEASEHOLD
            elif lp and lp.tenure != Tenure.UNKNOWN:
                tenure = MergedTenure[lp.tenure.name]
            else:
                tenure = MergedTenure.UNKNOWN

            merged_property = MergedProperty(
                historical_id=hp.id,
                listing_id=lp.id if lp else None,
                price=hp.price or (lp.price if lp else None),
                postal_code=hp.postal_code,
                property_type=standardize_property_type(
                    enum_to_string(hp.property_type) or (lp.property_type if lp else None)),
                date=hp.date_of_transaction,
                property_age=enum_to_string(hp.property_age),
                tenure=tenure,
                bedrooms=extract_number(lp.bedrooms if lp else None),
                bathrooms=extract_number(lp.bathrooms if lp else None),
                epc_rating=standardize_epc_rating(lp.epc_rating if lp else None),
                size=standardize_size(lp.size if lp else None),
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
                property_type=standardize_property_type(lp.property_type),  # Apply standardization here
                date=date.today(),  # Use current date for listing properties
                bedrooms=extract_number(lp.bedrooms),
                bathrooms=extract_number(lp.bathrooms),
                epc_rating=standardize_epc_rating(lp.epc_rating),
                size=standardize_size(lp.size),
                features=lp.features,
                data_source='listing',
                listing_time=lp.listing_time,
                latitude=lp.latitude,
                longitude=lp.longitude,
                tenure=MergedTenure[lp.tenure.name]
            )
            db.add(merged_property)

        db.commit()
        logger.info("Merged data created in the database.")
    except Exception as e:
        db.rollback()
        logger.error(f"An error occurred while merging data: {e}")
    finally:
        db.close()

