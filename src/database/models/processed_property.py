from sqlalchemy import Column, Integer, Float, Boolean, ForeignKey, Enum
from src.database.database import Base
import enum

class EncodedTenure(enum.IntEnum):
    FREEHOLD = 0
    LEASEHOLD = 1
    UNKNOWN = 2

class ProcessedProperty(Base):
    __tablename__ = 'processed_properties'

    id = Column(Integer, primary_key=True, index=True)
    original_id = Column(Integer, ForeignKey('merged_properties.id'))
    price = Column(Float)
    size_sq_ft = Column(Float)
    year = Column(Integer)
    month = Column(Integer)
    day_of_week = Column(Integer)
    price_to_income_ratio = Column(Float)
    price_to_savings_ratio = Column(Float)
    affordability_score = Column(Float)
    has_garden = Column(Boolean)
    has_parking = Column(Boolean)
    location_Urban = Column(Boolean)
    location_Suburban = Column(Boolean)
    location_Rural = Column(Boolean)
    latitude = Column(Float)
    longitude = Column(Float)
    epc_rating_encoded = Column(Integer)
    property_type_Detached = Column(Boolean)
    property_type_Semi_Detached = Column(Boolean)
    property_type_Terraced = Column(Boolean)
    property_type_Flat_Maisonette = Column(Boolean)
    property_type_Other = Column(Boolean)
    bedrooms = Column(Integer)
    bathrooms = Column(Integer)
    tenure = Column(Enum(EncodedTenure))

    def __repr__(self):
        return f"<ProcessedProperty(id={self.id}, original_id={self.original_id}, price={self.price})>"
