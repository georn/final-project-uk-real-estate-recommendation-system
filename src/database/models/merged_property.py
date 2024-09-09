import enum

from sqlalchemy import Column, Integer, String, Float, Date, JSON, ForeignKey, Enum

from src.database.database import Base


class Tenure(enum.Enum):
    FREEHOLD = 'Freehold'
    LEASEHOLD = 'Leasehold'
    UNKNOWN = 'Unknown'


class MergedProperty(Base):
    __tablename__ = 'merged_properties'

    id = Column(Integer, primary_key=True, index=True)
    historical_id = Column(String, ForeignKey('historical_properties.id'), nullable=True)
    listing_id = Column(Integer, ForeignKey('listing_properties.id'), nullable=True)
    price = Column(Float)
    postal_code = Column(String)
    property_type = Column(String)
    date = Column(Date)
    property_age = Column(String)
    tenure = Column(Enum(Tenure))
    bedrooms = Column(Integer)
    bathrooms = Column(Integer)
    epc_rating = Column(String)
    size = Column(String)
    features = Column(JSON)
    data_source = Column(String)
    listing_time = Column(String)
    latitude = Column(Float)
    longitude = Column(Float)
    county = Column(String)

    def __repr__(self):
        return f"<MergedProperty(id={self.id}, postal_code='{self.postal_code}', price={self.price}, county='{self.county}')>"
