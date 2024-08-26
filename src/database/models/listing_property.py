from sqlalchemy import Column, Integer, String, Float, JSON

from src.database.database import Base


class ListingProperty(Base):
    __tablename__ = 'listing_properties'

    id = Column(Integer, primary_key=True, index=True)
    property_url = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    address = Column(String, nullable=False)
    price = Column(Float, nullable=False)
    pricing_qualifier = Column(String)
    listing_time = Column(String)
    property_type = Column(String, nullable=False)
    bedrooms = Column(String)
    bathrooms = Column(String)
    epc_rating = Column(String)
    size = Column(String)
    features = Column(JSON)
    latitude = Column(Float)
    longitude = Column(Float)

    def __repr__(self):
        return f"<ListingProperty(id={self.id}, address='{self.address}', price={self.price})>"
