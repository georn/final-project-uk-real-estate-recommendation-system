from sqlalchemy import Column, Integer, String, Float, JSON
from src.database.database import Base

class ListingProperty(Base):
    __tablename__ = 'listing_properties'

    id = Column(Integer, primary_key=True, index=True)
    property_url = Column(String, unique=True, nullable=False)
    title = Column(String, nullable=False)
    address = Column(String, nullable=False)
    price = Column(String, nullable=False)  # Keep as string to preserve formatting
    pricing_qualifier = Column(String)
    listing_time = Column(String)
    property_type = Column(String, nullable=False)
    bedrooms = Column(String)  # Keep as string because it's in the format "3 bed"
    bathrooms = Column(String)  # Keep as string because it's in the format "2 bath"
    epc_rating = Column(String)
    size = Column(String)
    features = Column(JSON)  # Store as JSON to keep the list structure

    def __repr__(self):
        return f"<ListingProperty(id={self.id}, address='{self.address}', price={self.price})>"
