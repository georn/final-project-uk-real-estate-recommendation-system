from sqlalchemy import Column, BigInteger, Integer, Float, String, JSON, Boolean

from src.database.database import Base


class ProcessedProperty(Base):
    __tablename__ = 'processed_properties'

    id = Column(BigInteger, primary_key=True, index=True)
    original_id = Column(BigInteger)
    price = Column(Float)
    size_sq_ft = Column(Float, nullable=True)
    year = Column(Integer)
    month = Column(Integer)
    day_of_week = Column(Integer)
    price_to_income_ratio = Column(Float, nullable=True)
    price_to_savings_ratio = Column(Float, nullable=True)
    affordability_score = Column(Float, nullable=True)
    has_garden = Column(Boolean)
    has_parking = Column(Boolean)
    location_Urban = Column(Boolean)
    location_Suburban = Column(Boolean)
    location_Rural = Column(Boolean)
    latitude = Column(Float, nullable=True)
    longitude = Column(Float, nullable=True)
    epc_rating_encoded = Column(Float, nullable=True)
    property_type = Column(String)
    additional_features = Column(JSON)

    def __repr__(self):
        return f"<ProcessedProperty(id={self.id}, original_id={self.original_id}, price={self.price})>"
