from sqlalchemy import Column, Integer, String, Date, Enum
from src.database.database import Base
import enum

class PropertyType(enum.Enum):
    DETACHED = 'D'
    SEMI_DETACHED = 'S'
    TERRACED = 'T'
    FLAT = 'F'
    OTHER = 'O'

class PropertyAge(enum.Enum):
    NEW = 'N'
    OLD = 'O'

class PropertyDuration(enum.Enum):
    FREEHOLD = 'F'
    LEASEHOLD = 'L'

class HistoricalProperty(Base):
    __tablename__ = 'historical_properties'

    id = Column(String, primary_key=True, index=True)  # Unique Transaction Identifier
    price = Column(Integer, nullable=False)
    date_of_transaction = Column(Date, nullable=False)
    postal_code = Column(String, nullable=False)
    property_type = Column(Enum(PropertyType), nullable=False)
    property_age = Column(Enum(PropertyAge), nullable=False)
    duration = Column(Enum(PropertyDuration), nullable=False)
    paon = Column(String)
    saon = Column(String)
    street = Column(String)
    locality = Column(String)
    town_city = Column(String)
    district = Column(String)
    ppd_category_type = Column(String)
    record_status = Column(String)

    def __repr__(self):
        return f"<HistoricalProperty(id='{self.id}', postal_code='{self.postal_code}', price={self.price})>"
