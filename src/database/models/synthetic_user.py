import enum

from sqlalchemy import Column, Integer, String, Float, Enum

from src.database.database import Base


class TenurePreference(enum.Enum):
    FREEHOLD = 'Freehold'
    LEASEHOLD = 'Leasehold'
    NO_PREFERENCE = 'No Preference'


class SyntheticUser(Base):
    __tablename__ = 'synthetic_users'

    id = Column(Integer, primary_key=True, index=True)
    income = Column(Float, nullable=False)
    savings = Column(Float, nullable=False)
    preferred_location = Column(String, nullable=False)
    desired_property_type = Column(String, nullable=False)
    must_have_features = Column(String, nullable=False)
    nice_to_have_features = Column(String, nullable=False)
    max_commute_time = Column(Integer, nullable=False)
    family_size = Column(Integer, nullable=False)
    tenure_preference = Column(Enum(TenurePreference), nullable=False)

    def __repr__(self):
        return f"<SyntheticUser(id={self.id}, income={self.income}, preferred_location='{self.preferred_location}', tenure_preference='{self.tenure_preference.value}')>"
