import numpy as np
from sqlalchemy.orm import Session

from src.database.database import SessionLocal, engine
from src.database.models.synthetic_user import SyntheticUser, Base, TenurePreference


def generate_synthetic_user_profiles(num_users=1000):
    np.random.seed(42)
    incomes = np.random.normal(70000, 15000, num_users)
    savings = np.random.normal(30000, 10000, num_users)
    locations = np.random.choice(['Urban', 'Suburban', 'Rural'], num_users)
    property_types = np.random.choice(['Apartment', 'House', 'Condo'], num_users)
    must_have_features = np.random.choice(['Garden', 'Parking', 'Swimming Pool', 'Gym', 'None'], num_users)
    nice_to_have_features = np.random.choice(['Balcony', 'Fireplace', 'Walk-in Closet', 'None'], num_users)
    commute_times = np.random.randint(10, 60, num_users)
    family_sizes = np.random.randint(1, 6, num_users)
    tenure_preferences = np.random.choice(
        [TenurePreference.FREEHOLD, TenurePreference.LEASEHOLD, TenurePreference.NO_PREFERENCE], num_users)

    Base.metadata.create_all(bind=engine)
    db: Session = SessionLocal()

    try:
        for i in range(num_users):
            user = SyntheticUser(
                income=float(incomes[i]),
                savings=float(savings[i]),
                preferred_location=locations[i],
                desired_property_type=property_types[i],
                must_have_features=must_have_features[i],
                nice_to_have_features=nice_to_have_features[i],
                max_commute_time=int(commute_times[i]),
                family_size=int(family_sizes[i]),
                tenure_preference=tenure_preferences[i]
            )
            db.add(user)

        db.commit()
        print(f"{num_users} synthetic user profiles generated and saved to the database.")
    except Exception as e:
        print(f"An error occurred: {e}")
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    generate_synthetic_user_profiles()
