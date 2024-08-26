import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from src.data_generator.synthethic_user_generator import generate_synthetic_user_profiles
from src.database.models.synthetic_user import SyntheticUser, Base

# Use an in-memory SQLite database for testing
TEST_DATABASE_URL = "sqlite:///:memory:"

@pytest.fixture(scope="function")
def db_session():
    engine = create_engine(TEST_DATABASE_URL)
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    session = SessionLocal()
    yield session
    session.close()

def test_generate_synthetic_user_profiles(db_session, monkeypatch):
    # Mock the SessionLocal to use our test session
    monkeypatch.setattr("src.data_generator.synthethic_user_generator.SessionLocal", lambda: db_session)

    # Generate synthetic users
    num_users = 100
    generate_synthetic_user_profiles(num_users)

    # Verify the number of users created
    users = db_session.query(SyntheticUser).all()
    assert len(users) == num_users

def test_synthetic_user_data_types(db_session, monkeypatch):
    monkeypatch.setattr("src.data_generator.synthethic_user_generator.SessionLocal", lambda: db_session)

    generate_synthetic_user_profiles(10)

    users = db_session.query(SyntheticUser).all()
    for user in users:
        assert isinstance(user.income, float)
        assert isinstance(user.savings, float)
        assert isinstance(user.preferred_location, str)
        assert isinstance(user.desired_property_type, str)
        assert isinstance(user.must_have_features, str)
        assert isinstance(user.nice_to_have_features, str)
        assert isinstance(user.max_commute_time, int)
        assert isinstance(user.family_size, int)

def test_synthetic_user_value_ranges(db_session, monkeypatch):
    monkeypatch.setattr("src.data_generator.synthethic_user_generator.SessionLocal", lambda: db_session)

    generate_synthetic_user_profiles(100)

    users = db_session.query(SyntheticUser).all()
    for user in users:
        assert 10000 <= user.income <= 130000  # Assuming 4 standard deviations from the mean
        assert 0 <= user.savings <= 70000  # Assuming 4 standard deviations from the mean
        assert user.preferred_location in ['Urban', 'Suburban', 'Rural']
        assert user.desired_property_type in ['Apartment', 'House', 'Condo']
        assert user.must_have_features in ['Garden', 'Parking', 'Swimming Pool', 'Gym', 'None']
        assert user.nice_to_have_features in ['Balcony', 'Fireplace', 'Walk-in Closet', 'None']
        assert 10 <= user.max_commute_time <= 60
        assert 1 <= user.family_size <= 5

def test_synthetic_user_unique_ids(db_session, monkeypatch):
    monkeypatch.setattr("src.data_generator.synthethic_user_generator.SessionLocal", lambda: db_session)

    generate_synthetic_user_profiles(100)

    users = db_session.query(SyntheticUser).all()
    user_ids = [user.id for user in users]
    assert len(user_ids) == len(set(user_ids))  # All IDs should be unique

@pytest.mark.skip(reason="Skipping edge cases for now")
def test_error_handling(db_session, monkeypatch):
    def mock_add(*args, **kwargs):
        raise Exception("Database error")

    monkeypatch.setattr("src.data_generator.synthethic_user_generator.SessionLocal", lambda: db_session)
    monkeypatch.setattr(db_session, "add", mock_add)

    with pytest.raises(Exception):
        generate_synthetic_user_profiles(10)

    # Verify that no users were added due to the rollback
    users = db_session.query(SyntheticUser).all()
    assert len(users) == 0
