from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.ext.declarative import declarative_base
import os

# Check if we're running in Docker
IN_DOCKER = os.environ.get('DOCKER_CONTAINER', False)

# Get the DATABASE_URL from environment variables
if IN_DOCKER:
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@db:5432/property_db")
else:
    # Use localhost when running outside Docker
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:password@localhost:5432/property_db")

# Create the SQLAlchemy engine
engine = create_engine(DATABASE_URL)

# Create a SessionLocal class
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create a Base class
Base = declarative_base()

def get_db():
    """
    Generator function to get a database session.
    Yields a session and ensures it's closed after use.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
