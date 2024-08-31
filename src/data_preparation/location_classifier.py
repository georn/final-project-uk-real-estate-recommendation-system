import numpy as np

def classify_location(latitude, longitude):
    # Define approximate boundaries for Buckinghamshire
    # These are rough estimates and should be refined with more accurate data
    BUCKINGHAMSHIRE_BOUNDS = {
        'min_lat': 51.5,
        'max_lat': 52.1,
        'min_lon': -1.0,
        'max_lon': -0.5
    }

    # Define major urban centers in Buckinghamshire
    URBAN_CENTERS = [
        {'name': 'High Wycombe', 'lat': 51.6285, 'lon': -0.7489},
        {'name': 'Aylesbury', 'lat': 51.8156, 'lon': -0.8125},
        {'name': 'Milton Keynes', 'lat': 52.0406, 'lon': -0.7594},
        {'name': 'Amersham', 'lat': 51.6746, 'lon': -0.6076},
    ]

    # Check if the property is within Buckinghamshire
    if (BUCKINGHAMSHIRE_BOUNDS['min_lat'] <= latitude <= BUCKINGHAMSHIRE_BOUNDS['max_lat'] and
            BUCKINGHAMSHIRE_BOUNDS['min_lon'] <= longitude <= BUCKINGHAMSHIRE_BOUNDS['max_lon']):

        # Calculate distances to urban centers
        distances = [np.sqrt((center['lat'] - latitude)**2 + (center['lon'] - longitude)**2)
                     for center in URBAN_CENTERS]
        min_distance = min(distances)

        # Classify based on distance
        if min_distance < 0.05:  # Approximately 5 km
            return True, False, False  # Urban
        elif min_distance < 0.1:  # Approximately 10 km
            return False, True, False  # Suburban
        else:
            return False, False, True  # Rural
    else:
        # Outside Buckinghamshire
        return False, False, False

# Usage in data processing:
# location_Urban, location_Suburban, location_Rural = classify_location(latitude, longitude)
