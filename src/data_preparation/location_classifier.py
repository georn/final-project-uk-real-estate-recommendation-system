import numpy as np


def classify_location(latitude, longitude, county):
    # Define major urban centers for each county
    URBAN_CENTERS = {
        'buckinghamshire': [
            {'name': 'High Wycombe', 'lat': 51.6285, 'lon': -0.7489},
            {'name': 'Aylesbury', 'lat': 51.8156, 'lon': -0.8125},
            {'name': 'Milton Keynes', 'lat': 52.0406, 'lon': -0.7594},
            {'name': 'Amersham', 'lat': 51.6746, 'lon': -0.6076},
        ],
        'bedfordshire': [
            {'name': 'Luton', 'lat': 51.8787, 'lon': -0.4200},
            {'name': 'Bedford', 'lat': 52.1364, 'lon': -0.4668},
        ],
        'hertfordshire': [
            {'name': 'Watford', 'lat': 51.6565, 'lon': -0.3903},
            {'name': 'St Albans', 'lat': 51.7517, 'lon': -0.3411},
        ],
        'oxfordshire': [
            {'name': 'Oxford', 'lat': 51.7520, 'lon': -1.2577},
            {'name': 'Banbury', 'lat': 52.0602, 'lon': -1.3403},
        ],
        'berkshire': [
            {'name': 'Reading', 'lat': 51.4543, 'lon': -0.9781},
            {'name': 'Slough', 'lat': 51.5105, 'lon': -0.5950},
        ],
        'northamptonshire': [
            {'name': 'Northampton', 'lat': 52.2405, 'lon': -0.9027},
            {'name': 'Kettering', 'lat': 52.3981, 'lon': -0.7271},
        ]
    }

    # Check if latitude and longitude are valid
    if latitude is None or longitude is None or not isinstance(latitude, (int, float)) or not isinstance(longitude,
                                                                                                         (int, float)):
        return False, False, False  # Unknown location

    if county.lower() in URBAN_CENTERS:
        centers = URBAN_CENTERS[county.lower()]

        # Calculate distances to urban centers
        distances = [np.sqrt((center['lat'] - latitude) ** 2 + (center['lon'] - longitude) ** 2)
                     for center in centers]
        min_distance = min(distances)

        # Classify based on distance
        if min_distance < 0.05:  # Approximately 5 km
            return True, False, False  # Urban
        elif min_distance < 0.1:  # Approximately 10 km
            return False, True, False  # Suburban
        else:
            return False, False, True  # Rural
    else:
        # If county is not recognized, return Unknown
        return False, False, False

# Usage in data processing:
# location_Urban, location_Suburban, location_Rural = classify_location(latitude, longitude)
