import logging

import joblib
import numpy as np
import pandas as pd
from flask import Flask, request, render_template
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from tensorflow.keras.models import load_model

from src.database.database import DATABASE_URL
from src.database.models.processed_property import ProcessedProperty

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model when the app starts
model = load_model('../../models/property_recommendation_model.keras')

# Load scalers
scaler_property = joblib.load('../../models/scaler_property.joblib')
scaler_user = joblib.load('../../models/scaler_user.joblib')

# Set up database connection
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# Load property data
def load_property_data():
    db = SessionLocal()
    try:
        properties = db.query(ProcessedProperty).all()
        return pd.DataFrame([p.__dict__ for p in properties])
    finally:
        db.close()


# Load property data when the app starts
property_data = load_property_data()


@app.route('/')
def home():
    return render_template('user_form.html')


@app.route('/submit_user_profile', methods=['POST'])
def submit_user_profile():
    user_profile = {
        'Income': float(request.form['income']),
        'Savings': float(request.form['savings']),
        'PreferredLocation': request.form['preferred_location'],
        'DesiredPropertyType': request.form['desired_property_type'],
        'MustHaveFeatures': request.form.getlist('must_have_features'),
        'NiceToHaveFeatures': request.form.getlist('nice_to_have_features'),
        'MaxCommuteTime': int(request.form['max_commute_time']),
        'FamilySize': int(request.form['family_size']),
        'TenurePreference': request.form['tenure_preference']
    }

    # Generate and return recommendations
    recommendations = generate_recommendations(user_profile)

    if not recommendations:
        return render_template('no_recommendations.html')

    return render_template('recommendations.html', recommendations=recommendations)


def preprocess_user_input(user_profile):
    user_features = pd.DataFrame({
        'income': [user_profile['Income']],
        'savings': [user_profile['Savings']],
        'max_commute_time': [user_profile['MaxCommuteTime']],
        'family_size': [user_profile['FamilySize']],
        'tenure_preference': [user_profile['TenurePreference']]
    })

    # Convert tenure preference to numeric
    tenure_mapping = {'FREEHOLD': 0, 'LEASEHOLD': 1, 'NO_PREFERENCE': 2}
    user_features['tenure_preference'] = user_features['tenure_preference'].map(tenure_mapping)

    # Scale user features
    scaled_user_features = scaler_user.transform(user_features)
    logger.info(f"Scaled user features shape: {scaled_user_features.shape}")

    return scaled_user_features


def generate_recommendations(user_profile):
    global property_data, scaler_property, model

    logger.info(f"Generating recommendations for user profile: {user_profile}")
    logger.info(f"Property data columns: {property_data.columns.tolist()}")

    # Preprocess user input
    user_input = preprocess_user_input(user_profile)

    # Define the exact features the model was trained on
    property_features = [
        'price', 'size_sq_ft', 'year', 'month', 'day_of_week',
        'price_to_income_ratio', 'price_to_savings_ratio', 'affordability_score',
        'has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural',
        'latitude', 'longitude', 'epc_rating_encoded',
        'property_type_Detached', 'property_type_Semi_Detached', 'property_type_Terraced',
        'property_type_Flat_Maisonette', 'property_type_Other',
        'bedrooms', 'bathrooms', 'tenure'
    ]

    # Ensure all required features exist in property data
    for feature in property_features:
        if feature not in property_data.columns:
            logger.warning(f"Feature '{feature}' not found in property data. Adding default column with 0s.")
            property_data[feature] = 0

    # Add log transformations
    property_data['log_price'] = np.log1p(property_data['price'])
    property_data['log_size'] = np.log1p(property_data['size_sq_ft'])

    # Add log_price and log_size to the features list
    property_features.extend(['log_price', 'log_size'])

    # Select features in the correct order
    X_property = property_data[property_features]

    # Scale property features
    scaled_property_features = scaler_property.transform(X_property)

    logger.info(f"Scaled property features shape: {scaled_property_features.shape}")
    logger.info(f"User input shape: {user_input.shape}")

    # Generate predictions
    predictions = model.predict([scaled_property_features, np.repeat(user_input, len(property_data), axis=0)])

    # Apply filters based on user preferences
    mask = (property_data['price'] <= user_profile['Income'] * 4 + user_profile['Savings'])

    # Check if location column exists before applying filter
    location_column = f"location_{user_profile['PreferredLocation']}"
    if location_column in property_data.columns:
        mask &= (property_data[location_column] == 1)
    else:
        logger.warning(f"Location column '{location_column}' not found in property data")

    # Check if property type column exists before applying filter
    property_type_column = f"property_type_{user_profile['DesiredPropertyType']}"
    if property_type_column in property_data.columns:
        mask &= (property_data[property_type_column] == 1)
    else:
        logger.warning(f"Property type column '{property_type_column}' not found in property data")

    # Apply must-have features filter
    for feature in user_profile['MustHaveFeatures']:
        if feature.lower() == 'garden' and 'has_garden' in property_data.columns:
            mask &= (property_data['has_garden'] == 1)
        elif feature.lower() == 'parking' and 'has_parking' in property_data.columns:
            mask &= (property_data['has_parking'] == 1)
        else:
            logger.warning(f"Feature '{feature}' not found in property data")

    filtered_predictions = predictions[mask]
    filtered_properties = property_data[mask]

    logger.info(f"Number of properties after filtering: {len(filtered_properties)}")

    if len(filtered_properties) == 0:
        logger.warning("No properties match the user's criteria.")
        return []

    # Get top 5 recommendations
    top_indices = np.argsort(filtered_predictions.flatten())[-5:][::-1]
    recommendations = filtered_properties.iloc[top_indices]

    logger.info(f"Number of recommendations generated: {len(recommendations)}")

    # Convert to list of dictionaries for template rendering
    return recommendations.to_dict('records')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001)
