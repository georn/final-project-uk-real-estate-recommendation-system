from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
import joblib
import logging

app = Flask(__name__)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load the model when the app starts
model = load_model('../../models/property_recommendation_model.h5')

# Load scalers
scaler_property = joblib.load('../../models/scaler_property.joblib')
scaler_user = joblib.load('../../models/scaler_user.joblib')

# Load property data
def load_property_data():
    property_data_path = '../../data/ml-ready-data/ml_ready_data.csv'
    return pd.read_csv(property_data_path)

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
        'DesiredPropertyType': request.form['property_type'],
        'MustHaveFeatures': request.form.getlist('must_have_features'),
        'NiceToHaveFeatures': request.form.getlist('nice_to_have_features'),
        'MaxCommuteTime': int(request.form['commute_time']),
        'FamilySize': int(request.form['family_size'])
    }

    # Convert to DataFrame
    user_profile_df = pd.DataFrame([user_profile])

    # Save to CSV (append mode)
    csv_path = os.path.join(os.path.dirname(__file__), 'user_profiles.csv')
    user_profile_df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

    # Generate and return recommendations
    recommendations = generate_recommendations(user_profile)

    return render_template('recommendations.html', recommendations=recommendations)

def preprocess_user_input(user_profile):
    user_features = pd.DataFrame({
        'Income': [user_profile['Income']],
        'Savings': [user_profile['Savings']],
        'MaxCommuteTime': [user_profile['MaxCommuteTime']],
        'FamilySize': [user_profile['FamilySize']]
    })

    # Scale user features
    scaled_user_features = scaler_user.transform(user_features)
    return scaled_user_features

def generate_recommendations(user_profile):
    global property_data

    logger.info(f"Generating recommendations for user profile: {user_profile}")
    logger.info(f"Property data columns: {property_data.columns.tolist()}")

    # Preprocess user input
    user_input = preprocess_user_input(user_profile)

    # Prepare property data
    property_features = ['price', 'size_sq_ft', 'has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural']
    property_features.extend([col for col in property_data.columns if col.startswith('Property Type_')])

    logger.info(f"Property features used: {property_features}")

    X_property = property_data[property_features]
    scaled_property_features = scaler_property.transform(X_property)

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
    property_type_column = f"Property Type_{user_profile['DesiredPropertyType']}"
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

    # Get top 5 recommendations
    top_indices = np.argsort(filtered_predictions.flatten())[-5:][::-1]
    recommendations = filtered_properties.iloc[top_indices]

    logger.info(f"Number of recommendations generated: {len(recommendations)}")

    # Convert to list of dictionaries for template rendering
    return recommendations.to_dict('records')

if __name__ == '__main__':
    app.run(debug=True)
