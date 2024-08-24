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

    # Get the feature names that the scaler was fitted with
    scaler_feature_names = scaler_user.feature_names_in_

    # Ensure user_features has the same features as the scaler, in the same order
    user_features_scaled = pd.DataFrame(index=user_features.index, columns=scaler_feature_names)
    for feature in scaler_feature_names:
        if feature in user_features.columns:
            user_features_scaled[feature] = user_features[feature]
        else:
            logger.warning(f"User feature '{feature}' not found. Adding default column with 0s.")
            user_features_scaled[feature] = 0

    # Scale user features
    scaled_user_features = scaler_user.transform(user_features_scaled)
    logger.info(f"Scaled user features shape: {scaled_user_features.shape}")
    logger.info(f"User features: {scaler_feature_names.tolist()}")

    # Ensure we have 4 features for user input
    if scaled_user_features.shape[1] < 4:
        padding = np.zeros((scaled_user_features.shape[0], 4 - scaled_user_features.shape[1]))
        scaled_user_features = np.hstack((scaled_user_features, padding))
        logger.info(f"Added padding to user features. New shape: {scaled_user_features.shape}")

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
        'Property Type_D', 'Property Type_F', 'Property Type_O', 'Property Type_S'
    ]

    # Ensure all required features exist in property data
    for feature in property_features:
        if feature not in property_data.columns:
            logger.warning(f"Feature '{feature}' not found in property data. Adding default column with 0s.")
            property_data[feature] = 0

    logger.info(f"Property features used: {property_features}")

    X_property = property_data[property_features]

    # Get the feature names that the scaler was fitted with
    scaler_feature_names = scaler_property.feature_names_in_

    # Ensure X_property has the same features as the scaler, in the same order
    X_property_scaled = pd.DataFrame(index=X_property.index, columns=scaler_feature_names)
    for feature in scaler_feature_names:
        if feature in X_property.columns:
            X_property_scaled[feature] = X_property[feature]
        else:
            logger.warning(f"Feature '{feature}' not found in property data. Adding default column with 0s.")
            X_property_scaled[feature] = 0

    # Now transform the data
    scaled_property_features = scaler_property.transform(X_property_scaled)

    logger.info(f"Scaled property features shape: {scaled_property_features.shape}")
    logger.info(f"User input shape: {user_input.shape}")

    # Ensure we have 17 features for property input
    if scaled_property_features.shape[1] < 17:
        padding = np.zeros((scaled_property_features.shape[0], 17 - scaled_property_features.shape[1]))
        scaled_property_features = np.hstack((scaled_property_features, padding))
        logger.info(f"Added padding to property features. New shape: {scaled_property_features.shape}")

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
    app.run(debug=True, host='0.0.0.0')
