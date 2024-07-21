from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
import os
from tensorflow.keras.models import load_model
import joblib

app = Flask(__name__)

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

    # Preprocess user input
    user_input = preprocess_user_input(user_profile)

    # Prepare property data
    property_features = ['price', 'size_sq_ft', 'has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural']
    property_features.extend([col for col in property_data.columns if col.startswith('Property Type_')])

    X_property = property_data[property_features]
    scaled_property_features = scaler_property.transform(X_property)

    # Generate predictions
    predictions = model.predict([scaled_property_features, np.repeat(user_input, len(property_data), axis=0)])

    # Get top 5 recommendations
    top_indices = np.argsort(predictions.flatten())[-5:][::-1]
    recommendations = property_data.iloc[top_indices]

    # Convert to list of dictionaries for template rendering
    return recommendations.to_dict('records')

if __name__ == '__main__':
    app.run(debug=True)