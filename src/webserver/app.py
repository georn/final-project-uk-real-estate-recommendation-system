from flask import Flask, request, render_template, jsonify
import pandas as pd
import os

app = Flask(__name__)

# Route for the home page to display the form
@app.route('/')
def home():
    return render_template('user_form.html')

# Route to handle form submission
@app.route('/submit_user_profile', methods=['POST'])
def submit_user_profile():
    user_profile = {
        'Income': request.form['income'],
        'Savings': request.form['savings'],
        'PreferredLocation': request.form['preferred_location'],
        'DesiredPropertyType': request.form['property_type'],
        'MustHaveFeatures': request.form.getlist('must_have_features'),
        'NiceToHaveFeatures': request.form.getlist('nice_to_have_features'),
        'MaxCommuteTime': request.form['commute_time'],
        'FamilySize': request.form['family_size']
    }

    # Convert to DataFrame
    user_profile_df = pd.DataFrame([user_profile])

    # Save to CSV (append mode)
    csv_path = os.path.join(os.path.dirname(__file__), 'user_profiles.csv')
    user_profile_df.to_csv(csv_path, mode='a', header=not os.path.exists(csv_path), index=False)

    # Generate and return recommendations (mocked for now)
    recommendations = generate_recommendations(user_profile)

    return render_template('recommendations.html', recommendations=recommendations)

# Mock function to generate recommendations
def generate_recommendations(user_profile):
    # For simplicity, returning mock recommendations
    properties = [
        {'Location': 'Urban', 'Type': 'Apartment', 'Price': 250000, 'Features': 'Parking'},
        {'Location': 'Suburban', 'Type': 'House', 'Price': 300000, 'Features': 'Garden'},
        {'Location': 'Rural', 'Type': 'Condo', 'Price': 150000, 'Features': 'Swimming Pool'}
    ]
    return properties

if __name__ == '__main__':
    app.run(debug=True)
