import numpy as np
import pandas as pd

def generate_synthetic_user_profiles(num_users=1000):
    np.random.seed(42)
    incomes = np.random.normal(70000, 15000, num_users)  # mean income 70k with sd of 15k
    savings = np.random.normal(30000, 10000, num_users)  # mean savings 30k with sd of 10k
    locations = np.random.choice(['Urban', 'Suburban', 'Rural'], num_users)
    property_types = np.random.choice(['Apartment', 'House', 'Condo'], num_users)
    must_have_features = np.random.choice(['Garden', 'Parking', 'Swimming Pool', 'Gym', 'None'], num_users)
    nice_to_have_features = np.random.choice(['Balcony', 'Fireplace', 'Walk-in Closet', 'None'], num_users)
    commute_times = np.random.randint(10, 60, num_users)  # maximum commute time between 10 and 60 minutes
    family_sizes = np.random.randint(1, 6, num_users)  # family size between 1 and 5 members

    user_profiles = pd.DataFrame({
        'Income': incomes,
        'Savings': savings,
        'PreferredLocation': locations,
        'DesiredPropertyType': property_types,
        'MustHaveFeatures': must_have_features,
        'NiceToHaveFeatures': nice_to_have_features,
        'MaxCommuteTime': commute_times,
        'FamilySize': family_sizes
    })

    return user_profiles

if __name__ == "__main__":
    user_profiles = generate_synthetic_user_profiles()
    user_profiles.to_csv('./synthetic_user_profiles.csv', index=False)
    print("Synthetic user profiles generated and saved to synthetic_user_profiles.csv")
