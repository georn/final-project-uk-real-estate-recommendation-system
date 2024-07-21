import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(file_path):
    # Load the preprocessed data
    df = pd.read_csv(file_path)

    # Select relevant features for the model
    features = df[['Price', 'Property Type', 'latitude', 'longitude']]

    # Handle missing values
    features = features.dropna()

    # Convert categorical features to numeric
    features = pd.get_dummies(features, columns=['Property Type'])

    # Ensure the target variable matches the processed features dataframe
    target = df.loc[features.index, 'Price']

    return features, target

def split_and_scale_data(features, target):
    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(features, target, test_size=0.2, random_state=42)

    # Normalize the features and target
    scaler_features = StandardScaler()
    X_train = scaler_features.fit_transform(X_train)
    X_val = scaler_features.transform(X_val)

    scaler_target = StandardScaler()
    y_train = scaler_target.fit_transform(y_train.values.reshape(-1, 1)).flatten()
    y_val = scaler_target.transform(y_val.values.reshape(-1, 1)).flatten()

    return X_train, X_val, y_train, y_val, scaler_features, scaler_target

def load_data(file_path):
    features, target = load_and_preprocess_data(file_path)
    return split_and_scale_data(features, target)