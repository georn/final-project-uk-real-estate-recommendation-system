from data_loader import load_data
from model_builder import build_model
from model_trainer import train_model, plot_training_history, save_trained_model
from model_evaluator import evaluate_model
import joblib
import os

def main():
    # Load and preprocess data
    property_file_path = '../../data/ml-ready-data/ml_ready_data.csv'
    user_file_path = '../../data/synthetic_user_profiles/synthetic_user_profiles.csv'
    X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test = load_data(property_file_path, user_file_path)

    # Build the model
    model = build_model(X_property_train.shape[1], X_user_train.shape[1])
    model.summary()

    # Train the model
    history = train_model(model, [X_property_train, X_user_train], y_train, [X_property_test, X_user_test], y_test)

    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    evaluate_model(model, [X_property_test, X_user_test], y_test)

    # Save the trained model
    models_dir = '../../models'
    os.makedirs(models_dir, exist_ok=True)
    model_save_path = os.path.join(models_dir, 'property_recommendation_model.h5')
    save_trained_model(model, model_save_path)

if __name__ == "__main__":
    main()