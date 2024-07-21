from data_loader import load_data
from model_builder import build_model
from model_trainer import train_model, plot_training_history
from model_evaluator import evaluate_model

def main():
    # Load and preprocess data
    property_file_path = '../../data/ml-ready-data/ml_ready_data.csv'
    user_file_path = '../../data/synthetic_user_profiles/synthetic_user_profiles.csv'
    X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test, scaler_property, scaler_user = load_data(property_file_path, user_file_path)

    # Build the model
    model = build_model(X_property_train.shape[1], X_user_train.shape[1])
    model.summary()

    # Train the model
    history = train_model(model, [X_property_train, X_user_train], y_train, [X_property_test, X_user_test], y_test)

    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    evaluate_model(model, [X_property_test, X_user_test], y_test)

if __name__ == "__main__":
    main()