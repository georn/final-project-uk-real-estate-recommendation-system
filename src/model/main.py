from data_loader import load_data
from model_builder import build_model
from model_trainer import train_model, plot_training_history
from model_evaluator import evaluate_model

def main():
    # Load and preprocess data
    file_path = '../../data/preprocessed-data/preprocessed.csv'
    X_train, X_val, y_train, y_val, scaler_features, scaler_target = load_data(file_path)

    # Build the model
    model = build_model(X_train.shape[1])
    model.summary()

    # Train the model
    history = train_model(model, X_train, y_train, X_val, y_val)

    # Plot training history
    plot_training_history(history)

    # Evaluate the model
    mae, rmse = evaluate_model(model, X_val, y_val, scaler_target)
    print(f'Mean Absolute Error: {mae}')
    print(f'Root Mean Squared Error: {rmse}')

if __name__ == "__main__":
    main()