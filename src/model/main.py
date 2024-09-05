import logging
import os

from data_loader import load_data
from model_builder import build_model
from model_evaluator import evaluate_model, plot_feature_importance, analyze_misclassifications
from model_trainer import train_model, plot_training_history, save_trained_model


def main():
    # Load and preprocess data
    sample_size = 1000
    pairs_per_user = 10

    result = load_data(sample_size=sample_size, pairs_per_user=pairs_per_user)

    if result is None or any(x is None for x in result):
        logging.error("Data loading failed. Exiting.")
    else:
        X_property_train, X_property_test, X_user_train, X_user_test, y_train, y_test = result

        # Build the model
        model = build_model(X_property_train.shape[1], X_user_train.shape[1])
        model.summary()

        # Train the model
        history = train_model(model, [X_property_train, X_user_train], y_train, [X_property_test, X_user_test], y_test)

        # Plot training history
        plot_training_history(history)

        # Evaluate the model
        accuracy, precision, recall, f1 = evaluate_model(model, [X_property_test, X_user_test], y_test)

        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")

        # Plot feature importance
        feature_names = [f'property_{i}' for i in range(X_property_train.shape[1])] + \
                        [f'user_{i}' for i in range(X_user_train.shape[1])]
        plot_feature_importance(model, feature_names)

        analyze_misclassifications(model, [X_property_test, X_user_test], y_test, feature_names)

        # Save the trained model
        models_dir = '../../models'
        os.makedirs(models_dir, exist_ok=True)
        model_save_path = os.path.join(models_dir, 'property_recommendation_model.keras')
        save_trained_model(model, model_save_path)


if __name__ == "__main__":
    main()
