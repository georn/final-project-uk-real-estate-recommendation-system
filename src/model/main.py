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
        accuracy, precision, recall, f1 = evaluate_model(model, [X_property_test, X_user_test], y_test, True, True, True)

        logging.info(f"Accuracy: {accuracy:.4f}")
        logging.info(f"Precision: {precision:.4f}")
        logging.info(f"Recall: {recall:.4f}")
        logging.info(f"F1 Score: {f1:.4f}")

        # Plot feature importance
        feature_names = [
            'price', 'size_sq_ft', 'year', 'month', 'day_of_week',
            'price_to_income_ratio', 'price_to_savings_ratio', 'affordability_score',
            'has_garden', 'has_parking', 'location_Urban', 'location_Suburban', 'location_Rural',
            'latitude', 'longitude', 'epc_rating_encoded',
            'property_type_Detached', 'property_type_Semi_Detached', 'property_type_Terraced',
            'property_type_Flat_Maisonette', 'property_type_Other',
            'bedrooms', 'bathrooms', 'tenure', 'price_relative_to_county_avg',
            'county_buckinghamshire', 'county_bedfordshire', 'county_hertfordshire',
            'county_oxfordshire', 'county_berkshire', 'county_northamptonshire',
            'log_price', 'log_size',
            'income', 'savings', 'max_commute_time', 'family_size', 'tenure_preference'
        ]
        logging.info(f"Feature names: {feature_names}")
        logging.info(f"Number of features: {len(feature_names)}")
        logging.info(f"X_property_test shape: {X_property_test.shape}")
        logging.info(f"X_user_test shape: {X_user_test.shape}")
        logging.info(f"Total features: {X_property_test.shape[1] + X_user_test.shape[1]}")
        importances = plot_feature_importance(model, [X_property_test, X_user_test], y_test, feature_names)

        analyze_misclassifications(model, [X_property_test, X_user_test], y_test, feature_names)

        # Save the trained model
        models_dir = '../../models'
        os.makedirs(models_dir, exist_ok=True)
        model_save_path = os.path.join(models_dir, 'property_recommendation_model.keras')
        save_trained_model(model, model_save_path)


if __name__ == "__main__":
    main()
