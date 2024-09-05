import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report


def evaluate_model(model, X_test, y_test):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred_binary)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title('Confusion Matrix')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

    # Classification Report
    report = classification_report(y_test, y_pred_binary)
    logging.info("Classification Report:\n" + report)

    # Class Distribution
    class_distribution = pd.Series(y_test).value_counts(normalize=True)
    logging.info("Class distribution in test set:\n" + str(class_distribution))

    return accuracy, precision, recall, f1


def analyze_misclassifications(model, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Convert y_test to numpy array if it's a pandas Series
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    misclassified = np.where(y_test != y_pred_binary.squeeze())[0]

    logging.info(f"Number of misclassified samples: {len(misclassified)}")

    if len(misclassified) > 0:
        for i, idx in enumerate(misclassified[:5]):  # Analyze the first 5 misclassifications
            logging.info(f"\nMisclassified sample {i + 1}:")
            logging.info(f"True label: {y_test[idx]}")
            logging.info(f"Predicted probability: {y_pred[idx][0]:.4f}")
            logging.info("Feature values:")
            for name, value in zip(feature_names, np.concatenate((X_test[0][idx], X_test[1][idx]))):
                logging.info(f"{name}: {value:.4f}")


def plot_feature_importance(model, feature_names):
    logging.warning("Feature importance is not directly available for this type of model.")
    logging.info(
        "Consider using techniques like permutation importance or SHAP values for feature importance in neural networks.")
