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


def plot_feature_importance(model, X_test, y_test, feature_names, top_n=15):
    X_property_test, X_user_test = X_test
    n_features = X_property_test.shape[1] + X_user_test.shape[1]

    # Get the base score
    base_score = model.evaluate(X_test, y_test, verbose=0)[1]

    # Calculate importance for each feature
    importances = []
    for i in range(n_features):
        if i < X_property_test.shape[1]:
            X_property_temp = X_property_test.copy()
            X_property_temp[:, i] = np.random.permutation(X_property_temp[:, i])
            X_temp = [X_property_temp, X_user_test]
        else:
            X_user_temp = X_user_test.copy()
            X_user_temp[:, i - X_property_test.shape[1]] = np.random.permutation(X_user_temp[:, i - X_property_test.shape[1]])
            X_temp = [X_property_test, X_user_temp]

        score = model.evaluate(X_temp, y_test, verbose=0)[1]
        importances.append(base_score - score)

    # Sort features by importance
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # Plot top N features
    plt.figure(figsize=(12, 10))  # Increase figure size
    y_pos = range(top_n)
    feature_names = [x[0] for x in feature_importance[:top_n]]
    feature_scores = [x[1] for x in feature_importance[:top_n]]

    plt.barh(y_pos, feature_scores, align='center')
    plt.yticks(y_pos, feature_names)
    plt.xlabel("Feature Importance (Accuracy Drop)")
    plt.title(f"Top {top_n} Most Important Features")
    plt.tight_layout()
    plt.gca().invert_yaxis()  # Invert y-axis to show most important at the top
    plt.show()

    # Log all feature importances
    logging.info("Feature Importance Ranking:")
    for i, (feature, importance) in enumerate(feature_importance):
        logging.info(f"{i+1}. {feature}: {importance:.4f}")

    return feature_importance
