import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, \
    classification_report


def evaluate_model(model, X_test, y_test, plot_cm=False, log_report=False, log_distribution=False):
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred_binary)
    precision = precision_score(y_test, y_pred_binary)
    recall = recall_score(y_test, y_pred_binary)
    f1 = f1_score(y_test, y_pred_binary)

    if plot_cm:
        cm = confusion_matrix(y_test, y_pred_binary)
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d')
        plt.title('Confusion Matrix')
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.show()

    if log_report:
        report = classification_report(y_test, y_pred_binary)
        logging.info("Classification Report:\n" + report)

    if log_distribution:
        class_distribution = pd.Series(y_test).value_counts(normalize=True)
        logging.info("Class distribution in test set:\n" + str(class_distribution))

    return accuracy, precision, recall, f1


def analyze_misclassifications(model, X_test, y_test, feature_names):
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    misclassified = np.where(y_test != y_pred_binary.squeeze())[0]
    total_samples = len(y_test)
    misclassified_count = len(misclassified)

    analysis_results = [
        f"Total samples: {total_samples}",
        f"Number of misclassified samples: {misclassified_count}",
        f"Misclassification rate: {misclassified_count/total_samples:.2%}"
    ]

    if misclassified_count > 0:
        # Analyze the top 3 most confidently misclassified samples
        confidences = np.abs(y_pred.squeeze()[misclassified] - 0.5)
        top_3_indices = misclassified[np.argsort(confidences)[-3:]]

        analysis_results.append("\nTop 3 most confidently misclassified samples:")
        for i, idx in enumerate(top_3_indices):
            true_label = y_test[idx]
            pred_prob = y_pred[idx].item()  # Convert to scalar
            analysis_results.extend([
                f"\nSample {i + 1}:",
                f"True label: {'Suitable' if true_label else 'Not Suitable'}",
                f"Predicted probability: {pred_prob:.4f}",
                "Key feature values:"
            ])
            # Include only the top 5 most important features for this sample
            feature_values = np.concatenate((X_test[0][idx], X_test[1][idx]))
            sorted_features = sorted(zip(feature_names, feature_values), key=lambda x: abs(x[1]), reverse=True)[:5]
            for name, value in sorted_features:
                analysis_results.append(f"  {name}: {value:.4f}")

    return "\n".join(analysis_results)


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
            X_temp = [X_property_test, X_user_test]

        score = model.evaluate(X_temp, y_test, verbose=0)[1]
        importances.append(base_score - score)

    # Sort features by importance
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)

    # Plot top N features
    plt.figure(figsize=(12, 10))  # Increase figure size
    y_pos = range(top_n)
    top_features = feature_importance[:top_n]
    feature_names = [x[0] for x in top_features]
    feature_scores = [x[1] for x in top_features]

    plt.barh(y_pos, feature_scores, align='center')
    plt.yticks(y_pos, feature_names)
    plt.xlabel("Feature Importance (Accuracy Drop)")
    plt.title(f"Top {top_n} Most Important Features")
    plt.tight_layout()
    plt.gca().invert_yaxis()  # Invert y-axis to show most important at the top
    plt.show()

    # Return only the top N features and their importance scores
    return top_features
