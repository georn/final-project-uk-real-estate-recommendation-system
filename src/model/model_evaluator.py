from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

def evaluate_model(model, X_val, y_val, scaler_target):
    # Predict on the validation set
    predictions = model.predict(X_val)

    # Rescale the predictions and true values back to the original scale
    y_val_rescaled = scaler_target.inverse_transform(y_val.reshape(-1, 1)).flatten()
    predictions_rescaled = scaler_target.inverse_transform(predictions).flatten()

    # Calculate evaluation metrics
    mae = mean_absolute_error(y_val_rescaled, predictions_rescaled)
    rmse = np.sqrt(mean_squared_error(y_val_rescaled, predictions_rescaled))

    return mae, rmse