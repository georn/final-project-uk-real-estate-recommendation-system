from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras.models import save_model
import matplotlib.pyplot as plt
import numpy as np
import logging
import os
from sklearn.metrics import f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class NanTerminateCallback(Callback):
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        loss = logs.get('loss')
        if np.isnan(loss) or np.isinf(loss):
            logging.info('NaN loss encountered, terminating training')
            self.model.stop_training = True

class LoggingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logging.info(f"Epoch {epoch+1}/{self.params['epochs']}")
        logging.info(f"Train loss: {logs['loss']:.4f}, accuracy: {logs['accuracy']:.4f}")
        logging.info(f"Val loss: {logs['val_loss']:.4f}, val_accuracy: {logs['val_accuracy']:.4f}")

def train_model(model, X_train, y_train, X_val, y_val, epochs=200, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    nan_terminate = NanTerminateCallback()
    logging_callback = LoggingCallback()

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping, nan_terminate, logging_callback],
        verbose=0
    )
    return history

def plot_training_history(history):
    if not history.history:
        logging.error("Training history is empty. Cannot plot.")
        return

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.yscale('log')  # Use log scale for loss

    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Log final training metrics
    logging.info(f"Final training loss: {history.history['loss'][-1]:.4f}")
    logging.info(f"Final validation loss: {history.history['val_loss'][-1]:.4f}")
    logging.info(f"Final training accuracy: {history.history['accuracy'][-1]:.4f}")
    logging.info(f"Final validation accuracy: {history.history['val_accuracy'][-1]:.4f}")

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred_binary = (y_pred > 0.5).astype(int)

    accuracy = model.evaluate(X_test, y_test)[1]
    f1 = f1_score(y_test, y_pred_binary)
    auc = roc_auc_score(y_test, y_pred)

    logging.info(f"Test Accuracy: {accuracy:.4f}")
    logging.info(f"F1 Score: {f1:.4f}")
    logging.info(f"AUC: {auc:.4f}")

def save_trained_model(model, save_path):
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # Save the model
    save_model(model, save_path)
    logging.info(f"Model saved to {save_path}")
