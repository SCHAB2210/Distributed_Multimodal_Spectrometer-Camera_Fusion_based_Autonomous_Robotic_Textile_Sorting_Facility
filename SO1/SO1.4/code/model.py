import os
import numpy as np
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import pandas as pd

# Configure paths
FEATURES_PATH = r"C:\Users\devTe\Desktop\TextileSorting\NIR\features\results\train_spectral_features.h5"
TEST_FEATURES_PATH = r"C:\Users\devTe\Desktop\TextileSorting\NIR\features\results\test_spectral_features.h5"
MODEL_DIR = r"C:\Users\devTe\Desktop\TextileSorting\NIR\models\feature_based"
LOG_DIR = os.path.join(MODEL_DIR, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))

# Add this configuration
VERSION = 1  # For versioning the saved files

# Training parameters
BATCH_SIZE = 32
EPOCHS = 1000  # Increased to allow for early stopping
INITIAL_LEARNING_RATE = 0.001
MIN_LEARNING_RATE = 1e-7
DROPOUT_RATE = 0.4
L2_LAMBDA = 0.001

# Add MetricsLogger class right after your imports and configuration
class MetricsLogger(tf.keras.callbacks.Callback):
    """Custom callback to log metrics to separate CSV files after each epoch."""
    def __init__(self, base_filepath):
        super().__init__()
        self.base_filepath = base_filepath
        self.loss_filepath = f"{base_filepath}_loss.csv"
        self.acc_filepath = f"{base_filepath}_accuracy.csv"
        self.metrics = []
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.base_filepath), exist_ok=True)
        
        # Initialize loss CSV with headers
        with open(self.loss_filepath, 'w') as f:
            f.write('epoch,train_loss,val_loss\n')
            
        # Initialize accuracy CSV with headers
        with open(self.acc_filepath, 'w') as f:
            f.write('epoch,train_accuracy,val_accuracy\n')
    
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        
        # Record epoch (1-indexed for readability)
        epoch_num = epoch + 1
        
        # Format loss data and save to loss file
        loss_row = [
            epoch_num,
            logs.get('loss', None),
            logs.get('val_loss', None)
        ]
        
        with open(self.loss_filepath, 'a') as f:
            f.write(','.join(str(value) for value in loss_row) + '\n')
        
        # Format accuracy data and save to accuracy file
        acc_row = [
            epoch_num,
            logs.get('accuracy', None),
            logs.get('val_accuracy', None)
        ]
        
        with open(self.acc_filepath, 'a') as f:
            f.write(','.join(str(value) for value in acc_row) + '\n')
        
        # Store in memory for potential use during training
        combined_row = [epoch_num, 
                        logs.get('loss', None), 
                        logs.get('accuracy', None),
                        logs.get('val_loss', None), 
                        logs.get('val_accuracy', None)]
        self.metrics.append(combined_row)

def load_data():
    """Load features and labels from H5 files."""
    with h5py.File(FEATURES_PATH, 'r') as h5f:
        train_features = h5f['features'][:]
        train_labels = h5f['labels'][:]
    
    with h5py.File(TEST_FEATURES_PATH, 'r') as h5f:
        test_features = h5f['features'][:]
        test_labels = h5f['labels'][:]
    
    # Determine the max class value to set num_classes correctly
    max_class = max(np.max(train_labels), np.max(test_labels))
    num_classes = max_class + 1  # Classes are 0-indexed
    print(f"Detected {num_classes} classes in the dataset")
    
    # Convert labels to categorical
    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes=num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes=num_classes)
    
    return train_features, test_features, train_labels, test_labels

def load_metrics(metric_type='loss', version=VERSION):
    """Load metrics from CSV file for plotting.
    
    Args:
        metric_type: Either 'loss' or 'accuracy'
        version: Model version to load
    
    Returns:
        DataFrame containing the metrics
    """
    csv_path = os.path.join(MODEL_DIR, f'feature_metrics_v{version}_{metric_type}.csv')
    
    if not os.path.exists(csv_path):
        print(f"Warning: Metrics file not found at {csv_path}")
        return None
    
    metrics_df = pd.read_csv(csv_path)
    print(f"Loaded {metric_type} metrics from {csv_path}")
    return metrics_df

def build_model(input_shape):
    """Build enhanced neural network for feature-based classification."""
    # Add L2 regularization
    regularizer = tf.keras.regularizers.l2(L2_LAMBDA)
    
    # Get number of classes from the labels
    with h5py.File(FEATURES_PATH, 'r') as h5f:
        train_labels = h5f['labels'][:]
    num_classes = np.max(train_labels) + 1
    
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=input_shape,
                            kernel_regularizer=regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        
        tf.keras.layers.Dense(256, activation='relu',
                            kernel_regularizer=regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        
        tf.keras.layers.Dense(128, activation='relu',
                            kernel_regularizer=regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE),
        
        tf.keras.layers.Dense(64, activation='relu',
                            kernel_regularizer=regularizer),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(DROPOUT_RATE/2),
        
        tf.keras.layers.Dense(num_classes, activation='softmax')  # Dynamic number of classes
    ])
    
    # Remove learning rate schedule and use fixed learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=INITIAL_LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
    )
    
    return model

def plot_training_history(history, save_path):
    """Plot training and validation loss curves with error handling."""
    plt.figure(figsize=(16, 6))
    
    # Check if history object contains expected keys
    if not isinstance(history, tf.keras.callbacks.History):
        print("Warning: history object is not a Keras History object")
        plt.text(0.5, 0.5, "Invalid history object", 
                 horizontalalignment='center', fontsize=16)
        plt.savefig(save_path)
        plt.close()
        return
    
    # Check if history contains expected keys
    if not all(key in history.history for key in ['loss', 'val_loss', 'accuracy', 'val_accuracy']):
        print("Warning: history object missing expected keys")
        print(f"Available keys: {list(history.history.keys())}")
        
        # Plot whatever data we have
        plt.subplot(1, 2, 1)
        for key in history.history:
            if 'loss' in key.lower():
                plt.plot(history.history[key], label=key)
        plt.title('Loss Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        for key in history.history:
            if 'acc' in key.lower():
                plt.plot(history.history[key], label=key)
        plt.title('Accuracy Metrics')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        plt.close()
        return
    
    # Normal plotting with expected keys
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training history plot saved to {save_path}")
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path, class_names=None):
    """Plot and save confusion matrix with percentage values."""
    # Get true and predicted labels
    y_true_labels = np.argmax(y_true, axis=1)
    y_pred_labels = np.argmax(y_pred, axis=1)
    
    # Get unique classes actually present in the data
    unique_classes = np.unique(np.concatenate([y_true_labels, y_pred_labels]))
    
    # If class names not provided, generate them
    if class_names is None:
        num_classes = len(unique_classes)
        if num_classes == 3:
            class_names = ['White', 'Black', 'Other']
        elif num_classes == 2:
            class_names = ['Cotton', 'Polyester']
        elif num_classes == 6:
            class_names = ['Cotton-White', 'Cotton-Black', 'Cotton-Other',
                          'Poly-White', 'Poly-Black', 'Poly-Other']
        else:
            class_names = [f'Class_{i}' for i in range(num_classes)]
    
    # Calculate confusion matrix with correct labels
    cm = confusion_matrix(y_true_labels, y_pred_labels, 
                         labels=range(len(class_names)))
    
    # Convert to percentages
    cm_percentage = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=True, fmt='.1f', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix (Percentage)')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_from_csv(save_path):
    """Create training history plot from the saved CSV files."""
    # Try to load the CSV files
    try:
        loss_df = pd.read_csv(os.path.join(MODEL_DIR, f'feature_metrics_v{VERSION}_loss.csv'))
        acc_df = pd.read_csv(os.path.join(MODEL_DIR, f'feature_metrics_v{VERSION}_accuracy.csv'))
        
        plt.figure(figsize=(16, 6))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(loss_df['epoch'], loss_df['train_loss'], label='Training Loss')
        plt.plot(loss_df['epoch'], loss_df['val_loss'], label='Validation Loss')
        plt.title('Model Loss (from CSV)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(acc_df['epoch'], acc_df['train_accuracy'], label='Training Accuracy')
        plt.plot(acc_df['epoch'], acc_df['val_accuracy'], label='Validation Accuracy')
        plt.title('Model Accuracy (from CSV)')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path)
        print(f"Training history plot from CSV saved to {save_path}")
        plt.close()
        return True
    except Exception as e:
        print(f"Could not create plot from CSV files: {str(e)}")
        return False

def get_callbacks(model_dir):
    """Define enhanced callbacks for training."""
    metrics_base_path = os.path.join(model_dir, f'feature_metrics_v{VERSION}')
    metrics_logger = MetricsLogger(metrics_base_path)
    
    return [
        # Model checkpoint
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(model_dir, f'best_model_v{VERSION}.h5'),
            monitor='val_precision',
            mode='max',
            save_best_only=True,
            verbose=1
        ),
        
        # Early stopping
        tf.keras.callbacks.EarlyStopping(
            monitor='val_precision',
            mode='max',
            patience=25,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_precision',
            mode='max',
            factor=0.5,
            patience=10,
            min_lr=MIN_LEARNING_RATE,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1,
            write_graph=True,
            update_freq='epoch'
        ),
        
        # Custom metrics logger
        metrics_logger
    ]

def main():
    """Train the model and save metrics and results."""
    print("\n" + "="*80)
    print("FEATURE-BASED MODEL TRAINING PIPELINE")
    print("="*80)
    
    # Create output directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load data
    print("Loading data...")
    train_features, test_features, train_labels, test_labels = load_data()
    
    # Split training data into train and validation
    train_features, val_features, train_labels, val_labels = train_test_split(
        train_features, train_labels, test_size=0.2, random_state=42
    )
    
    # Print dataset statistics
    print(f"\nDataset statistics:")
    print(f"Training set: {train_features.shape[0]} samples")
    print(f"Validation set: {val_features.shape[0]} samples")
    print(f"Test set: {test_features.shape[0]} samples")
    print(f"Feature dimensions: {train_features.shape[1]}")
    
    # Check the actual number of classes in the data
    num_classes = train_labels.shape[1]
    print(f"Number of classes in the data: {num_classes}")
    
    # Determine appropriate class names based on the number of classes
    if num_classes == 3:
        class_names = ['White', 'Black', 'Other']
    elif num_classes == 2:
        class_names = ['Cotton', 'Polyester']
    elif num_classes == 6:
        class_names = ['Cotton-White', 'Cotton-Black', 'Cotton-Other', 
                      'Poly-White', 'Poly-Black', 'Poly-Other']
    else:
        # Generic class names if the number doesn't match known configurations
        class_names = [f'Class_{i}' for i in range(num_classes)]
    
    print(f"Using class names: {class_names}")
    
    # Build and train model
    print("\nBuilding and training model...")
    model = build_model((train_features.shape[1],))
    
    # Calculate and print model size and parameters
    model_parameters = model.count_params()
    # Calculate model size in MB (4 bytes per float32 parameter)
    model_size_bytes = model_parameters * 4
    model_size_mb = model_size_bytes / (1024 * 1024)
    
    print("\nModel Information:")
    print(f"Model Parameters: {model_parameters:,}")
    print(f"Model Size (MB): {model_size_mb:.2f}")
    
    # Print model summary
    model.summary(line_length=100)
    
    # Save model summary to text file
    with open(os.path.join(MODEL_DIR, f'feature_model_summary_v{VERSION}.txt'), 'w') as f:
        # Also save model size information to the summary file
        f.write(f"Model Parameters: {model_parameters:,}\n")
        f.write(f"Model Size (MB): {model_size_mb:.2f}\n\n")
        model.summary(print_fn=lambda x: f.write(x + '\n'),
                     line_length=100)
    
    # Define callbacks
    callbacks = get_callbacks(MODEL_DIR)
    
    try:
        # Train model
        print("\nTraining model...")
        history = model.fit(
            train_features, train_labels,
            validation_data=(val_features, val_labels),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1  # Make sure verbose is set to 1
        )
        
        # Print history keys for debugging
        print("History contains the following metrics:")
        for key in history.history.keys():
            print(f"  - {key}")
        
        # Evaluate and plot regardless of early stopping
        print("\nEvaluating model...")
        test_results = model.evaluate(test_features, test_labels, verbose=1)
        
        # Print metrics based on the actual metrics returned
        metric_names = ['loss', 'accuracy']
        if len(model.metrics) > 2:  # If we have additional metrics
            metric_names.extend(['precision', 'recall'])
        
        for i, name in enumerate(metric_names):
            if i < len(test_results):
                print(f"Test {name}: {test_results[i]:.4f}")
        
        # Generate predictions
        predictions = model.predict(test_features)
        
        # Plot results
        plot_training_history(history, os.path.join(MODEL_DIR, f'training_history_v{VERSION}.png'))
        
        # Get true values and predictions
        y_true = np.argmax(test_labels, axis=1)
        y_pred = np.argmax(predictions, axis=1)
        
        # Determine actual number of unique classes in predictions
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        actual_num_classes = len(unique_classes)
        print(f"Detected {actual_num_classes} unique classes in predictions.")
        
        # Check if number of unique classes matches our expected number
        if actual_num_classes != len(class_names):
            print(f"Warning: Number of detected classes ({actual_num_classes}) doesn't match expected ({len(class_names)})")
            # Adjust class_names to match the actual number of classes
            if actual_num_classes == 3:
                class_names = ['White', 'Black', 'Other']
                print(f"Adjusted class names to: {class_names}")
            elif actual_num_classes == 2:
                class_names = ['Cotton', 'Polyester'] 
                print(f"Adjusted class names to: {class_names}")
            else:
                class_names = [f'Class_{i}' for i in unique_classes]
                print(f"Using generic class names: {class_names}")
        
        # Plot confusion matrix with adjusted class names
        plot_confusion_matrix(test_labels[:, :actual_num_classes], 
                             predictions[:, :actual_num_classes], 
                             os.path.join(MODEL_DIR, f'confusion_matrix_v{VERSION}.png'),
                             class_names)
        
        # Print classification report with correct labels parameter
        print("\nClassification Report:")
        report_text = classification_report(y_true, y_pred, 
                                          labels=range(actual_num_classes),
                                          target_names=class_names)
        print(report_text)
        
        # Get report as dictionary for CSV export
        report = classification_report(y_true, y_pred, 
                                     labels=range(actual_num_classes),
                                     target_names=class_names, 
                                     output_dict=True)
        
        # Save classification report as CSV
        report_df = pd.DataFrame(report).transpose()
        report_csv_path = os.path.join(MODEL_DIR, f'feature_classification_report_v{VERSION}.csv')
        report_df.to_csv(report_csv_path)
        print(f"Classification report saved to: {report_csv_path}")
        
        # Save confusion matrix as CSV with adjusted class names
        cm = confusion_matrix(y_true, y_pred, 
                            labels=range(actual_num_classes),
                            normalize='true') * 100
        cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
        cm_csv_path = os.path.join(MODEL_DIR, f'feature_confusion_matrix_v{VERSION}.csv')
        cm_df.to_csv(cm_csv_path)
        print(f"Confusion matrix saved to: {cm_csv_path}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()
        # Still try to plot if we have history
        if 'history' in locals():
            plot_training_history(history, os.path.join(MODEL_DIR, f'training_history_v{VERSION}.png'))

    # After all other operations in main()
    # Try to create plots from CSV if they exist
    csv_plot_path = os.path.join(MODEL_DIR, f'training_history_from_csv_v{VERSION}.png')
    plot_from_csv(csv_plot_path)

    print("\nTraining complete. Model and results saved.")
    print("="*80)

if __name__ == "__main__":
    main()
# %%
