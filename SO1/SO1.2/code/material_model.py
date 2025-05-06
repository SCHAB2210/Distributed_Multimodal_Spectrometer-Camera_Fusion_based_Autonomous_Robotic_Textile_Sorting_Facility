import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import os
from tensorflow.keras import regularizers
from datetime import datetime

# Configuration
BATCH_SIZE = 16
MIN_SAMPLE_THRESHOLD = 6000  # Change as needed
VERSION = 1  # For versioning the saved files

# Define your results directories
ROOT_RESULTS_DIR = r'C:\Users\devTe\Desktop\TextileSorting\NIR\results'
BEST_MODELS_DIR = os.path.join(ROOT_RESULTS_DIR, 'best_models')
SINGLE_OUTPUT_RESULTS_DIR = os.path.join(ROOT_RESULTS_DIR, 'single_output_model_results')

# Path to dataset
DATASET_PATH = r'C:\Users\devTe\Desktop\TextileSorting\NIR\samples\combined_fixed\data_material_balanced3_fixed.csv'

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
        
        # Debug - print available keys
        if epoch == 0:
            print(f"Debug: Available logs keys: {list(logs.keys())}")
        
        # Record epoch (1-indexed for readability)
        epoch_num = epoch + 1
        
        # Access metrics with correct keys from logs
        # In TF/Keras, validation metrics have 'val_' prefix
        train_loss = logs.get('loss', None)
        val_loss = logs.get('val_loss', None) 
        train_acc = logs.get('accuracy', None)
        val_acc = logs.get('val_accuracy', None)
        
        # Format loss data and save to loss file
        loss_row = [epoch_num, train_loss, val_loss]
        
        with open(self.loss_filepath, 'a') as f:
            f.write(','.join([str(val) if val is not None else "NA" for val in loss_row]) + '\n')
        
        # Format accuracy data and save to accuracy file
        acc_row = [epoch_num, train_acc, val_acc]
        
        with open(self.acc_filepath, 'a') as f:
            f.write(','.join([str(val) if val is not None else "NA" for val in acc_row]) + '\n')
        
        # Store in memory for potential use during training
        self.metrics.append([epoch_num, train_loss, train_acc, val_loss, val_acc])

def load_and_preprocess_material_data():
    """Load, preprocess, and split the material dataset, ensuring exactly **122** input features."""
    # Load dataset
    material_data = pd.read_csv(DATASET_PATH)

    # Extract **exactly 122** features
    X = material_data.iloc[:, :-1].values  # Features (wavelength data)
    y = material_data.iloc[:, -1].astype(int).values  # Labels

    # Print dataset information
    num_samples = X.shape[0]
    print(f"Number of material samples loaded: {num_samples}")
    print(f"Feature shape: {X.shape} (should be [samples, 122]), Label shape: {y.shape}")

    if num_samples < MIN_SAMPLE_THRESHOLD:
        print(f"Warning: Sample size is below {MIN_SAMPLE_THRESHOLD}. Consider augmentation.")

    # Print class distribution in original dataset
    unique, counts = np.unique(y, return_counts=True)
    print("\nOriginal class distribution:")
    for class_idx, count in zip(unique, counts):
        class_name = ["Cotton", "Polyester"][class_idx]
        print(f"  Class {class_idx} ({class_name}): {count} samples ({count/len(y):.2%})")

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split dataset: 70% Train, 15% Validation, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Print split dataset sizes
    print("\nSplit dataset sizes:")
    print(f"  Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/X.shape[0]:.2%})")
    print(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/X.shape[0]:.2%})")
    print(f"  Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/X.shape[0]:.2%})")
    
    # Print class distribution in training set
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print("\nTraining set class distribution:")
    for class_idx, count in zip(unique_train, counts_train):
        class_name = ["Cotton", "Polyester"][class_idx]
        print(f"  Class {class_idx} ({class_name}): {count} samples ({count/len(y_train):.2%})")
    
    # Print class distribution in validation set
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    print("\nValidation set class distribution:")
    for class_idx, count in zip(unique_val, counts_val):
        class_name = ["Cotton", "Polyester"][class_idx]
        print(f"  Class {class_idx} ({class_name}): {count} samples ({count/len(y_val):.2%})")
    
    # Print class distribution in test set
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print("\nTest set class distribution:")
    for class_idx, count in zip(unique_test, counts_test):
        class_name = ["Cotton", "Polyester"][class_idx]
        print(f"  Class {class_idx} ({class_name}): {count} samples ({count/len(y_test):.2%})")

    # One-hot encode labels (2 material classes: Cotton & Polyester)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=2)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=2)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=2)

    print(f"Train samples: {X_train.shape[0]}, Validation: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    print(f"Input shape per sample: {X_train.shape[1]} points (should be 122)")

    return X_train, X_val, X_test, y_train, y_val, y_test

def create_material_model(input_shape):
    """CNN Model for Material Classification (Input: 122,1)"""
    inputs = tf.keras.layers.Input(shape=input_shape)
    # First Convolutional Block
    x = tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu', padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    # Second Convolutional Block
    x = tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    # Third Convolutional Block
    x = tf.keras.layers.Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    # Fully Connected Layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(2, activation='softmax')(x)  # Output for 2 classes
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=5e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
    )

    return model

def plot_training_history(history, save_path=None):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8, 6))

    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title('Material Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved at {save_path}")
    else:
        plt.show()

def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix for test predictions."""
    cm = confusion_matrix(np.argmax(y_true, axis=1), y_pred, normalize='true') * 100
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues')
    plt.title('Material Confusion Matrix (Percentage)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved at {save_path}")
    else:
        plt.show()

def save_model(model, save_dir):
    """Save trained material model to the given directory."""
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'material_model.h5')
    model.save(model_path)
    print(f"Material model saved at {model_path}")

def load_metrics(metric_type='loss', version=VERSION):
    """Load metrics from CSV file for plotting.
    
    Args:
        metric_type: Either 'loss' or 'accuracy'
        version: Model version to load
    
    Returns:
        DataFrame containing the metrics
    """
    csv_path = os.path.join(SINGLE_OUTPUT_RESULTS_DIR, 
                           f'material_metrics_v{version}_{metric_type}.csv')
    
    if not os.path.exists(csv_path):
        print(f"Warning: Metrics file not found at {csv_path}")
        return None
    
    metrics_df = pd.read_csv(csv_path)
    print(f"Loaded {metric_type} metrics from {csv_path}")
    return metrics_df

def main():
    """Train the model and generate evaluation reports."""
    # Print separator for better readability
    print("\n" + "="*80)
    print("MATERIAL MODEL TRAINING PIPELINE")
    print("="*80)
    
    # Load and preprocess data - this will show all the dataset statistics
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_material_data()
    
    # Create model
    model = create_material_model(input_shape=(X_train.shape[1], 1))
    
    # Print model architecture
    print("\nModel Architecture Summary:")
    model.summary(line_length=120, show_trainable=True)
    
    # Prepare datasets
    train_dataset = tf.data.Dataset.from_tensor_slices(
        (X_train.reshape(-1, 122, 1), y_train)).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        (X_val.reshape(-1, 122, 1), y_val)).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    
    # Define metrics logger with base path
    os.makedirs(SINGLE_OUTPUT_RESULTS_DIR, exist_ok=True)  # Ensure directory exists
    metrics_base_path = os.path.join(SINGLE_OUTPUT_RESULTS_DIR, f'material_metrics_v{VERSION}')
    metrics_logger = MetricsLogger(metrics_base_path)
    
    print(f"Metrics will be logged to: {metrics_base_path}_loss.csv and {metrics_base_path}_accuracy.csv")
    
    # Setup callbacks
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(BEST_MODELS_DIR, f'material_model_best_v{VERSION}.h5'), 
        save_best_only=True, 
        monitor='val_loss',
        verbose=1)
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        patience=5, 
        restore_best_weights=True,
        verbose=1)
    reduceLR_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1
    )
    
    callbacks = [
        checkpoint_cb,
        earlystop_cb,
        reduceLR_cb,
        metrics_logger  # Make sure this is included in the list
    ]
    
    # Save model summary to text file
    with open(os.path.join(SINGLE_OUTPUT_RESULTS_DIR, f'material_model_summary_v{VERSION}.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'),
                     line_length=120,
                     show_trainable=True)
    
    # Train model
    print("\nStarting model training...")
    history = model.fit(
        train_dataset, 
        validation_data=val_dataset, 
        epochs=50, 
        callbacks=callbacks)  # Pass the list of callbacks here
    
    # Evaluate and print detailed metrics
    print("\nModel Performance Metrics:")
    test_results = model.evaluate(X_test.reshape(-1, 122, 1), y_test, verbose=1)
    
    # Print metrics
    metric_names = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    for name, value in zip(metric_names, test_results):
        print(f"{name.capitalize()}: {value:.4f}")
    
    # Calculate and print classification report
    predictions = model.predict(X_test.reshape(-1, 122, 1))
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\nDetailed Classification Report:")
    class_names = ['Cotton', 'Polyester']
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True
    )
    
    # Convert classification report to DataFrame and save as CSV
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(SINGLE_OUTPUT_RESULTS_DIR, f'material_classification_report_v{VERSION}.csv')
    report_df.to_csv(report_csv_path)
    print(f"Classification report saved to: {report_csv_path}")
    
    # Save confusion matrix as CSV
    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = os.path.join(SINGLE_OUTPUT_RESULTS_DIR, f'material_confusion_matrix_v{VERSION}.csv')
    cm_df.to_csv(cm_csv_path)
    print(f"Confusion matrix saved to: {cm_csv_path}")
    
    # Save plots
    training_plot_path = os.path.join(SINGLE_OUTPUT_RESULTS_DIR, f'material_training_history_v{VERSION}.png')
    confusion_plot_path = os.path.join(SINGLE_OUTPUT_RESULTS_DIR, f'material_confusion_matrix_v{VERSION}.png')
    
    plot_training_history(history, save_path=training_plot_path)
    plot_confusion_matrix(y_test, y_pred, save_path=confusion_plot_path)
    
    # Save model
    model_save_path = os.path.join(SINGLE_OUTPUT_RESULTS_DIR, f'material_model_v{VERSION}.h5')
    model.save(model_save_path)
    print(f"Material model saved at {model_save_path}")
    
    print("\nTraining complete. Model and results saved.")
    print("="*80)

if __name__ == "__main__":
    main()
