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
MIN_SAMPLE_THRESHOLD = 300  # Ensure dataset is sufficiently large
VERSION = 1  # For versioning the saved files

# Define result directories
ROOT_RESULTS_DIR = r'C:\Users\devTe\Desktop\TextileSorting\NIR\results'
BEST_MODELS_DIR = os.path.join(ROOT_RESULTS_DIR, 'best_models')
SINGLE_OUTPUT_RESULTS_DIR = os.path.join(ROOT_RESULTS_DIR, 'single_output_model_results')

# Path to dataset
DATASET_PATH = r'C:\Users\devTe\Desktop\TextileSorting\NIR\samples\combined_fixed\data_color_balanced3_fixed.csv'

# Metrics Logger Class
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

# ----- Load and Preprocess Data -----
def load_and_preprocess_color_data():
    """Load and preprocess color dataset."""
    color_data = pd.read_csv(DATASET_PATH)
    
    print(f"Total dataset size: {color_data.shape[0]} samples")
    print(f"Dataset features: {color_data.shape[1] - 1}")
    
    # Print class distribution in original dataset
    class_counts = color_data.iloc[:, -1].astype(int).value_counts().sort_index()
    print("\nOriginal class distribution:")
    for class_idx, count in class_counts.items():
        class_name = ["White", "Black", "Other"][class_idx]
        print(f"  Class {class_idx} ({class_name}): {count} samples ({count/color_data.shape[0]:.2%})")

    # Extract features (all columns except last) and labels (last column)
    X = color_data.iloc[:, :-1].values  
    y = color_data.iloc[:, -1].astype(int).values  # Class labels

    # Normalize features using StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)  # Normalize absorbance values

    # Split dataset: 70% Train, 15% Validation, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # Print split dataset sizes
    print("\nSplit dataset sizes:")
    print(f"  Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/color_data.shape[0]:.2%})")
    print(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/color_data.shape[0]:.2%})")
    print(f"  Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/color_data.shape[0]:.2%})")
    
    # Print class distribution in training set
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    print("\nTraining set class distribution:")
    for class_idx, count in zip(unique_train, counts_train):
        class_name = ["White", "Black", "Other"][class_idx]
        print(f"  Class {class_idx} ({class_name}): {count} samples ({count/len(y_train):.2%})")
    
    # Print class distribution in validation set
    unique_val, counts_val = np.unique(y_val, return_counts=True)
    print("\nValidation set class distribution:")
    for class_idx, count in zip(unique_val, counts_val):
        class_name = ["White", "Black", "Other"][class_idx]
        print(f"  Class {class_idx} ({class_name}): {count} samples ({count/len(y_val):.2%})")
    
    # Print class distribution in test set
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    print("\nTest set class distribution:")
    for class_idx, count in zip(unique_test, counts_test):
        class_name = ["White", "Black", "Other"][class_idx]
        print(f"  Class {class_idx} ({class_name}): {count} samples ({count/len(y_test):.2%})")

    # One-hot encode labels (assuming 3 classes: White, Black, Other)
    y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
    y_val = tf.keras.utils.to_categorical(y_val, num_classes=3)
    y_test = tf.keras.utils.to_categorical(y_test, num_classes=3)

    return X_train, X_val, X_test, y_train, y_val, y_test
# ----- Create Optimized CNN Model -----
def create_color_model(input_shape):
    """Optimized CNN model for color classification."""
    inputs = tf.keras.layers.Input(shape=input_shape)

    # Convolutional Block 1
    x = tf.keras.layers.Conv1D(16, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Convolutional Block 2
    x = tf.keras.layers.Conv1D(32, kernel_size=3, activation='relu', padding='same', kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Dropout(0.3)(x)

    # Flatten and Dense Layers
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.001))(x)
    x = tf.keras.layers.Dropout(0.5)(x)

    outputs = tf.keras.layers.Dense(3, activation='softmax')(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=outputs)

    # Use Adam optimizer with **lower learning rate**
    optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, clipnorm=1.0)

    # Update compilation with additional metrics
    model.compile(
        optimizer=optimizer,
        loss='categorical_crossentropy',
        metrics=['accuracy',
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall'),
                tf.keras.metrics.AUC(name='auc')]
    )

    return model


# ----- Plot Training History & Save -----
def plot_training_history(history, save_path=None):
    """Plot training and validation loss curves."""
    plt.figure(figsize=(8, 6))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')

    plt.title('Color Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Training history plot saved at {save_path}")
    else:
        plt.show()


# ----- Plot Confusion Matrix & Save -----
def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Plot confusion matrix for test predictions."""
    cm = confusion_matrix(np.argmax(y_true, axis=1), y_pred, normalize='true') * 100
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues')
    plt.title('Color Confusion Matrix (Percentage)')
    plt.xlabel('Predicted')
    plt.ylabel('True')

    if save_path:
        plt.savefig(save_path)
        print(f"Confusion matrix plot saved at {save_path}")
    else:
        plt.show()


# ----- Save Trained Model -----
def save_model(model, save_dir):
    """Save trained color model to the given directory."""
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'color_model.h5')
    model.save(model_path)
    print(f"Color model saved at {model_path}")


# ----- Train and Evaluate Model -----
def train_model_pipeline():
    """Load data, train the model, and evaluate performance."""
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_preprocess_color_data()

    # Create model (input shape is based on dataset)
    model = create_color_model(input_shape=(X_train.shape[1], 1))

    # Prepare TensorFlow datasets
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train.reshape(-1, X_train.shape[1], 1), y_train)).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val.reshape(-1, X_val.shape[1], 1), y_val)).batch(BATCH_SIZE).cache().prefetch(tf.data.AUTOTUNE)

    # Setup callbacks for training
    # Define metrics logger with base path
    metrics_base_path = os.path.join(SINGLE_OUTPUT_RESULTS_DIR, f'color_metrics_v{VERSION}')
    metrics_logger = MetricsLogger(metrics_base_path)
    
    checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
        filepath=os.path.join(BEST_MODELS_DIR, 'color_model_best.h5'), 
        save_best_only=True, 
        monitor='val_loss',
        verbose=1
    )
    earlystop_cb = tf.keras.callbacks.EarlyStopping(
        patience=20, 
        restore_best_weights=True,
        verbose=1
    )
    reduceLR_cb = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-6,
        verbose=1
    )

    # Train the model with metrics logging
    history = model.fit(
        train_dataset, 
        validation_data=val_dataset, 
        epochs=500, 
        callbacks=[earlystop_cb, checkpoint_cb, reduceLR_cb, metrics_logger]
    )

    return model, history, X_test, y_test


# ----- Main Function -----
def main():
    """Train the model and generate evaluation reports."""
    print("\n" + "="*80)
    print("COLOR MODEL TRAINING PIPELINE")
    print("="*80)
    
    model, history, X_test, y_test = train_model_pipeline()

    # Print model architecture
    print("\nModel Architecture Summary:")
    model.summary(line_length=120, show_trainable=True, expand_nested=True)
    
    # Save model summary to text file
    with open(os.path.join(SINGLE_OUTPUT_RESULTS_DIR, f'color_model_summary_v{VERSION}.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'),
                     line_length=120,
                     show_trainable=True)

    # Evaluate and print detailed metrics
    print("\nModel Performance Metrics:")
    test_results = model.evaluate(
        X_test.reshape(-1, X_test.shape[1], 1), 
        y_test, 
        verbose=1
    )
    
    # Print metrics
    metric_names = ['loss', 'accuracy', 'precision', 'recall', 'auc']
    for name, value in zip(metric_names, test_results):
        print(f"{name.capitalize()}: {value:.4f}")

    # Generate predictions and classification report
    predictions = model.predict(X_test.reshape(-1, X_test.shape[1], 1))
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\nDetailed Classification Report:")
    class_names = ['White', 'Black', 'Other']
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True
    )
    
    # Convert classification report to DataFrame and save as CSV
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(SINGLE_OUTPUT_RESULTS_DIR, f'color_classification_report_v{VERSION}.csv')
    report_df.to_csv(report_csv_path)
    print(f"Classification report saved to: {report_csv_path}")
    
    # Save confusion matrix as CSV
    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = os.path.join(SINGLE_OUTPUT_RESULTS_DIR, f'color_confusion_matrix_v{VERSION}.csv')
    cm_df.to_csv(cm_csv_path)
    print(f"Confusion matrix saved to: {cm_csv_path}")
    
    # Keep the existing plots if needed
    training_plot_path = os.path.join(SINGLE_OUTPUT_RESULTS_DIR, f'color_training_history_v{VERSION}.png')
    confusion_plot_path = os.path.join(SINGLE_OUTPUT_RESULTS_DIR, f'color_confusion_matrix_v{VERSION}.png')
    
    plot_training_history(history, save_path=training_plot_path)
    plot_confusion_matrix(y_test, y_pred, save_path=confusion_plot_path)
    save_model(model, SINGLE_OUTPUT_RESULTS_DIR)
    
    print("\nTraining complete. Model and results saved.")
    print("="*80)


if __name__ == "__main__":
    main()
