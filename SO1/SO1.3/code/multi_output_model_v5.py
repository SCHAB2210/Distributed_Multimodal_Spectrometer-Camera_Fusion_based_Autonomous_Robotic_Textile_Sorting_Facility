import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, models, regularizers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix  
from sklearn.metrics import classification_report, f1_score  
from datetime import datetime  
VERSION = 20
# =============================================================================
# PATH CONFIGURATION
# =============================================================================
def get_next_version():
    """Get next available version number by checking existing files."""
    base_dir = r'C:\Users\devTe\Desktop\TextileSorting\NIR\results\multi_output_model_combined_v5'
    version = 20
    # Check for existing model files and increment version if necessary
    while os.path.exists(os.path.join(base_dir, f'multi_output_model_v5_{version}.h5')):
        version += 1
    return version

#VERSION = get_next_version()
COMBINED_CSV_PATH = r'C:\Users\devTe\Desktop\TextileSorting\NIR\samples\combined_fixed\data_combined_balanced3_fixed.csv'
RESULTS_DIR = r'C:\Users\devTe\Desktop\TextileSorting\NIR\results\multi_output_model_combined_v5'
MODEL_SAVE_PATH = os.path.join(RESULTS_DIR, f'multi_output_model_v5_{VERSION}.h5')
LOG_DIR = os.path.join(RESULTS_DIR, 'logs', f'v{VERSION}_{datetime.now().strftime("%Y%m%d-%H%M%S")}')

# Training parameters
BATCH_SIZE = 64  # Increased from 16
EPOCHS = 300    # Increased from 100
LEARNING_RATE = 1e-4  # Reduced learning rate
NOISE_FACTOR = 0.001   # Reduced noise
DROPOUT_RATE = 0.4    # Reduced dropout
L2_REG = 0.0005      # Reduced L2 regularization

# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================
def load_combined_data():
    """Load combined dataset with wavelengths 900-1700nm."""
    df = pd.read_csv(COMBINED_CSV_PATH)
    X = df.iloc[:, :-1].values  # All features except last column
    y = df.iloc[:, -1].astype(int).values  # Labels 0-5
    print("Combined data loaded:", X.shape, "Labels:", y.shape)
    return X, y

def load_training_metrics(version=VERSION):
    """Load training metrics from CSV file for plotting."""
    csv_path = os.path.join(RESULTS_DIR, f'training_metrics_v{version}.csv')
    
    if not os.path.exists(csv_path):
        print(f"Warning: Metrics file not found at {csv_path}")
        return None
    
    metrics_df = pd.read_csv(csv_path)
    print(f"Loaded training metrics from {csv_path}")
    return metrics_df

# =============================================================================
# PREPROCESSING FUNCTION
# =============================================================================
def preprocess_data():
    """Loads data and splits into train/val/test sets for a single categorical output (0-5)."""
    df = pd.read_csv(COMBINED_CSV_PATH)
    
    # Features and labels
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].astype(int).values  # Labels are now 0-5

    # Split data
    indices = np.arange(len(X))
    train_idx, temp_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=y)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42, stratify=y[temp_idx])

    # Split datasets
    X_train, X_val, X_test = X[train_idx], X[val_idx], X[test_idx]
    
    # One-hot encode labels (6 classes: 0-5)
    y_train = tf.keras.utils.to_categorical(y[train_idx], num_classes=6)
    y_val = tf.keras.utils.to_categorical(y[val_idx], num_classes=6)
    y_test = tf.keras.utils.to_categorical(y[test_idx], num_classes=6)

    return X_train, X_val, X_test, y_train, y_val, y_test

# =============================================================================
# MODEL DEFINITION: SINGLE-OUTPUT MULTICLASS CLASSIFICATION [class:0-5]
# =============================================================================
def residual_block(x, filters, kernel_size=3, l2_reg=L2_REG):
    """Enhanced residual block with more regularization"""
    shortcut = x
    
    # First convolution path
    x = tf.keras.layers.Conv1D(
        filters, kernel_size, 
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE/2)(x)
    
    # Second convolution path
    x = tf.keras.layers.Conv1D(
        filters, kernel_size, 
        padding='same',
        kernel_regularizer=regularizers.l2(l2_reg)
    )(x)
    x = tf.keras.layers.BatchNormalization()(x)
    
    # Adjust shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = tf.keras.layers.Conv1D(
            filters, 1, 
            padding='same',
            kernel_regularizer=regularizers.l2(l2_reg)
        )(shortcut)
    
    # Add and activate
    x = tf.keras.layers.Add()([x, shortcut])
    x = tf.keras.layers.Activation('relu')(x)
    return x

def build_single_output_model(input_shape):
    """Build model for predicting a single label (0-5) instead of two separate outputs."""
    input_layer = tf.keras.layers.Input(shape=input_shape)

    # Split the input spectrum into color and material regions
    color_input = tf.keras.layers.Lambda(lambda x: x[:, :105, :])(input_layer)
    material_input = tf.keras.layers.Lambda(lambda x: x[:, 105:, :])(input_layer)

    # Color branch
    color_x = tf.keras.layers.Conv1D(128, kernel_size=5, padding='same')(color_input)
    color_x = tf.keras.layers.BatchNormalization()(color_x)
    color_x = tf.keras.layers.Activation('relu')(color_x)
    color_x = residual_block(color_x, 128)
    color_x = tf.keras.layers.MaxPooling1D(2)(color_x)
    color_x = residual_block(color_x, 256)
    color_x = tf.keras.layers.MaxPooling1D(2)(color_x)
    color_x = residual_block(color_x, 512)
    
    # Material branch
    material_x = tf.keras.layers.Conv1D(64, kernel_size=5, padding='same')(material_input)
    material_x = tf.keras.layers.BatchNormalization()(material_x)
    material_x = tf.keras.layers.Activation('relu')(material_x)
    material_x = residual_block(material_x, 64)
    material_x = tf.keras.layers.MaxPooling1D(2)(material_x)
    material_x = residual_block(material_x, 128)

    # Apply Global Average Pooling to both branches before merging
    color_x = tf.keras.layers.GlobalAveragePooling1D()(color_x)
    material_x = tf.keras.layers.GlobalAveragePooling1D()(material_x)

    # Merge both branches (shapes now match)
    merged_x = tf.keras.layers.Concatenate()([color_x, material_x])

    # Fully Connected Layers
    merged_x = tf.keras.layers.Dense(256, activation='relu')(merged_x)
    merged_x = tf.keras.layers.Dropout(DROPOUT_RATE)(merged_x)

    # Final Output Layer (6 classes)
    final_output = tf.keras.layers.Dense(6, activation='softmax', name='final_output')(merged_x)

    # Create model
    model = tf.keras.models.Model(inputs=input_layer, outputs=final_output)

    # Compile with single loss function
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0, clipvalue=0.5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

# =============================================================================
# PLOTTING FUNCTIONS
# =============================================================================
def smooth_curve(points, factor=0.9):
    """Exponential moving average smoothing."""
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return np.array(smoothed_points)


def plot_training_history(history, save_path=None, smoothing=True):
    """Plot training and validation loss & accuracy with optional smoothing."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

    train_loss = np.array(history.history['loss'])  
    val_loss = np.array(history.history['val_loss'])  
    train_acc = np.array(history.history['accuracy'])  
    val_acc = np.array(history.history['val_accuracy'])  

    if smoothing:
        train_loss = smooth_curve(train_loss)
        val_loss = smooth_curve(val_loss)
        train_acc = smooth_curve(train_acc)
        val_acc = smooth_curve(val_acc)

    ax1.plot(train_loss, label='Training Loss')
    ax1.plot(val_loss, label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_acc, label='Training Accuracy')
    ax2.plot(val_acc, label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        print("Training history plot saved at", save_path)
    plt.show()



def plot_confusion_matrix(y_true, y_pred, class_names, title, save_path=None):
    """Plot a normalized confusion matrix with improved readability."""
    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=class_names, 
                yticklabels=class_names)
    
    plt.title(title, pad=20)
    plt.xlabel('Predicted', labelpad=10)
    plt.ylabel('True', labelpad=10)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', pad_inches=0.5)
        print(f"Confusion matrix plot saved at {save_path}")
    else:
        plt.show()


def plot_combined_confusion_matrix(y_color_true, y_material_true, y_color_pred, y_material_pred, save_path=None):
    """
    Plot a combined confusion matrix showing all color-material combinations.
    Labels are formatted as: 'Material-Color'
    """
    y_true_combined = y_material_true * 3 + y_color_true
    y_pred_combined = y_material_pred * 3 + y_color_pred
    
    material_names = ["Cotton", "Polyester"]
    color_names = ["White", "Black", "Other"]
    class_names = [f"{m}-{c}" for m in material_names for c in color_names]
    
    cm = confusion_matrix(y_true_combined, y_pred_combined, normalize='true') * 100
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt=".2f", cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Combined Color-Material Confusion Matrix (%)")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
        print("Combined confusion matrix saved at", save_path)
    plt.show()

# =============================================================================
# CALLBACKS & TRAINING PIPELINE
# =============================================================================
class ValidationGapCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        train_loss = logs.get('loss')
        val_loss = logs.get('val_loss')
        gap = val_loss - train_loss
        
        if gap < 0:
            print(f'\nWarning: Epoch {epoch}: Validation loss ({val_loss:.4f}) is lower than '
                  f'training loss ({train_loss:.4f}). Gap: {gap:.4f}')

def augment_spectral_data(x, noise_factor=NOISE_FACTOR):
    """Data augmentation with only Gaussian noise"""
    x = tf.cast(x, tf.float32)
    noise = tf.random.normal(shape=tf.shape(x), mean=0.0, stddev=noise_factor, dtype=tf.float32)
    x = x + noise
    return x

# ---------------------------
# Warmup Learning Rate Callback
# ---------------------------
class WarmUpLearningRateScheduler(tf.keras.callbacks.Callback):
    """
    Warm-up learning rate scheduler: linearly increases the learning rate 
    from 0 to the specified initial_lr over warmup_epochs.
    """
    def __init__(self, initial_lr, warmup_epochs, verbose=0):
        super(WarmUpLearningRateScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            warmup_lr = self.initial_lr * (epoch + 1) / self.warmup_epochs
            tf.keras.backend.set_value(self.model.optimizer.lr, warmup_lr)
            if self.verbose > 0:
                print(f"\nEpoch {epoch+1}: Warm-up LR set to {warmup_lr:.6f}")

# Add this class after your other callbacks
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

def get_callbacks():
    """Enhanced callbacks with separate metrics logging"""
    
    # Define metrics logger with base path
    metrics_base_path = os.path.join(RESULTS_DIR, f'metrics_v{VERSION}')
    metrics_logger = MetricsLogger(metrics_base_path)
    
    return [
        # Save best model during training
        tf.keras.callbacks.ModelCheckpoint(
            filepath=MODEL_SAVE_PATH, 
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            patience=20, 
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when plateauing
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=LOG_DIR,
            histogram_freq=1,
            write_graph=True,
            write_images=False,  # Set to False to reduce file size
            update_freq='epoch',
            profile_batch=0
        ),
        
        # Custom metrics logger that saves to CSV
        metrics_logger
    ]

# =============================================================================
# MAIN FUNCTION
# =============================================================================
def main():
    """Train the model and save metrics to CSV files."""
    print("\n" + "="*80)
    print("MULTI-OUTPUT MODEL TRAINING PIPELINE v5")
    print("="*80)
    
    # Preprocess data and show statistics
    print("\nPreprocessing data...")
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data()

    # Reshape for Conv1D
    feature_length = X_train.shape[1]
    X_train = X_train.reshape(-1, feature_length, 1)
    X_val = X_val.reshape(-1, feature_length, 1)
    X_test = X_test.reshape(-1, feature_length, 1)

    # Build model
    print("\nBuilding model...")
    model = build_single_output_model((feature_length, 1))
    
    print("\nModel Architecture Summary:")
    model.summary(line_length=120, show_trainable=True)

    # Train model - always train, no conditional logic
    print("\nTraining model...")
    history = model.fit(
        X_train,
        y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=get_callbacks()
    )

    # Save model summary to text file
    with open(os.path.join(RESULTS_DIR, f'model_summary_v{VERSION}.txt'), 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'),
                      line_length=120,
                      show_trainable=True)

    print("\nEvaluating on TEST set:")
    test_results = model.evaluate(X_test, y_test, verbose=1)
    print(f"  Test Loss: {test_results[0]:.4f}, Test Accuracy: {test_results[1]:.4f}")
    
    # Generate predictions and classification metrics
    print("\nGenerating classification metrics on TEST set...")
    predictions = model.predict(X_test)
    y_pred = np.argmax(predictions, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Save test results to CSV
    class_names = ['Cotton-White', 'Cotton-Black', 'Cotton-Other', 'Polyester-White', 'Polyester-Black', 'Polyester-Other']
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4, output_dict=True)
    
    # Convert classification report to DataFrame and save as CSV
    report_df = pd.DataFrame(report).transpose()
    report_csv_path = os.path.join(RESULTS_DIR, f'classification_report_v{VERSION}.csv')
    report_df.to_csv(report_csv_path)
    print(f"Classification report saved to: {report_csv_path}")
    
    # Save confusion matrix as CSV (not as plot)
    cm = confusion_matrix(y_true, y_pred, normalize='true') * 100
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    cm_csv_path = os.path.join(RESULTS_DIR, f'confusion_matrix_v{VERSION}.csv')
    cm_df.to_csv(cm_csv_path)
    print(f"Confusion matrix saved to: {cm_csv_path}")

    print("\nTraining complete. Model and results saved.")
    print("="*80)
    
    # No plotting here - metrics have been saved to CSV files


if __name__ == '__main__':
    main()
