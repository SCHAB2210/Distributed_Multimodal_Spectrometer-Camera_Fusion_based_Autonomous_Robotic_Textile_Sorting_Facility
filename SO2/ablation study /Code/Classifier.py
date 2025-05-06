#%%
import os
import numpy as np
import h5py
import tensorflow as tf
from sklearn.model_selection import train_test_split
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# Configure paths
FEATURES_PATH = r"C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\Ensemble Learning\feature\test\features_resnet_nasnet.h5"
MODEL_DIR = r"C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\Ensemble Learning\feature\test"
LOG_DIR = os.path.join(MODEL_DIR, 'logs', datetime.now().strftime("%Y%m%d-%H%M%S"))

# Training parameters
BATCH_SIZE = 32
EPOCHS = 500
LEARNING_RATE = 0.001

def load_data():
    """Load features and labels from H5 file."""
    with h5py.File(FEATURES_PATH, 'r') as h5f:
        features = h5f['features'][:]
        labels = h5f['labels'][:]
    
    # Convert labels to categorical
    labels = tf.keras.utils.to_categorical(labels, num_classes=6)
    return features, labels

def build_model(input_shape):
    """Build neural network for feature-based classification."""
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=input_shape),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Dense(6, activation='softmax')  # 2 classes
    ])
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def plot_training_history(history, save_path):
    """Plot and save training history."""
    plt.figure(figsize=(12, 4))
    
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
    plt.show()
    plt.close()

def plot_confusion_matrix(y_true, y_pred, save_path):
    """Plot and save confusion matrix with percentage values."""
    class_names = ['Cotton white','Cotton black']
    
    # Calculate confusion matrix
    cm = confusion_matrix(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
    
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
    plt.show()
    plt.close()

def main():
    # Create output directories
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Load data
    print("Loading data...")
    features, labels = load_data()
    
    # Split into train and test sets
    train_features, test_features, train_labels, test_labels = train_test_split(
        features, labels, test_size=0.2, random_state=42
    )
    
    # Build and train model
    print("Building and training model...")
    model = build_model((train_features.shape[1],))
    
    # Define callbacks
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            os.path.join(MODEL_DIR, 'best_model.h5'),
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=10, min_lr=1e-6),
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR)
    ]
    
    # Train model
    history = model.fit(
        train_features, train_labels,
        validation_data=(test_features, test_labels),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks
    )
    
    # Evaluate model
    print("\nEvaluating model...")
    test_loss, test_accuracy = model.evaluate(test_features, test_labels)
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Generate predictions
    predictions = model.predict(test_features)
    
    # Plot results
    plot_training_history(history, os.path.join(MODEL_DIR, 'training_history.png'))
    plot_confusion_matrix(test_labels, predictions, os.path.join(MODEL_DIR, 'confusion_matrix.png'))
    
    # Print classification report
    print("\nClassification Report:")
    #class_names = ['Cotton white','Cotton black','Cotton other', 'Polyester white','Polyester black','Polyester other',]
    class_names = ['Cotton white','Cotton black']
    print(classification_report(
        np.argmax(test_labels, axis=1),
        np.argmax(predictions, axis=1),
        target_names=class_names
    ))

if __name__ == "__main__":
    main()


# %%
