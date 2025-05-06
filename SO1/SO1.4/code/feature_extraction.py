#%%
import tensorflow as tf
import numpy as np
import h5py
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Define Parameters
DATASET_PATH = r"C:\Users\devTe\Desktop\TextileSorting\NIR\samples\combined_fixed\data_combined_balanced3_fixed.csv"
OUTPUT_DIR = r"C:\Users\devTe\Desktop\TextileSorting\NIR\features"

def load_and_split_data():
    """Load spectral data and split into train/test sets."""
    print("\nLoading dataset...")
    df = pd.read_csv(DATASET_PATH)
    
    # Split features and labels and reshape labels to 2D
    X = df.iloc[:, :-1].values  # All columns except last
    y = df.iloc[:, -1].values.reshape(-1, 1)   # Reshape labels to (n_samples, 1)
    
    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Data loaded: {X.shape[0]} samples, {X.shape[1]} wavelength points")
    return X_train, X_test, y_train, y_test

def extract_spectral_features(X_data, pca=None, scaler=None, train=False):
    """Extract features from spectral data."""
    features = []
    
    # Split into color and material regions
    color_region = X_data[:, :105]    # Points 1-105 (900-1300nm)
    material_region = X_data[:, 105:] # Points 106-228 (1300-1700nm)
    
    # Calculate statistical features
    features.append(np.mean(color_region, axis=1).reshape(-1, 1))
    features.append(np.std(color_region, axis=1).reshape(-1, 1))
    features.append(np.max(color_region, axis=1).reshape(-1, 1))
    features.append(np.min(color_region, axis=1).reshape(-1, 1))
    
    features.append(np.mean(material_region, axis=1).reshape(-1, 1))
    features.append(np.std(material_region, axis=1).reshape(-1, 1))
    features.append(np.max(material_region, axis=1).reshape(-1, 1))
    features.append(np.min(material_region, axis=1).reshape(-1, 1))
    
    # Combine all features
    combined_features = np.hstack(features)
    
    # Apply PCA if needed
    if train:
        pca = PCA(n_components=min(64, combined_features.shape[1]))
        scaler = StandardScaler()
        combined_features = pca.fit_transform(combined_features)
        combined_features = scaler.fit_transform(combined_features)
    else:
        combined_features = pca.transform(combined_features)
        combined_features = scaler.transform(combined_features)
    
    return combined_features, pca, scaler

def main():
    # Load and split data
    X_train, X_test, y_train, y_test = load_and_split_data()
    
    # Extract features for training set
    print("\nExtracting training features...")
    train_features, pca, scaler = extract_spectral_features(X_train, train=True)
    
    # Save training features
    TRAIN_H5_PATH = f"{OUTPUT_DIR}/train_spectral_features.h5"
    with h5py.File(TRAIN_H5_PATH, "w") as h5f:
        h5f.create_dataset("features", data=train_features)
        h5f.create_dataset("labels", data=y_train)
    
    # Extract features for test set
    print("\nExtracting test features...")
    test_features, _, _ = extract_spectral_features(X_test, pca=pca, scaler=scaler)
    
    # Save test features
    TEST_H5_PATH = f"{OUTPUT_DIR}/test_spectral_features.h5"
    with h5py.File(TEST_H5_PATH, "w") as h5f:
        h5f.create_dataset("features", data=test_features)
        h5f.create_dataset("labels", data=y_test)
    
    print("\nFeature extraction completed!")
    print(f"Training features shape: {train_features.shape}")
    print(f"Test features shape: {test_features.shape}")

if __name__ == "__main__":
    main()
