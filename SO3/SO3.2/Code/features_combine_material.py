# %%
import numpy as np
import pandas as pd
import h5py
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
import tensorflow as tf

# Paths to your models
model1_path = r'C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\Ensemble Model Fusion\features\Camera_material_model_no_softmax.h5'
model2_path = r'C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\Ensemble Model Fusion\features\NIR_material_model_no_softmax.h5'

# Load the models (they are Sequential models with no defined input)
model1 = load_model(model1_path, compile=False)
model2 = load_model(model2_path, compile=False)

# ----- Rebuild Model 1 (RGB) as a functional model with explicit input -----
rgb_input = Input(shape=(256, 256, 3))
x = rgb_input
for layer in model1.layers:
    x = layer(x)
# The full RGB model already outputs a 64-dimensional feature vector.
rgb_feature_extractor = Model(inputs=rgb_input, outputs=x)

# ----- Rebuild Model 2 (NIR) as a functional model with explicit input -----
# For the NIR model, we want to exclude the last layer (assumed to be Dropout) and get the output from the Dense layer before it.
nir_input = Input(shape=(122, 1))
x = nir_input
for layer in model2.layers[:-1]:  # Exclude the last layer (e.g., Dropout)
    x = layer(x)
nir_feature_extractor = Model(inputs=nir_input, outputs=x)

# ----- Load NIR data from CSV -----
nir_data_path = r'C:\Users\amirs\Desktop\MAS500\MAS500\NIR\samples\combined_fixed\data_material_balanced3_fixed.csv'
nir_df = pd.read_csv(nir_data_path)

# Separate features and labels from NIR data
nir_features = nir_df.drop(columns=['label']).values  # (N, 122)
nir_labels = nir_df['label'].values  # (N,)
# Reshape NIR features to match model input shape: (N, 122, 1)
nir_features = nir_features.reshape(-1, 122, 1)

# ----- Load the RGB test dataset -----
rgb_test_directory = r'C:\Users\amirs\Desktop\backup\camera\models\dataset\material\material_HR\train'
image_size_rgb = (256, 256)
batch_size = 32

rgb_test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    rgb_test_directory,
    color_mode='rgb',
    batch_size=batch_size,
    image_size=image_size_rgb,
    shuffle=False,
    seed=120
)

# ----- Feature Extraction Function -----
def extract_and_concatenate_features(rgb_batch, nir_batch):
    # Pass the batch through the functional models
    rgb_feats = rgb_feature_extractor(rgb_batch, training=False).numpy()  # (N, 64)
    nir_feats = nir_feature_extractor(nir_batch, training=False).numpy()  # (N, 64)
    # Concatenate along the feature dimension to form (N, 128)
    concatenated = np.concatenate([rgb_feats, nir_feats], axis=1)
    return concatenated

# ----- Loop Through Datasets and Extract Features -----
all_features = []
all_labels = []
nir_data_index = 0  # To index into the NIR data

for rgb_batch, labels_batch in rgb_test_ds:
    current_bs = rgb_batch.shape[0]
    # Get corresponding NIR vectors
    nir_batch = nir_features[nir_data_index: nir_data_index + current_bs]
    nir_data_index += current_bs
    # Extract and concatenate features
    concat_feats = extract_and_concatenate_features(rgb_batch, nir_batch)
    all_features.append(concat_feats)
    all_labels.append(labels_batch.numpy())

all_features = np.concatenate(all_features, axis=0)  # Expected shape: (N, 128)
all_labels = np.concatenate(all_labels, axis=0).reshape(-1, 1)  # Expected shape: (N, 1)

print("All features shape:", all_features.shape)
print("All labels shape:", all_labels.shape)

# ----- Save the Extracted Features and Labels -----
save_path = r'C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\Ensemble Model Fusion\features\combined_mererial_features_labels.h5'
with h5py.File(save_path, 'w') as h5f:
    h5f.create_dataset('features', data=all_features)
    h5f.create_dataset('labels', data=all_labels)

print(f"Features and labels saved successfully to {save_path}")
# %%
