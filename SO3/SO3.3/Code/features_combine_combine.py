# %%
import numpy as np
import pandas as pd
import h5py
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
import tensorflow as tf

# Paths to your models
model1_path = r'C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\Ensemble Model Fusion\features\camera_combine_model_no_softmax.h5'
model2_path = r'C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\Ensemble Model Fusion\features\NIR_combine_model_no_softmax.h5'

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
# Instead of manually iterating through layers (which fails on merge layers), we build a new model 
# that uses the original model2's input and takes the penultimate layer's output (excluding the last Dropout).
# Note: The saved NIR model expects an input shape of (228, 1) as shown in its summary.
nir_feature_extractor = Model(inputs=model2.input, outputs=model2.layers[-2].output)

# ----- Load NIR data from CSV -----
nir_data_path = r'C:\Users\amirs\Desktop\MAS500\MAS500\NIR\samples\combined_fixed\data_combined_balanced3_fixed.csv'
nir_df = pd.read_csv(nir_data_path)

# Separate features and labels from NIR data
# Assumes that the CSV file has 229 columns (228 features + 1 label). Adjust if necessary.
nir_features = nir_df.drop(columns=['label']).values  # Expected shape: (N, 228)
nir_labels = nir_df['label'].values  # (N,)
# Reshape NIR features to match model input shape: (N, 228, 1)
nir_features = nir_features.reshape(-1, 228, 1)

# ----- Load the RGB test dataset -----
rgb_test_directory = r'C:\Users\amirs\Desktop\backup\camera\models\combine HR\train'
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
    # Extract features from the RGB and NIR models.
    # The RGB model outputs a 64-d vector.
    rgb_feats = rgb_feature_extractor(rgb_batch, training=False).numpy()  # (N, 64)
    # The NIR feature extractor now outputs the penultimate layer's features (expected to be 64-d).
    nir_feats = nir_feature_extractor(nir_batch, training=False).numpy()  # (N, 64)
    # Concatenate along the feature dimension to form a combined (N, 128) feature vector.
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
save_path = r'C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\Ensemble Model Fusion\features\combined_combined_features_labels.h5'
with h5py.File(save_path, 'w') as h5f:
    h5f.create_dataset('features', data=all_features)
    h5f.create_dataset('labels', data=all_labels)

print(f"Features and labels saved successfully to {save_path}")
# %%
