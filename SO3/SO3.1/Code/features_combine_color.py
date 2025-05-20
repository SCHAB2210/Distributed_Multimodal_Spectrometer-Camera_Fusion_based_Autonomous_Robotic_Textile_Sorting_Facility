# %%
import numpy as np
import pandas as pd
import h5py
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input
import tensorflow as tf

# Paths to your models
model1_path = r'C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\Ensemble Model Fusion\features\Camera_Color_model_no_softmax.h5'
model2_path = r'C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\Ensemble Model Fusion\features\NIR_Color_model_no_softmax.h5'


model1 = load_model(model1_path, compile=False)
model2 = load_model(model2_path, compile=False)


rgb_input = Input(shape=(256, 256, 3))
x = rgb_input
for layer in model1.layers:
    x = layer(x)

rgb_feature_extractor = Model(inputs=rgb_input, outputs=x)


nir_input = Input(shape=(105, 1))
x = nir_input
for layer in model2.layers[:-1]:  
    x = layer(x)
nir_feature_extractor = Model(inputs=nir_input, outputs=x)

# ----- Load NIR data from CSV -----
nir_data_path = r'C:\Users\amirs\Desktop\MAS500\MAS500\NIR\samples\combined_fixed\data_color_balanced3_fixed.csv'
nir_df = pd.read_csv(nir_data_path)


nir_features = nir_df.drop(columns=['label']).values  
nir_labels = nir_df['label'].values  


nir_labels_original = np.copy(nir_labels)
nir_labels_remapped = np.copy(nir_labels_original)
nir_labels_remapped[nir_labels_original == 0] = 2
nir_labels_remapped[nir_labels_original == 1] = 0
nir_labels_remapped[nir_labels_original == 2] = 1

nir_labels = nir_labels_remapped


nir_features = nir_features.reshape(-1, 105, 1)


rgb_test_directory = r'C:\Users\amirs\Desktop\backup\camera\models\dataset\own_color_model_HR_1\dataset\train'
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


def extract_and_concatenate_features(rgb_batch, nir_batch):
    
    rgb_feats = rgb_feature_extractor(rgb_batch, training=False).numpy()  
    nir_feats = nir_feature_extractor(nir_batch, training=False).numpy()  
    # Concatenate the features along the last dimension to form (N, 128)
    concatenated = np.concatenate([rgb_feats, nir_feats], axis=1)
    return concatenated


all_features = []
all_labels = []
nir_data_index = 0  

for rgb_batch, labels_batch in rgb_test_ds:
    current_bs = rgb_batch.shape[0]
    
    nir_batch = nir_features[nir_data_index: nir_data_index + current_bs]
    nir_data_index += current_bs
    
    concat_feats = extract_and_concatenate_features(rgb_batch, nir_batch)
    all_features.append(concat_feats)
    all_labels.append(labels_batch.numpy())

all_features = np.concatenate(all_features, axis=0)  
all_labels = np.concatenate(all_labels, axis=0).reshape(-1, 1)  

print("All features shape:", all_features.shape)
print("All labels shape:", all_labels.shape)


save_path = r'C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\Ensemble Model Fusion\features\combined_features_labels.h5'
with h5py.File(save_path, 'w') as h5f:
    h5f.create_dataset('features', data=all_features)
    h5f.create_dataset('labels', data=all_labels)

print(f"Features and labels saved successfully to {save_path}")

# %%
