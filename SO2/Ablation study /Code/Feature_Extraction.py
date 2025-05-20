#%% Cell 1: Setup and Data Preparation
import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import ResNet101, InceptionResNetV2, NASNetMobile
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define dataset path and base directory for saving files
dataset_path = r'C:\Users\amirs\Desktop\backup\camera\models\Ensemble Learning\raw_gray_dataset\train'
base_dir = r'C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\Ensemble Learning\feature\test'
final_h5_file_path = os.path.join(base_dir, 'features.h5')

# Create ImageDataGenerator and data generator
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32
target_size = (256, 256)
data_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Compute steps per epoch for prediction
steps = data_generator.samples // batch_size
if data_generator.samples % batch_size != 0:
    steps += 1

#%% Cell 2: Extract Individual Model Features
def build_feature_extractor(model_fn):
    base_model = model_fn(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    return Model(inputs=base_model.input, outputs=x)

# Build the pre-trained models
resnet_model = build_feature_extractor(ResNet101)
inception_model = build_feature_extractor(InceptionResNetV2)
nasnet_model = build_feature_extractor(NASNetMobile)

# Extract features with ResNet101
print("Extracting features with ResNet101...")
data_generator.reset()
resnet_features = resnet_model.predict(data_generator, steps=steps, verbose=1)
labels = data_generator.classes.reshape(-1, 1)
resnet_h5_path = os.path.join(base_dir, 'features_resnet101.h5')
with h5py.File(resnet_h5_path, 'w') as h5f:
    h5f.create_dataset("features", data=resnet_features)
    h5f.create_dataset("labels", data=labels)
print(f"Saved ResNet101 features to {resnet_h5_path}")

# Extract features with InceptionResNetV2
print("Extracting features with InceptionResNetV2...")
data_generator.reset()
inception_features = inception_model.predict(data_generator, steps=steps, verbose=1)
inception_h5_path = os.path.join(base_dir, 'features_inceptionResNetV2.h5')
with h5py.File(inception_h5_path, 'w') as h5f:
    h5f.create_dataset("features", data=inception_features)
    h5f.create_dataset("labels", data=labels)
print(f"Saved InceptionResNetV2 features to {inception_h5_path}")

# Extract features with NASNetMobile
print("Extracting features with NASNetMobile...")
data_generator.reset()
nasnet_features = nasnet_model.predict(data_generator, steps=steps, verbose=1)
nasnet_h5_path = os.path.join(base_dir, 'features_nasnetmobile.h5')
with h5py.File(nasnet_h5_path, 'w') as h5f:
    h5f.create_dataset("features", data=nasnet_features)
    h5f.create_dataset("labels", data=labels)
print(f"Saved NASNetMobile features to {nasnet_h5_path}")

#%% Cell 3: Pairwise Feature Combination
def load_features(file_path):
    with h5py.File(file_path, 'r') as h5f:
        features = np.array(h5f['features'])
        labels = np.array(h5f['labels'])
    return features, labels

# Load features for each model
resnet_feat, labels_resnet = load_features(resnet_h5_path)
inception_feat, labels_inception = load_features(inception_h5_path)
nasnet_feat, labels_nasnet = load_features(nasnet_h5_path)

# Combine ResNet101 and InceptionResNetV2 features
if not np.array_equal(labels_resnet, labels_inception):
    raise ValueError("Labels from ResNet101 and InceptionResNetV2 do not match!")
features_resnet_inception = np.concatenate([resnet_feat, inception_feat], axis=1)
pair_inception_path = os.path.join(base_dir, 'features_resnet_inception.h5')
with h5py.File(pair_inception_path, 'w') as h5f:
    h5f.create_dataset("features", data=features_resnet_inception)
    h5f.create_dataset("labels", data=labels_resnet)
print(f"Saved pairwise features (ResNet101 + InceptionResNetV2) to {pair_inception_path}")

# Combine ResNet101 and NASNetMobile features
if not np.array_equal(labels_resnet, labels_nasnet):
    raise ValueError("Labels from ResNet101 and NASNetMobile do not match!")
features_resnet_nasnet = np.concatenate([resnet_feat, nasnet_feat], axis=1)
pair_resnet_nasnet_path = os.path.join(base_dir, 'features_resnet_nasnet.h5')
with h5py.File(pair_resnet_nasnet_path, 'w') as h5f:
    h5f.create_dataset("features", data=features_resnet_nasnet)
    h5f.create_dataset("labels", data=labels_resnet)
print(f"Saved pairwise features (ResNet101 + NASNetMobile) to {pair_resnet_nasnet_path}")

# Combine InceptionResNetV2 and NASNetMobile features
if not np.array_equal(labels_inception, labels_nasnet):
    raise ValueError("Labels from InceptionResNetV2 and NASNetMobile do not match!")
features_inception_nasnet = np.concatenate([inception_feat, nasnet_feat], axis=1)
pair_inception_nasnet_path = os.path.join(base_dir, 'features_inception_nasnet.h5')
with h5py.File(pair_inception_nasnet_path, 'w') as h5f:
    h5f.create_dataset("features", data=features_inception_nasnet)
    h5f.create_dataset("labels", data=labels_inception)
print(f"Saved pairwise features (InceptionResNetV2 + NASNetMobile) to {pair_inception_nasnet_path}")

#%% Cell 4: Triple Feature Combination (All Three Models)
# Concatenate features from all three models
features_all = np.concatenate([resnet_feat, inception_feat, nasnet_feat], axis=1)
with h5py.File(final_h5_file_path, 'w') as h5f:
    h5f.create_dataset("features", data=features_all)
    h5f.create_dataset("labels", data=labels)
print(f"Saved combined features (all three models) to {final_h5_file_path}")



# %%
# %% full VGG16

import os
import h5py
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# 1. Set up your dataset path

dataset_path = r'C:\Users\amirs\Desktop\backup\camera\models\dataset\material SR'  # Change this if needed


# 2. Create a data generator

datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32
target_size = (256, 256)  # VGG16 default input size

data_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=target_size,
    batch_size=batch_size,
    class_mode='categorical',  # Use 'categorical' for multiple classes
    shuffle=False
)


# 3. Load the pre-trained model for feature extraction

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(256, 256, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
feature_extractor = Model(inputs=base_model.input, outputs=x)


# 4. Extract features from dataset

steps = data_generator.samples // batch_size
if data_generator.samples % batch_size != 0:
    steps += 1

print("Extracting features from images...")
features = feature_extractor.predict(data_generator, steps=steps, verbose=1)

# Extract labels and reshape them to (N,1)
labels = data_generator.classes.reshape(-1, 1)  # <-- Fix: Reshaping labels to (N,1)


# 5. Save Features and Labels in a Single HDF5 File

h5_file_path = r'C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\Ensemble Learning\feature\material\SR features\features_VGG16.h5'

with h5py.File(h5_file_path, 'w') as h5f:
    h5f.create_dataset("features", data=features)
    h5f.create_dataset("labels", data=labels)

print(f"Feature data saved successfully in {h5_file_path}")
