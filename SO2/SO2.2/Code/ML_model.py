# NR 6
#%%
import numpy as np
import os
import PIL #pip install PIL
import PIL.Image 
import tensorflow as tf
#from tensorflow import keras
import tensorflow_datasets as tfds # need to install this seperately - pip install tensorflow_datasets
import pathlib
from skimage import io
import datetime
import matplotlib.pyplot as plt
import h5py #pip install h5py
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import confusion_matrix, classification_report




#%% define training data set at custom location


# Define the directories for train, validation, and test datasets
train_directory = r'C:\Users\amirs\Desktop\backup\camera\models\dataset\color\own_color_model_HR_2\dataset\train'
val_directory = r'C:\Users\amirs\Desktop\backup\camera\models\dataset\color\own_color_model_HR_2\dataset\val'
test_directory = r'C:\Users\amirs\Desktop\backup\camera\models\dataset\color\own_color_model_HR_2\dataset\test'
#test_directory=r'C:\Users\amirs\OneDrive\Skrivebord\mas512\assignemt2\Q3\tested'

image_size = (256, 256)  # Resize images to 180x180
batch_size = 32

# Load the training dataset
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory,
    batch_size=batch_size,
    image_size=image_size,
    seed=123
)

# Find the class names (folders are used as labels)
class_names = train_ds.class_names
print(class_names)

# Load the validation dataset
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_directory,
    batch_size=batch_size,
    image_size=image_size,
    seed=123
)

# Load the test dataset
test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_directory,
    batch_size=batch_size,
    image_size=image_size,
    shuffle=False,  # No need to shuffle test data
    seed=120
)


#%%
#look into training data shape and label
for image_batch, labels_batch in train_ds:
  print(image_batch.shape)  #(32,180,180,3)- one batch consists of 32 images of shape , image size, channel #
  print(labels_batch.shape) # 32 - corresponding label of 32 images in batch
  #print(train_ds.list_files)
  break



#%% visualize data

plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
 for i in range(9):
   ax = plt.subplot(3, 3, i + 1)
   plt.imshow(images[i].numpy().astype("uint8"))
   plt.title(class_names[labels[i]])
   plt.axis("off")
plt.show()






#%% train, compile, fit model

num_classes = 3
# Define the model based on the corrected architecture
model = tf.keras.Sequential([
    
    # Normalization Layer
    tf.keras.layers.experimental.preprocessing.Rescaling(1./255, input_shape=(256, 256, 3)),

    # First Convolutional Block
    tf.keras.layers.Conv2D(8, (3, 3), activation='relu'),  # Conv2D Layer 1: 8 filters, 3x3 kernel
    tf.keras.layers.MaxPooling2D(),                        # MaxPooling Layer 1
    # End of First Convolutional Block

    # Second Convolutional Block
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu'), # Conv2D Layer 2: 16 filters, 3x3 kernel
    tf.keras.layers.MaxPooling2D(),                        # MaxPooling Layer 2
    # End of Second Convolutional Block

    # Third Convolutional Block (Without MaxPooling)
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # Conv2D Layer 3: 64 filters, 3x3 kernel
    # End of Third Convolutional Block

    # Fourth Convolutional Block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # Conv2D Layer 4: 64 filters, 3x3 kernel
    tf.keras.layers.MaxPooling2D(),                        # MaxPooling Layer 4
    # End of Fourth Convolutional Block

    # Fifth Convolutional Block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # Conv2D Layer 5: 64 filters, 3x3 kernel
    tf.keras.layers.MaxPooling2D(),                        # MaxPooling Layer 5
    # End of Fifth Convolutional Block

    # Sixth Convolutional Block
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'), # Conv2D Layer 6: 64 filters, 3x3 kernel
    tf.keras.layers.MaxPooling2D(),                        # MaxPooling Layer 6
    # End of Sixth Convolutional Block

    # Flattening Block
    tf.keras.layers.Flatten(),                             # Flatten Layer
    # End of Flattening Block

    # Fully Connected Block 1
    tf.keras.layers.Dense(64, activation='relu',           # Fully Connected (Dense) Layer 1: 64 units
                          activity_regularizer=tf.keras.regularizers.L2(0.01)),
    # End of Fully Connected Block 1

    # Fully Connected Block 2
    tf.keras.layers.Dense(64, activation='relu',           # Fully Connected (Dense) Layer 2: 64 units
                          activity_regularizer=tf.keras.regularizers.L2(0.01)),
    # End of Fully Connected Block 2

    # Output Block
    tf.keras.layers.Dense(num_classes, activation='softmax')         # Output Layer: 2 units (for binary classification)
    # End of Output Block
])
#tf.keras.Sequential([model, tf.keras.layers.Softmax()])
    
# use optimizer - adam, use loss function - SparseCategoricalCrossentropy


    
model.compile(
#optimizer='sgd',
optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
metrics=['accuracy'],
) # accuracy  see training and validation accuracy in each epoch   


#log_path=r'C:\Users\ajitj\OneDrive - Universitetet i Agder\PhD_Research\NN_Practice\Classification_Custom\'
log_path=r'C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\own_model\Color_model_HR\saved_model'
log_dir = log_path + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

#log_dir = directory + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorboard_callback = tf.keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1,
        write_graph=False,
        write_images=False, #write model weights to visualize as image in TensorBoard.
        write_steps_per_second=False,
        update_freq='epoch', #'batch'
        profile_batch=0,
        embeddings_freq=0,
        embeddings_metadata=None,
        #**kwargs
)

# Define Early Stopping Callback
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',   # Monitor validation loss
    patience=20,          # Stop training if val_loss doesn't improve for 10 epochs
    restore_best_weights=True,  # Restore model weights from the best epoch
    verbose=1
)

history=model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=500,
  verbose=1,
  callbacks=[tensorboard_callback , early_stopping]
)
# import os
# os.system('python -m tensorflow.tensorboard --logdir=' + log_dir)
#tensorboard --logdir log_dir




#%%
#evaluate model - https://www.tensorflow.org/tutorials/images/cnn
print(history.history.keys()) # see the key parameters stored in history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
#plt.ylim([0.01, 1])
plt.legend(loc='upper right')
plt.show()


plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_los')
plt.xlabel('Epoch')
plt.ylabel('loss')
#plt.ylim([0.5, 5])
plt.legend(loc='upper right')
plt.show()

#%% save weights
model_dir=r'C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\own_model\Color_model_HR\saved_model'
#model_save_dir = model_dir + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
ap=datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
model_save_path=os.path.join(model_dir,  ap + '.' + 'h5')
model.save(model_save_path)




#%% reload model
loaded_model = tf.keras.models.load_model(model_save_path) #model_save_path

# Check its architecture
loaded_model.summary()







#%% confusion matrix

# Get the predicted labels and true labels for the validation set
y_pred = np.argmax(loaded_model.predict(test_ds), axis=-1) #val_ds
y_true = np.concatenate([y for x, y in test_ds], axis=0)
# Compute the confusion matrix
cm=confusion_matrix(y_true, y_pred)
#C_{i, j} is equal to the number of observations known to be in group i but predicted to be in group j
#normalize
cm_n = cm / cm.sum(axis=1)[:, np.newaxis]

import seaborn as sn
#cm1=tf.math.confusion_matrix(y_true, y_pred)

class_names = ['Black','Other','White']

sn.heatmap(
        cm_n, #cm_n for %
        xticklabels=class_names, #comment this line to get label in numeric
        yticklabels=class_names, #comment this line to get label in numeric
        annot=True, 
        fmt='.2f',
        annot_kws={"size": 16},      
) # font size
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()




# %%
