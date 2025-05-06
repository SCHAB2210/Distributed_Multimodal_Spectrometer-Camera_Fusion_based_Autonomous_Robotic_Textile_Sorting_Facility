
# %%
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import InputLayer

# Load the model
model_path = r"C:\Users\amirs\Desktop\MAS500\MAS500\NIR\results\multi_output_model_combined_v5\multi_output_model_v5_20.h5"
model = load_model(model_path, compile=False)

# Define the correct input shape
input_shape = model.input_shape[1:]  # (256, 256, 3)

# Rebuild the model without the softmax layer
new_model = Sequential()
new_model.add(InputLayer(input_shape=input_shape))  # Proper input layer for Sequential model

# Add all layers except the softmax
for layer in model.layers[:-1]:
    new_model.add(layer)

# Save the model without the softmax layer
save_path = r"C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\Ensemble Model Fusion\features\NIR_combine_model_no_softmax.h5"
new_model.save(save_path)

# %%
from tensorflow.keras.models import Sequential, load_model

# Load the model
model_path = r"C:\Users\amirs\Desktop\MAS500\MAS500\NIR\results\multi_output_model_combined_v5\multi_output_model_v5_20.h5"
model = load_model(model_path, compile=False)

# Rebuild the model without the softmax layer
new_model = Sequential()

# Directly copy all layers except the softmax
for layer in model.layers[:-1]:  # Exclude the softmax layer
    new_model.add(layer)

# Save the model without the softmax layer
save_path = r"C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\Ensemble Model Fusion\features\NIR_combine_model_no_softmax.h5"
new_model.save(save_path)

# %%

from tensorflow.keras.models import load_model, Model

# Load the original model
model_path = r"C:\Users\amirs\Desktop\MAS500\MAS500\NIR\results\multi_output_model_combined_v5\multi_output_model_v5_20.h5"
model = load_model(model_path, compile=False)

# Create a new model that outputs the second-to-last layer (removing the softmax)
new_model = Model(inputs=model.input, outputs=model.layers[-2].output)

# Save the new model
save_path = r"C:\Users\amirs\Desktop\MAS500\MAS500\camera\models\Ensemble Model Fusion\features\NIR_combine_model_no_softmax.h5"
new_model.save(save_path)

# %%
