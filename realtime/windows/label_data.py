import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

CLASS_NAMES = {
    0: "Cotton-White",
    1: "Cotton-Black",
    2: "Cotton-Other",
    3: "Polyester-White",
    4: "Polyester-Black",
    5: "Polyester-Other"
}


def read_data(file_path):
    print(f"\n Reading file: {file_path}")

    # Read CSV file
    df = pd.read_csv(file_path)

    # Verify necessary columns
    if 'Absorbance (AU)' not in df.columns or 'Wavelength (nm)' not in df.columns:
        raise ValueError("Missing required columns in CSV file.")

    # Extract first 228 absorbance values
    absorbance_values = df['Absorbance (AU)'].values[:228]

    if len(absorbance_values) < 228:
        raise ValueError(f"Insufficient data points: got {len(absorbance_values)}, expected 228.")

    # Convert to DataFrame
    df_processed = pd.DataFrame(
        data=[absorbance_values],
        columns=[f"{i}" for i in range(1, 229)]
    )

    # Preprocess — exactly like training
    df_processed = df_processed.fillna(0)
    df_processed = df_processed.replace([np.inf, -np.inf], 0)
    df_processed = df_processed.abs()  # match `.abs()` used in training

    # Debug info
    print(" absorbance length:", df_processed.shape[1])
    print(" First 5 values:", df_processed.iloc[0, :5].tolist())
    print(" Final shape for model:", df_processed.shape)

    return df_processed


def predict_label(file_path, model):
    try:
        x = read_data(file_path)

        # Predict
        prediction = model.predict(x)
        predicted_class = int(np.argmax(prediction, axis=1)[0])
        class_name = CLASS_NAMES.get(predicted_class, "Unknown")

        print(f" Predicted label: {predicted_class} → {class_name}")
        return str(predicted_class)  #  return number, not name
    except Exception as e:
        print(f" Error during prediction: {e}")
        return ''
