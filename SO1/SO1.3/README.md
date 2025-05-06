
# SO1.3 – Combined Color & Material Classification (NIR Spectrum)

**Task:** SO1 NIR – 1.3  
**Architecture:** NCMC (NIR Combined Model Classifier)

---

##  Folder Contents

- `code/multi_output_model_v5.py`  
  Python script for training and evaluating a combined classification model that outputs both color and material.

- `model/multi_output_model_v5_20.h5`  
  Trained multi-output Keras model with two prediction heads (color and material).

- `results/`  
  Evaluation results and visualizations:
  - `confusion_matrix.png` – Combined confusion matrix showing model predictions.
  - `model_architecture.png` – Visual summary of the model structure.
  - `model_summary_v20.txt` – Detailed model summary from Keras.
  - `multi_output_model_metrics.png` – Accuracy/loss plots for both outputs.

---

##  How to Run the Model

### 1. Navigate to the directory:
```bash
cd SO1/SO1.3
```

### 2. Run the evaluation script:
```bash
python code/multi_output_model_v5.py
```

>  Required libraries:
> - `tensorflow`, `numpy`, `pandas`, `matplotlib`, `sklearn`

---

##  About the Model

This model performs **dual classification** of textile samples based on NIR spectra in the **900–1700 nm** range:
- Predicts **material type** (cotton or polyester)
- Predicts **color** (white, black, or other)

The model is trained on a **combined dataset** (`data_combined.csv`) located in [`SO1/data`](../data/).

---

##  Related Files

- [Data File](../data/data_combined.csv)
- [Main README](../../README.md)

---
