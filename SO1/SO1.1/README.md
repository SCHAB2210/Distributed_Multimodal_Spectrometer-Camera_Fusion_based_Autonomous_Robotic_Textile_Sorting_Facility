# SO1.1 – Color Classification range 900-1400nm (NIR Spectrum)

**Task:** SO1 NIR – 1.1  
**Architecture:** NCC (NIR Color Classifier)

---

## Folder Contents

- `code/color_model.py`  
  Python script for training and evaluating the color classification model.

- `model/color_model.h5`  
  Pretrained Keras model for color classification using NIR spectrum data.

- `results/`  
  Folder containing model evaluation results:
  - `color_confusion_matrix_report.png` – Confusion matrix image showing prediction accuracy.
  - `color_model_metrics.png` – Training/validation accuracy and loss plots.
  - `color_model_summary_v1.txt` – Keras model architecture summary.

---

## How to Run the Model

### 1. Navigate to the directory:
```bash
cd SO1/SO1.1
```

### 2. Run the evaluation script:
```bash
python code/color_model.py
```

>  Make sure you have the required libraries installed:
> - `tensorflow`, `numpy`, `pandas`, `matplotlib`, `sklearn`

---

##  About the Model

This model is designed to classify the **color** of textile materials based on their NIR (Near-Infrared) spectra, specifically from the **900–1400 nm** region. It was trained using the `data_color.csv` file located in [`SO1/data`](../data/).

**Color classes:**
- White
- Black
- Other

---

##  Related Files

- [Data File](../data/data_color.csv)
- [Main README](../../README.md)

---
