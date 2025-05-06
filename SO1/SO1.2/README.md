
# SO1.2 – Material Classification (NIR Spectrum)

**Task:** SO1 NIR – 1.2  
**Architecture:** NMC (NIR Material Classifier)

---

##  Folder Contents

- `code/material_model.py`  
  Python script used to train and evaluate the material classification model.

- `model/material_model.h5`  
  Pretrained Keras model trained on the NIR material dataset.

- `results/`  
  Contains evaluation output and visualizations:
  - `material_confusion_matrix.png` – Confusion matrix showing classification performance.
  - `material_model_metrics.png` – Accuracy/loss curves over training epochs.
  - `material_model_summary_v1.txt` – Keras model architecture and parameter summary.

---

##  How to Run the Model

### 1. Navigate to the directory:
```bash
cd SO1/SO1.2
```

### 2. Run the evaluation script:
```bash
python code/material_model.py
```

>  Required libraries:
> - `tensorflow`, `numpy`, `pandas`, `matplotlib`, `sklearn`

---

##  About the Model

This model classifies the **material type** of textile samples using spectral data from the **1300–1700 nm** wavelength range.  
It was trained using the `data_material.csv` file found in [`SO1/data`](../data/).

**Material classes:**
- Cotton
- Polyester

---

##  Related Files

- [Data File](../data/data_material.csv)
- [Main README](../../README.md)

---
