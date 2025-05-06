
# SO1.4 – Combined Classification Using Statistical Features

**Task:** SO1 NIR – 1.4  
**Architecture:** NCMFC (NIR Combined Model using Feature Compression)

---

##  Folder Contents

- `code/`  
  *(May be empty or reused from previous models. If a script is added, document it here.)*

- `model/best_model.h5`  
  Trained Keras model using statistical features (mean, std, min, max) instead of full spectral data.

- `results/`  
  *(May include performance metrics if added later.)*

---

##  How to Run the Model

```bash
python ../SO1.3/code/multi_output_model_v5.py --model_path model/best_model.h5
```

>  Required libraries:
> - `tensorflow`, `numpy`, `pandas`, `matplotlib`, `sklearn`

---

##  About the Model

Unlike SO1.3, which uses the full **228-point NIR spectrum**, this model is based on **statistical features** extracted from the spectrum:
- **Mean**
- **Standard Deviation**
- **Minimum**
- **Maximum**

These features are calculated separately for color and material spectral ranges and used as compressed inputs to a smaller, faster model.

---

##  Related Files

- [Combined Dataset](../data/data_combined.csv)
- [Main README](../../README.md)

---
