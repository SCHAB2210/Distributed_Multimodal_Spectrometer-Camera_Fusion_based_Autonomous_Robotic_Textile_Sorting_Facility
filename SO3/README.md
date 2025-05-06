# SO3 – Camera & spectrometer (Fusion)​ classification 

This section includes all models trained and tested using **Near-Infrared (NIR) and camera**  data.

---

##  Subfolders Overview

Each subfolder corresponds to a specific subtask and contains:
- The model architecture and training script
- model features 
- A pretrained `.h5` model
- Evaluation metrics and result visualizations
- A dedicated `README.md` explaining the contents

| Subtask    | Description                        | Folder Link         |
|------------|------------------------------------|---------------------|
| SO3.1      | Color classification  | [SO3.1](./SO3.1/)   |
| SO3.2      | Material classification  | [SO3.2](./SO3.2/)   |
| SO3.3      | Combined color + material          | [SO3.3](./SO3.3/)   |



---

##  Notes

- All models were trained using spectral and image data from textile materials collected with the NIRScan sensor and L515 camera.
- Each model is self-contained and can be tested individually by following its respective `README.md`.


---

##  Back to Project Overview

[ Return to Main README](../README.md)