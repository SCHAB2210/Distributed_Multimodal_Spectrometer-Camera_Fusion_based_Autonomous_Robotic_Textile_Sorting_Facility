
# SO1 – Spectral Classification Models (NIR)

This section includes all models trained and tested using **Near-Infrared (NIR)** spectral data.

---

##  Subfolders Overview

Each subfolder corresponds to a specific subtask and contains:
- The model architecture and training script
- A pretrained `.h5` model
- Evaluation metrics and result visualizations
- A dedicated `README.md` explaining the contents

| Subtask    | Description                        | Folder Link         |
|------------|------------------------------------|---------------------|
| SO1.1      | Color classification (900–1400 nm) | [SO1.1](./SO1.1/)   |
| SO1.2      | Material classification (1400–1700 nm) | [SO1.2](./SO1.2/)   |
| SO1.3      | Combined color + material          | [SO1.3](./SO1.3/)   |
| SO1.4      | Best combined model (fine-tuned)   | [SO1.4](./SO1.4/)   |
| data       | Input datasets used for training   | [data](./data/)     |

---

##  Notes

- All models were trained using spectral data from textile materials collected with the NIRScan sensor.
- Each model is self-contained and can be tested individually by following its respective `README.md`.
- Common datasets (`data_color.csv`, `data_material.csv`, `data_combined.csv`) are stored in the `data/` folder.

---

##  Back to Project Overview

[🔙 Return to Main README](../README.md)
