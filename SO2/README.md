
# SO2 –  image classification (camera)

This section includes all models trained and tested using **Camera** image data.

---

##  Subfolders Overview

Each subfolder corresponds to a specific subtask and contains:
- The model architecture and training script
- A pretrained `.h5` model
- Evaluation metrics and result visualizations
- A dedicated `README.md` explaining the contents

| Subtask    | Description                        | Folder Link         |
|------------|------------------------------------|---------------------|
| SO2.1      | Color classification (LR+SR) | [SO2.1](./SO2.1/)   |
| SO2.2      | Color classification (HR) | [SO2.2](./SO2.2/)   |
| SO2.3      | Material classification (LR+SR) | [SO2.3](./SO2.3/)   |
| SO2.4      | Material classification (HR) | [SO2.4](./SO2.4/)   |
| SO2.5      | Combined color + material (LR+SR)          | [SO2.5](./SO2.5/)   |
| SO2.6      | Combined color + material (HR)  | [SO2.6](./SO2.6/)   |
| Ablation study      | Combined and material | [Ablation study ](./Ablation%20study%20/)   |
| Yolo     | classification and Detection | [Yolo](./YOLO/)   |
| data       | Input datasets used for training   | [data](./DATA/)     |

---

##  Notes

- All models were trained using image data from textile materials collected with the L515 camera.
- Each model is self-contained and can be tested individually by following its respective `README.md`.


---

##  Back to Project Overview

[ Return to Main README](../README.md)