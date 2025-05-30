
# MAS500 Project – Model Architectures by Task

This repository contains deep learning models for textile classification using NIR spectral data, camera images, and feature fusion techniques.

---

##  Quick Setup

Follow these steps to clone the project and install dependencies:

```bash
# 1. Clone the repository
git clone https://github.com/SCHAB2210/Distributed_Multimodal_Spectrometer-Camera_Fusion_based_Autonomous_Robotic_Textile_Sorting_Facility.git
cd handover_MAS500

# 2. Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install required packages
pip install -r requirements.txt
```

>  All required packages are listed in [requirements.txt](./requirements.txt).

---

## Model Architectures by Task

| Task                        | Architecture     | Link                                           |
|----------------------------|------------------|------------------------------------------------|
| SO1 NIR – 1.1 Color        | NCC              | [SO1.1](./SO1/SO1.1/)                          |
| SO1 NIR – 1.2 Material     | NMC              | [SO1.2](./SO1/SO1.2/)                          |
| SO1 NIR – 1.3 Both         | NCMC             | [SO1.3](./SO1/SO1.3/)                          |
| SO1 NIR – 1.4 Both         | NCMFC            | [SO1.4](./SO1/SO1.4/)                          |
| SO2 CAM – 2.1 Color LR+SR | CCMC-LR-GAN-3    | [SO2.1](./SO2/SO2.1/)                          |
| SO2 CAM – 2.2 Color HR     | CCMC-HR-3        | [SO2.2](./SO2/SO2.2/)                          |
| SO2 CAM – 2.3 Material LR+SR | CCMC-LR-G-GAN-2 | [SO2.3](./SO2/SO2.3/)                          |
| SO2 CAM – 2.4 Material HR  | CCMC-LR-G-2      | [SO2.4](./SO2/SO2.4/)                          |
| SO2 CAM – 2.5 Both LR+SR  | CCMC-LR-GAN-6    | [SO2.5](./SO2/SO2.5/)                          |
| SO2 CAM – 2.6 Both HR      | CCMC-HR-6        | [SO2.6](./SO2/SO2.6/)                          |
| SO3 FUSION – 3.1 Color NIR+CAM | NCFCC          | [SO3.1](./SO3/SO3.1/)                          |
| SO3 FUSION – 3.2 Material NIR+CAM | NCFMC      | [SO3.2](./SO3/SO3.2/)                          |
| SO3 FUSION – 3.3 Both NIR+CAM | NCFCMC         | [SO3.3](./SO3/SO3.3/)                          |
| Ablation study  – CAM |         | [Ablation study](./SO2/Ablation%20study%20/)                          |
| yolo  – CAM |         | [Yolo](./SO2/YOLO/)                          |
