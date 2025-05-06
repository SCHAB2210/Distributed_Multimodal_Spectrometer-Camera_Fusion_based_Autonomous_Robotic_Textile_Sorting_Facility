
# MAS500 – Model Architectures by Task

This repository includes models developed for textile classification based on NIR spectra, camera images, and sensor fusion.

---

##  Setup

Install all required dependencies:

```bash
pip install -r requirements.txt
```

>  All required packages are listed in [requirements.txt](./requirements.txt).

---

##  Model Architectures by Task

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
| SO3 FUSION – 3.1 Color SP+HR | FCMC-C          | [SO3.1](./SO3/SO3.1/)                          |
| SO3 FUSION – 3.2 Material SP+HR | FCMC-M      | [SO3.2](./SO3/SO3.2/)                          |
| SO3 FUSION – 3.3 Both SP+HR | FCMC-CM         | [SO3.3](./SO3/SO3.3/)                          |
