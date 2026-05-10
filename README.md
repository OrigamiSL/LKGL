# LKGL: Enhancing Cross-View Geo-localization via Global--Local Feature Matching and Keypoint Guidance

[![GitHub Stars](https://img.shields.io/github/stars/OrigamiSL/LKGL?style=social)](https://github.com/OrigamiSL/LKGL)

The official PyTorch implementation of **LKGL** (submitted to IEEE TCSVT). 

## 📖 Abstract
Cross-view geo-localization (CVGL), which aims to match ground- or drone-view imagery with satellite maps, has emerged as a fundamental task in remote sensing, with broad applications in urban monitoring and unmanned aerial vehicle (UAV) localization. Despite remarkable progress, existing approaches rely heavily on metric-based comparisons of global image features, which often leads to mismatches in regions with highly similar global appearances but distinct local structures. To address this limitation, this work introduces a novel CVGL framework, termed LKGL, which integrates two complementary methods. First, the global--local feature matching (GLFM) method computes similarity at both holistic and localized levels and integrates a reliability-based filtering mechanism to remove ambiguous or low-confidence local correspondences, thereby enhancing robustness against confusing global patterns. Second, the keypoint-guided feature pyramid network (KGFPN) integrates multiscale features with spatial priors derived from keypoint maps through attention-based interactions between shallow and deep layers. This design further enhances local feature expressiveness and structural sensitivity. Together, the GLFM and KGFPN enable LKGL to balance global contextual information with fine-grained local cues, achieving state-of-the-art performance on University-1652 and SUES-200. The source code is released on [https://github.com/OrigamiSL/LKGL](https://github.com/OrigamiSL/LKGL).

## 🖼️ Architecture
![Architecture Overview](img/architecture.png)

## ⚙️ Environment Setup

You can easily set up the environment using Conda and the provided `LKGL.yml` file.

```bash
conda env create -f LKGL.yml
conda activate LKGL
```
### Key Dependency Versions:
* **Python:** 3.9.23
* **PyTorch:** 2.6.0+cu118
* **NumPy:** 2.0.2
* **CUDA Toolkit:** 11.8

## 📂 Datasets & Preparation

We evaluate our models on four datasets: **University-1652**, **SUES-200**, **Multi-weather University-1652**, and **DenseUAV**.

### 1. Dataset Acquisition
* **University-1652:** Request from the official repository at [layumi/University1652-Baseline](https://github.com/layumi/University1652-Baseline).
* **SUES-200:** Download from [Reza-Zhu/SUES-200-Benchmark](https://github.com/Reza-Zhu/SUES-200-Benchmark).
* **DenseUAV:** Request details can be found at [Dmmm1997/DenseUAV](https://github.com/Dmmm1997/DenseUAV).
* **Multi-weather University-1652:** This dataset is simulated and derived from the original University-1652 dataset.

### 2. Directory Structures


**University-1652 & Multi-weather University-1652**
```
    ├── University-1652
    │   ├── train
    │   │   ├── drone
    │   │   ├── google
    │   │   ├── satellite
    │   │   └── street
    │   ├── test
    │   │   ├── 4K_drone
    │   │   ├── gallery_drone
    │   │   ├── gallery_satellite
    │   │   ├── gallery_street
    │   │   ├── query_drone
    │   │   ├── query_satellite
    │   │   └── query_street
```
**SUES-200**
*Note: You need to split the origin dataset into the appropriate format using the split_datasets.py script. The processed format should be:*
```
    ├─ SUES-200
      ├── Training
        ├── 150/
        ├── 200/
        ├── 250/
        └── 300/
      ├── Testing
        ├── 150/
        ├── 200/ 
        ├── 250/	
        └── 300/
```
**DenseUAV**
```
    ├── DenseUAV
    │   ├── Dense_GPS_ALL.txt
    │   ├── Dense_GPS_test.txt
    │   ├── Dense_GPS_train.txt
    │   ├── train
    │   │   ├── drone
    │   │   └── satellite
    │   ├── test
    │   │   ├── query_drone
    │   │   └── gallery_satellite
```
## 📦 Pre-trained Weights

Please download the necessary pre-trained weights and place them in the pretrained/ directory before training.

1. **ALIKED-N(16, rot):** Provided at pretrained/aliked-n16rot.pth.
2. **ConvNeXt-Base:** Download [here](https://mega.nz/file/ONA1RYiJ#jhm9skMDrizYBle_DYHREt4ZYlM0vAl6IT-lVUp8RjU) and save to pretrained/convnext_base_22k_1k_224.pth.
3. **DINOv2-Base:** Download [here](https://mega.nz/file/KdJHCa6S#BOXidjInqGx3fkSDkAfcuegkHt_bunqNr0y-HPFvRaQ) and save to pretrained/dinov2_vitb14_pretrain.pth.

## 🚀 Training & Testing

> **⚠️ Note on Dataset Paths:** Before running any script below, please open the script and modify the DATA_PATH variable to point to the corresponding dataset's location on your machine.

### LKGL (ConvNeXt-Base Backbone)
To train and test the standard LKGL model, run the following scripts:
* **University-1652:**
  ```
   bash vote_university.sh
  ```
* **SUES-200:**
* ```
        bash vote_sues200.sh
  ```

### LKGL-D (DINOv2-Base Backbone)
To train and test the LKGL-D variant, use the scripts below:
* **University-1652:**
* ```
        bash vote_uni_dinov2.sh
  ```
* **SUES-200:**
* ```
        bash vote_sues_dinov2.sh
  ```
* **DenseUAV:**
* ```
        bash vote_denseuav.sh
  ```
* **Multi-weather University-1652:**
* ```
        bash train_test_multi_weather_uni.sh
  ```
