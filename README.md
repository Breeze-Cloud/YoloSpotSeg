# YoloSpotSeg

## 1. Software Introduction

This repository presents an enhanced implementation based on the YOLOv8 object detection framework. Key improvements include:
- Addition of Upsampling Layer: An additional upsampling layer is integrated into the original Yolov8 architecture to enhance segmentation performance for spot
- Image Block Partitioning and Fourier Interpolation Preprocessing: A preprocessing pipeline is proposed, which involves partitioning images into blocks and applying Fourier interpolation to optimize input data quality.
- Gaussian Diffusion Label Conversion Interface: An interface is provided to convert spot center point labels into instance segmentation labels using a Gaussian diffusion function, improving the model's ability to capture target details.



## 2. Installation Guide

### Requirements
- Python >= 3.8
- PyTorch >= 1.13.1
- CUDA >= 11.6 

### Example conda environment setup
```bash
conda create -n yolospotseg python==3.8
conda activate yolospotseg

# Clone repository with submodules
git clone --recurse-submodules https://github.com/yourusername/your-repo.git
cd your-repo

pip install torch==1.13.1+cu116 torchvision==0.14.1+cu116 torchaudio==0.13.1 --extra-index-url https://download.pytorch.org/whl/cu116
pip install -r requirements.txt

# Install custom modules
pip install -e .
```


## 3. Usage Example
Its usage is fully consistent with the Yolo model.

### Preprocess
```bash
python tools/imgSplit/imgSplit.py --pathRoot /path/to/your/dataset --outputDir /path/save/result  --splitFactor 4
```
- pathRoot: The directory path of the dataset to be preprocessed.
- outputDir: The output directory path where the preprocessed files will be saved.
- splitFactor:The factor determining how an image is divided into splitFactor × splitFactor blocks.

### Train & valid

```bash
from ultralytics import YOLO

# Load a model
model = YOLO("your_config.yaml")

# Train the model
train_results = model.train(
    data="your_dataset.yaml", 
    epochs=100, 
    imgsz=640, 
)

# Evaluate model performance on the validation set
metrics = model.val()
```

### Web-based interactive usage
For real-time model inference using pre-trained weights on image data, researchers can directly utilize our web-based analytical tool through [aclsfip](http://aclsfip.xyz/)


## 4. License
This project is built upon Ultralytics [Yolov8](https://github.com/ultralytics) which is licensed under AGPL-3.0.

### Code Derivative Statement 
This software constitutes a derivative work of the YOLOv8 framework (AGPL-3.0 licensed). All modifications and extensions are expressly released under identical open-source terms, maintaining full compliance with the original license's copyleft provisions.

### Non-commercial Nature  
This implementation contains no commercial components or monetization mechanisms. The codebase is strictly maintained for academic research and open scientific collaboration purposes.

## 5. Contact
For technical support:
- Corresponding author: Liu Huan
- Email: lh2022@stu.xjtu.edu.cn
