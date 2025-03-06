import os

from ultralytics import YOLO
imgs_path = '/home/pointseg/datasets/cellData/images/train/99.tif'
model_path  ='/home/pointseg/Yolov8/runs/segment/train25/weights/best.pt'

import cv2
import torch
from ultralytics import YOLO
import time

# 加载模型和测试图像
model = YOLO(model_path)
image = cv2.imread('/home/pointseg/datasets/cellData/images/train/99.tif')  # 输入图像路径
image = cv2.resize(image, (786, 786))

# 预热 GPU
for _ in range(1):
    _ = model(image, augment=False)

# 测量端到端时间
num_runs = 1
start_time = time.time()
for _ in range(num_runs):
    results = model(image, augment=False)  # 包含预处理、推理、后处理
torch.cuda.synchronize()

# 计算 FPS
total_time = time.time() - start_time
fps = num_runs / total_time
print(f"端到端 FPS: {fps:.2f}")

# 输出详细时间分析
print("\n详细时间分析（单次迭代）:")
print(f"- 预处理: {results[0].speed['preprocess']:.4f} ms")
print(f"- 推理: {results[0].speed['inference']:.4f} ms")
print(f"- 后处理: {results[0].speed['postprocess']:.4f} ms")
