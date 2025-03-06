from ultralytics import YOLO
import tempfile
import cv2
import tempfile
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os

def selectWeight(pthDropdown):
    if pthDropdown == "cellData":
        pthPath = "/home/pointseg/Yolov8/runs/segment/cellData/train43/weights/last.pt"
    return pthPath





def cellSeg(imagePth, pthDropdown):
    image = cv2.imread(imagePth)

    model = YOLO(selectWeight(pthDropdown))

    predictResult = model.predict(source=imagePth, augment=False)

    imgBbox = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    bboxAnn = predictResult[0].boxes.boxes
    for bbox in bboxAnn:
        x_min, y_min, x_max, y_max, conf, cls = bbox
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        cv2.rectangle(imgBbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    imgWithBbox = cv2.addWeighted(image, 0.7, imgBbox, 0.3, 0)

    # plot mask
    imgMask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
    MaskAnn = predictResult[0].masks.segments
    for mask in MaskAnn:
        mask[:, 0] = mask[:, 0] * image.shape[1]
        mask[:, 1] = mask[:, 1] * image.shape[0]
        mask = mask.astype(np.int32)
        cv2.fillPoly(imgMask, [mask], (0, 255, 0))
        cv2.polylines(imgMask, [mask], isClosed=True, color=(0, 0, 255), thickness=2)  # 边界颜色为红色，厚度为5

    imgWithMask = cv2.addWeighted(image, 0.7, imgMask, 0.4, 0)

    #cell number
    number = len(MaskAnn)

    return imgWithBbox, imgWithMask, number

def cellLoadExample():
    img_path = "/home/pointseg/datasets/cellData/images/test/10.tif"
    weight = "cellData"
    return img_path, weight


if __name__ == "__main__":
    img_path = "/home/pointseg/datasets/cellData/images/test/60.tif"

    cellSeg(img_path, "cellData")