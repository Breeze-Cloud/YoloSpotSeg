import os.path
import shutil

from ultralytics import YOLO
import argparse
import cv2
from PIL import Image
import numpy as np


def get_parser():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--imgPath", type=str)
    parser.add_argument("--gtpath", type=str)
    parser.add_argument("--predpath", type=str)
    parser.add_argument("--savepath", type=str)
    return parser


def maskPlot(pathImg, pathLabel, savepath, savename):
    # get image
    image = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)

    if image.max() <= 1:
        image = image * 255

    # if image.dtype == np.uint8:
    #     min = image.min()
    #     max = image.max()
    #     image = (image - min) * (255 - 1) / (max - min) + 1
    #     image = image.astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # get label
    with open(pathLabel, 'r') as file:
        labels = file.read().splitlines()
    # plot

    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    for label in labels:
        label_parts = label.split(' ')
        points = label_parts[1:]
        newPoints = []
        for i in range(0, len(points), 2):
            point = points[i:i + 2]
            x = round(float(point[0]) * image.shape[1])
            y = round(float(point[1]) * image.shape[0])
            newPoints.append(x)
            newPoints.append(y)

        newPoints = np.array(newPoints).reshape(-1, 2)
        cv2.fillPoly(mask, [newPoints], (0, 255, 0))
        cv2.polylines(mask, [newPoints], isClosed=True, color=(0, 0, 255), thickness=2)  # 边界颜色为红色，厚度为5

    image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
    image = Image.fromarray(image)
    image.save(os.path.join(savepath, savename))

def bboxPlot(pathImg, pathLabel, savepath, savename):
    # get image
    image = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)

    if image.max() <= 1:
        image = image * 255

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # get label
    with open(pathLabel, 'r') as file:
        labels = file.read().splitlines()
    # plot

    mask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)

    for label in labels:
        label_parts = label.split(' ')
        points = label_parts[1:]
        newPoints = []
        for i in range(0, len(points), 2):
            point = points[i:i + 2]
            x = round(float(point[0]) * image.shape[1])
            y = round(float(point[1]) * image.shape[0])
            newPoints.append(x)
            newPoints.append(y)

        x_coords = newPoints[::2]  # 偶数索引为x坐标
        y_coords = newPoints[1::2]  # 奇数索引为y坐标

        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)

        cv2.rectangle(mask, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    image = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
    image = Image.fromarray(image)
    image.save(os.path.join(savepath, savename))


def main(args):

    shutil.copy(args.imgPath, os.path.join(args.savepath, "image.tif"))

    maskPlot(args.imgPath, args.gtpath, args.savepath, "ground.tif")
    maskPlot(args.imgPath, args.predpath, args.savepath, "maskpred.tif")
    bboxPlot(args.imgPath, args.predpath, args.savepath, "bboxpred.tif")




if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)

    # imgPath = "/home/pointseg/datasets/deepblink/receptorSpl/images/test/0.tif"
    # labelPath = "/home/pointseg/datasets/deepblink/receptorSpl/labels/test/0.txt"
    # bboxPlot(imgPath, labelPath, "123")