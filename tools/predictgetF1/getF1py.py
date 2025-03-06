import os

from ultralytics import YOLO
import argparse
import matplotlib.pyplot as plt
import xlsxwriter
import pandas as pd
import numpy as np

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_yaml", type=str)
    parser.add_argument("--pthPath", type=str)
    parser.add_argument("--factor", type=str)
    parser.add_argument("--iou", type=float)
    parser.add_argument("--saveName", type=str)
    return parser

# def get_F1(result, factor, saveName):
#     imgF1 = result.per_image_stats
#
#     imgIDList = []
#     F1List = []
#     for key, value in imgF1.items():
#         imgID = int(key.replace(".tif", ""))
#         imgIDList.append(imgID)
#         F1List.append(value[0])
#
#     paired = list(zip(imgIDList, F1List))
#     sorted_paired = sorted(paired, key=lambda x: x[0])
#
#     sorted_list1, sorted_list2 = zip(*sorted_paired)
#
#     F1List = list(sorted_list2)
#
#     step = factor * factor
#     averages = []
#     for i in range(0, len(F1List), step):
#         chunk = F1List[i:i + step]
#         avg = sum(chunk) / len(chunk)
#         averages.append(avg)
#
#
#     workbook = xlsxwriter.Workbook(saveName)
#     worksheet = workbook.add_worksheet()
#     for row, value in enumerate(averages):
#         worksheet.write(row, 0, value)
#     workbook.close()

def print_meanF1(filePath):
    data = pd.read_excel(filePath, header=None, engine='openpyxl')
    dataNp = data.to_numpy()
    # dataNp = np.append(dataNp, 0)
    # dataNp = np.append(dataNp, 0)
    print(np.mean(dataNp))


if __name__ == "__main__":

    args = get_parser().parse_args()
    os.environ["factor"] = args.factor
    os.environ["saveName"] = args.saveName

    model = YOLO(args.pthPath)
    result = model.val(args.datasets_yaml, conf=0.001, iou=args.iou, max_det=3000, plots=False, half=False, )

    print_meanF1(args.saveName)

    # get_F1(result, args.factor, "./spotData.xlsx")

# /home/pointseg/Yolov8/tools/predictgetF1/spotdata_spl.xlsx





