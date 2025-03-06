import argparse
import os
from typing import Union, List
import numpy as np
import shutil
import tifffile as tiff
import cv2
from PIL import Image
from numpy.core.numeric import zeros_like
from tqdm import tqdm
import matplotlib.pyplot as plt

from npz2files import checkPath


# ---------------------------------------------------------
# get path of npz files
# ---------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser("get path of npz files")
    parser.add_argument("--datapath", type=str, default="./datasets/receptor")
    return parser


# ---------------------------------------------------------
# get other save path of generate files
# ---------------------------------------------------------
# - rowdata
# - normalData
#       - train
#           - images
#           - txtlabels
#           - masklabels
#       - val
#           - images
#           - txtlabels
#           - masklabels
#       - test
#           - images
#           - txtlabels
#           - masklabels
# - yoloData
#       - images
#           - train
#           - val
#           - test
#       - labels
#           - train
#           - val
#           - test
#       - masklabels
#           - train
#           - val
#           - test


def argsPlus(args):
    # nomalData Path

    args.rowPathRoot = os.path.join(args.datapath, "normalData")
    args.gradPathRoot = os.path.join(args.datapath, "gradData")
    args.unetPathRoot = os.path.join(args.datapath, "unetData")
    args.normalPathRoot = os.path.join(args.datapath, "normalData")
    args.filtPathRoot = os.path.join(args.datapath, "filtData")

    args.yoloPathRoot = os.path.join(args.datapath, "yoloData")

    checkPath(args.yoloPathRoot)

    return args

def copyfiles(pathSrc, pathDest):
    tvtList = ["train", "val", "test"]
    imgLabelList = ["images", "txtlabels", "masklabels"]

    for tvt in tqdm(tvtList):
        for imgLabel in tqdm(imgLabelList):
            filrDir = os.path.join(pathSrc, tvt, imgLabel)
            fileNameList = os.listdir(filrDir)

            if imgLabel == "txtlabels":
                imgLabel_dest = "labels"
            else:
                imgLabel_dest = imgLabel

            fileDestDir = os.path.join(pathDest, imgLabel_dest, tvt)
            checkPath(fileDestDir)

            for fileName in fileNameList:
                filePath = os.path.join(filrDir, fileName)
                fileDestPath = os.path.join(fileDestDir, fileName)

                shutil.copyfile(filePath, fileDestPath)

def main():
    args = get_parser().parse_args()
    args = argsPlus(args)

    copyfiles(args.filtPathRoot, args.yoloPathRoot)


if __name__ == "__main__":
    main()
