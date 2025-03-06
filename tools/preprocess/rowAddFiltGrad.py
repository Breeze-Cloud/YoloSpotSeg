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


def argsPlus(args):
    # nomalData Path

    args.rowPathRoot = os.path.join(args.datapath, "normalData")
    args.gradPathRoot = os.path.join(args.datapath, "gradData")
    args.unetPathRoot = os.path.join(args.datapath, "unetData")

    args.normalPathRoot = os.path.join(args.datapath, "normalData")

    args.filtPathRoot = os.path.join(args.datapath, "filtData")
    checkPath(args.filtPathRoot)

    return args


def imgConv(imgDir, gradDir, unetDir, filtDir):
    imgNameList = os.listdir(imgDir)
    for imgName in tqdm(imgNameList):
        imgPath = os.path.join(imgDir, imgName)
        gradPath = os.path.join(gradDir, imgName)
        unetPath = os.path.join(unetDir, imgName.replace("tif", "png"))

        # get img
        img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
        dtype = img.dtype


        # get grad and filt
        grad = cv2.imread(gradPath, cv2.IMREAD_UNCHANGED)
        unet = cv2.imread(unetPath, cv2.IMREAD_UNCHANGED)

        if len(unet.shape) == 3:
            unet = unet[:, :, 0]

        filtGrad = zeros_like(grad).astype(np.float32)
        filtGrad[unet > 0] = grad[unet > 0]

        # add
        img = img.astype(np.float32)
        img = img + filtGrad
        if dtype == np.uint8:
            img[img > 255] = 255


        img = img.astype(dtype)

        # save
        filtPath = os.path.join(filtDir, imgName)
        tiff.imwrite(filtPath, img)




def convert(args):
    datatype = ["train", "val", "test"]
    for type in datatype:
        rowRoot = os.path.join(args.rowPathRoot, type, "images")
        gradRoot = os.path.join(args.gradPathRoot, type, "images")
        unetRoot = os.path.join(args.unetPathRoot, type, "images", "pseudo_color_prediction")

        filtRoot = os.path.join(args.filtPathRoot, type, "images")
        checkPath(filtRoot)

        imgConv(rowRoot, gradRoot, unetRoot, filtRoot)


def moveTxtMaskLabel(args):
    datatype = ["train", "val", "test"]
    labeltype = ["txtlabels", "masklabels"]
    for data in datatype:
        for label in labeltype:
            src = os.path.join(args.normalPathRoot, data, label)
            dest = os.path.join(args.filtPathRoot, data, label)
            shutil.copytree(src, dest)

def main():
    args = get_parser().parse_args()
    args = argsPlus(args)

    convert(args)
    moveTxtMaskLabel(args)





if __name__ == "__main__":
    main()

