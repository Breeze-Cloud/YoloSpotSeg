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

    parser.add_argument("--segratio", type=float, default=0.25)
    parser.add_argument("--stepartio")


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

