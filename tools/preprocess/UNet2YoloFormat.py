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



def argsPlus(args):

    args.filtPathRoot = os.path.join(args.datapath, "filtData")

    args.yoloPathRoot = os.path.join(args.datapath, "yoloData")
    checkPath(args.filtPathRoot)

    return args

