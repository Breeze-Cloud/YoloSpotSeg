import argparse
import os
import shutil
from tqdm import tqdm

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataPathRoot", type=str)
    return parser


def argsAdd(args):
    args.imgTestPath = os.path.join(args.dataPathRoot, "images", "test")
    args.labelTestPath = os.path.join(args.dataPathRoot, "labels", "test")
    return args


def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)


def main(args):
    imgNameList = os.listdir(args.imgTestPath)
    imgNameList = sorted(imgNameList, key=lambda x: int(x.split('.')[0]))
    imgNameDividedList = [imgNameList[i:i + len(imgNameList) // 4] for i in range(0, len(imgNameList), len(imgNameList) // 4)]
    SNRNameList = ["SNR7", "SNR4", "SNR2", "SNR1"]
    for SNRName, SNRImgNameList in tqdm(zip(SNRNameList, imgNameDividedList)):
        SNRImgDir = os.path.join(args.dataPathRoot, SNRName, "images", "test")
        SNRLabelDir = os.path.join(args.dataPathRoot, SNRName, "labels", "test")

        checkpath(SNRImgDir)
        checkpath(SNRLabelDir)

        for imgName in  SNRImgNameList:
            imgPath = os.path.join(args.imgTestPath, imgName)
            SNRImgPath = os.path.join(SNRImgDir, imgName)
            shutil.copy(imgPath, SNRImgPath)

            labelPath = os.path.join(args.labelTestPath, imgName.replace("tif", "txt"))
            SNRLabelPath = os.path.join(SNRLabelDir, imgName.replace("tif", "txt"))
            shutil.copy(labelPath, SNRLabelPath)


if __name__ == "__main__":
    args =get_parser().parse_args()
    args= argsAdd(args)
    main(args)
