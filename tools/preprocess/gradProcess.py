import argparse
import os
from typing import Union, List
import numpy as np
import shutil
import tifffile as tiff
import cv2
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

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
    normalPath = os.path.join(args.datapath, "normalData")
    args.normalPath = normalPath

    args.trainPathSrc = os.path.join(args.normalPath, "train")
    args.valPathSrc = os.path.join(args.normalPath, "val")
    args.testPathSrc = os.path.join(args.normalPath, "test")

    # gradData Path
    gradPath = os.path.join(args.datapath, "gradData")
    checkPath(gradPath)
    args.gradPath = gradPath

    args.trainPathDest = os.path.join(args.gradPath, "train")
    args.valPathDest = os.path.join(args.gradPath, "val")
    args.testPathDest = os.path.join(args.gradPath, "test")

    return args


# ---------------------------------------------------------
# ensure directory is exist
# ---------------------------------------------------------
def checkPath(path):
    if not os.path.exists(path):
        os.mkdir(path)


# ---------------------------------------------------------
# delete all file and fold of one path
# ---------------------------------------------------------
def deleteAllFile(folder_path):
    """

    Args:
        folder_path: the path of need deletw all subfold and fils

    Returns:

    """
    if os.path.isdir(folder_path):
        for item in os.listdir(folder_path):
            item_path = os.path.join(folder_path, item)
            if os.path.isfile(item_path):
                os.remove(item_path)
            elif os.path.isdir(item_path):
                shutil.rmtree(item_path)
    else:
        print(f"The path '{folder_path}' is not a directory or does not exist.")


# ---------------------------------------------------------
# grad images
# ---------------------------------------------------------
def imgGradient(image):

    grad_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradxy = np.sqrt(grad_x ** 2 + grad_y ** 2)
    gradxy = gradxy.astype(image.dtype)
    return gradxy


# ---------------------------------------------------------
# conv images
# ---------------------------------------------------------
def imgdirEveryFileCov(pathSrc, pathDest):

    checkPath(pathDest)

    imgNameList = os.listdir(os.path.join(pathSrc, "images"))
    newImgDir = os.path.join(pathDest, "images")
    checkPath(newImgDir)
    deleteAllFile(newImgDir)

    print(f"Now is process {pathSrc}")
    for imgName in tqdm(imgNameList):
        # get img
        imgPath = os.path.join(os.path.join(pathSrc, "images"), imgName)
        img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)

        newImg = imgGradient(img)


        # save image
        newImgPath = os.path.join(newImgDir, imgName)
        tiff.imwrite(newImgPath, newImg)



# ---------------------------------------------------------
# conv images
# ---------------------------------------------------------
def imgConv(args):
    imgdirEveryFileCov(args.trainPathSrc, args.trainPathDest)
    imgdirEveryFileCov(args.valPathSrc, args.valPathDest)
    imgdirEveryFileCov(args.testPathSrc, args.testPathDest)


# ---------------------------------------------------------
# move txtlabel adn masklabel
# ---------------------------------------------------------
def moveTxtLabels(args):
    src = os.path.join(args.trainPathSrc, "txtlabels")
    dest = os.path.join(args.trainPathDest, "txtlabels")
    shutil.copytree(src, dest)

    src = os.path.join(args.valPathSrc, "txtlabels")
    dest = os.path.join(args.valPathDest, "txtlabels")
    shutil.copytree(src, dest)

    src = os.path.join(args.testPathSrc, "txtlabels")
    dest = os.path.join(args.testPathDest, "txtlabels")
    shutil.copytree(src, dest)



# ---------------------------------------------------------
# conv mask
# ---------------------------------------------------------
def maskdirEveryFileCov(pathSrc, pathDest):

    checkPath(pathDest)

    imgNameList = os.listdir(os.path.join(pathSrc, "masklabels"))
    newImgDir = os.path.join(pathDest, "masklabels")
    checkPath(newImgDir)
    deleteAllFile(newImgDir)

    print(f"Now is process {pathSrc}")
    for imgName in tqdm(imgNameList):
        # get img
        imgPath = os.path.join(os.path.join(pathSrc, "masklabels"), imgName)
        img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)

        newImg = np.zeros_like(img).astype(np.uint8)
        newImg[img>0] = 1

        # save image
        newImgPath = os.path.join(newImgDir, imgName)
        tiff.imwrite(newImgPath, newImg)


# ---------------------------------------------------------
# conv images
# ---------------------------------------------------------
def maskConv(args):
    maskdirEveryFileCov(args.trainPathSrc, args.trainPathDest)
    maskdirEveryFileCov(args.valPathSrc, args.valPathDest)
    maskdirEveryFileCov(args.testPathSrc, args.testPathDest)


# ---------------------------------------------------------
# check accuracy of mask label
# ---------------------------------------------------------
def checkImgMask(args):

    imgPath = os.path.join(args.trainPathDest, "images", "0.tif")
    maskPath = os.path.join(args.trainPathDest, "masklabels", "0.tif")

    # image
    image = tiff.imread(imgPath)
    plt.imshow(image)
    plt.show()

    # masklabel
    label = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)
    label[label>0] =255

    label = Image.fromarray(label)
    label.show("masklabels")


# ---------------------------------------------------------
# ensure grad is currect
# ---------------------------------------------------------
def testGrad(args):
    imgPath = os.path.join(args.trainPathSrc, "images", "0.tif")
    img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)

    newImg = imgGradient(img)

    img = Image.fromarray(img)
    img.show()

    newImg = Image.fromarray(newImg)
    newImg.show()


# ---------------------------------------------------------
# generate txt files
# ---------------------------------------------------------
def generateTxt(pathSrc):

    type = os.path.basename(pathSrc)

    imgPath = os.path.join(pathSrc, "images")
    imgNameList = os.listdir(imgPath)

    txtPath = os.path.join(pathSrc, f"{type}.txt")
    with open(txtPath, 'a') as file:
        for imgName in imgNameList:
            if imgName == imgNameList[-1]:
                line = f"images/{imgName} masklabels/{imgName}"
            else:
                line = f"images/{imgName} masklabels/{imgName}" + '\n'
            file.write(line)


def main():
    args = get_parser().parse_args()
    args = argsPlus(args)

    # imgConv(args)
    # moveTxtLabels(args)
    # maskConv(args)

    generateTxt(args.trainPathDest)
    generateTxt(args.valPathDest)
    generateTxt(args.testPathDest)


    # checkImgMask(args)

    # testGrad(args)





if __name__ == "__main__":
    main()
