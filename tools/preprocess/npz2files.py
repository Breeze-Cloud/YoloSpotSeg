import argparse
import os
from typing import Union, List
import numpy as np
import shutil
import tifffile as tiff
import cv2
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------
# get path of npz files
# ---------------------------------------------------------
def get_parser():
    parser = argparse.ArgumentParser("get path of npz files")
    parser.add_argument("--datapath", type=str, default="./datasets/receptor")
    return parser


# ---------------------------------------------------------
# ensure directory is exist
# ---------------------------------------------------------
def checkPath(path):
    if not os.path.exists(path):
        os.makedirs(path)


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
    datasetName = os.path.basename(args.datapath)
    npzPath = os.path.join(args.datapath, "rowData", f"{datasetName}.npz")
    args.npzPath = npzPath

    pathRoot = args.datapath

    args.normalPath = os.path.join(pathRoot, "normalData")
    checkPath(args.normalPath)

    args.trainPath = os.path.join(args.normalPath, "train")
    checkPath(args.trainPath)

    args.valPath = os.path.join(args.normalPath, "val")
    checkPath(args.valPath)

    args.testPath = os.path.join(args.normalPath, "test")
    checkPath(args.testPath)

    # get gauss params corresponding different dataset
    if datasetName == "particle":
        ksize = (9, 9)
        sigmaX = 2.5
    elif datasetName == "receptor":
        ksize = (5, 5)
        sigmaX = 0.9
    elif datasetName == "smfish":
        ksize = (5, 5)
        sigmaX = 0.9
    elif datasetName == "suntag":
        ksize = (5, 5)
        sigmaX = 0.9
    elif datasetName == "vesicle":
        ksize = (5, 5)
        sigmaX = 0.9
    else:
        raise KeyError("{} is not exist".format(datasetName))

    args.ksize = ksize
    args.sigmaX = sigmaX

    return args



# ---------------------------------------------------------
# load numpy data(train, valid, test) from npz
# ---------------------------------------------------------
def load_npz(
        fname: Union[str, "os.PathLike[str]"], test_only: bool = False
) -> List[np.ndarray]:
    """Imports the standard npz file format used for custom training and inference.

    Only for files saved using "np.savez_compressed(fname, x_train, y_train...)".

    Args:
        fname: Path to npz file.
        test_only: Only return testing images and labels.

    Returns:
        A list of the required numpy arrays. If no "test_only" arguments were passed,
        returns [x_train, y_train, x_valid, y_valid, x_test, y_test].

    Raises:
        ValueError: If not all datasets are found.
    """
    expected = ["x_train", "y_train", "x_valid", "y_valid", "x_test", "y_test"]
    if test_only:
        expected = expected[-2:]

    with np.load(fname, allow_pickle=True) as data:
        if not all([e in data.files for e in expected]):
            raise ValueError(f"{expected} must be present. Only found {data.files}.")
        return [data[f] for f in expected]


# ---------------------------------------------------------
# print datasets information
# ---------------------------------------------------------
def npzDatasetsInfo(path):
    """
    Args:
        path: path to npz files

    Returns:
        no, print information about npz images
    """
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_npz(path)
    img_shape = x_train.shape
    print(path)
    print("train data: N-H-W=", img_shape, "   ", "pixeltype=", x_train.dtype)
    print("   ", "pixelMin=", x_train.min(), "   pixelMax=", x_train.max())

    label_shape = y_train.shape
    print("train data label: N-H-W=", label_shape, "   ", "pixeltype=", y_train.dtype)
    # print("   ", "pixelMin=", y_train.min(), "   pixelMax=", y_train.max()) # not mask, [x, y]

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
# convert x numpy to img
# ---------------------------------------------------------
def x2img(x, pathRoot):
    """
    Args:
        x: numpy file
        pathRoot:  path to save data images, not labels
    Returns:
    """

    pathToSave = os.path.join(pathRoot, "images")
    checkPath(pathToSave)
    deleteAllFile(pathToSave)
    print(f"Now is process {pathToSave}")
    for i in tqdm(range(x.shape[0])):
        img = x[i, :, :]
        imgPath = os.path.join(pathToSave, f"{i}.tif")
        tiff.imwrite(imgPath, img)


# ---------------------------------------------------------
# convert npz to img
# ---------------------------------------------------------
def npz2image(args):
    x_train, y_train, x_val, y_val, x_test, y_test = load_npz(args.npzPath)
    x2img(x_train, args.trainPath)
    x2img(x_val, args.valPath)
    x2img(x_test, args.testPath)


# ---------------------------------------------------------
# calculate pixel axis of one point base on gaussian diffusion
# ---------------------------------------------------------
def get_gaussDiffusion_pixels(center, shape, ksize, sigmaX):
    """
    Args:
        center: x,y axis of point
        shape: h,w of image
        ksize: kernal size of gauss blur
        sigmaX: sigmaX of gauss

    Returns:
    """
    x, y = int(center[0]), int(center[1])
    h, w = shape[0], shape[1]
    # create mask for one  point
    mask = np.zeros((h, w)).astype(np.uint8)

    if x >= w:
        x = w-1
    if y >= h:
        y = h-1

    mask[x, y] = 255
    # gauss blur for mask
    blur = cv2.GaussianBlur(mask, ksize=ksize, sigmaX=sigmaX)
    _, blur = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)
    blur = blur.astype(np.uint8)
    # find outline of point
    _, one_point_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY)
    cnt, hit = cv2.findContours(one_point_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = list(cnt)
    one_point_cnt = cnt[0]
    outline = []
    for boundary_pixel in one_point_cnt:
        temp = list(boundary_pixel[0])
        outline.append(temp)
    return outline


# ---------------------------------------------------------
# convert x,y to txtlabel
# ---------------------------------------------------------
def xy2txtlabel(x, y, pathRoot, ksize, sigmaX):
    """
    Args:
        x: numpy
        y: numpy
        pathRoot:
    Returns:
    """

    pathToSave = os.path.join(pathRoot, "txtlabels")
    checkPath(pathToSave)
    deleteAllFile(pathToSave)


    img = x[0, :, :]
    print(f"Now is process {pathToSave}")
    for i in tqdm(range(y.shape[0])):
        label = y[i]
        labelPath = os.path.join(pathToSave, f"{i}.txt")
        point_num = label.shape[0]
        for j in range(point_num):
            x_axis = label[j][1]
            y_axis = label[j][0]
            # circle_points = get_circle_pixels((x_axis, y_axis), 3)
            outline_axis = get_gaussDiffusion_pixels((x_axis, y_axis), img.shape, ksize, sigmaX)
            point_class = 0

            save_str = "{}".format(point_class)

            for point in outline_axis:
                save_str = "{} {} {}".format(save_str, point[1] / img.shape[0], point[0] / img.shape[1])
            save_str = "{}\n".format(save_str)
            with open(labelPath, 'a') as file:
                file.write(save_str)

# ---------------------------------------------------------
# convert npz to txtlabels
# ---------------------------------------------------------
def npz2txtlabel(args):
    x_train, y_train, x_val, y_val, x_test, y_test = load_npz(args.npzPath)
    xy2txtlabel(x_train, y_train, args.trainPath, args.ksize, args.sigmaX)
    xy2txtlabel(x_val, y_val, args.valPath, args.ksize, args.sigmaX)
    xy2txtlabel(x_test, y_test, args.testPath, args.ksize, args.sigmaX)


# ---------------------------------------------------------
# x, y 2 masklabes
# ---------------------------------------------------------
def xy2masklabel(x, y, pathRoot, ksize, sigmaX):
    """
    Args:
        x: npz filesrhnl
        y:  npz files
        pathRoot:  save txt label
    Returns:
    """
    pathToSave = os.path.join(pathRoot, "masklabels")
    checkPath(pathToSave)
    deleteAllFile(pathToSave)

    # get gauss params corresponding different dataset
    print(f"Now is process {pathToSave}")
    img = x[0, :, :]
    h, w = img.shape[0], img.shape[1]
    for i in tqdm(range(y.shape[0])): # get label of one image
        label = y[i]
        labelPath = os.path.join(pathToSave, f"{i}.tif")
        # get mask label
        mask = np.zeros_like(img).astype(np.uint8)
        point_num = label.shape[0]
        for j in range(point_num): # get one point
            x_axis = round(label[j][0])
            y_axis = round(label[j][1])

            if x_axis >= w:
                x_axis = w - 1
            if y_axis >= h:
                y_axis = h - 1

            mask[x_axis, y_axis] = 255

        # gaussDiffusion
        blur = cv2.GaussianBlur(mask, ksize=ksize, sigmaX=sigmaX)
        _, blur = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)
        blur = blur.astype(np.uint8)
        blur[blur > 0] = 1

        tiff.imwrite(labelPath, blur)

# ---------------------------------------------------------
# convert npz to masklabels
# ---------------------------------------------------------
def npz2masklabel(args):
    x_train, y_train, x_val, y_val, x_test, y_test = load_npz(args.npzPath)
    xy2masklabel(x_train, y_train, args.trainPath, args.ksize, args.sigmaX)
    xy2masklabel(x_val, y_val, args.valPath, args.ksize, args.sigmaX)
    xy2masklabel(x_test, y_test, args.testPath, args.ksize, args.sigmaX)


# ---------------------------------------------------------
# check accuracy of txt label
# ---------------------------------------------------------
def checkLabel(args):

    pathImg = os.path.join(args.trainPath, "images", "0.tif")
    pathTxtLabel = os.path.join(args.trainPath, "txtlabels", "0.txt")

    # get image
    image = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)
    if image.dtype == np.float32:
        min = image.min()
        max = image.max()
        image = (image - min) * (255 - 1) / (max - min) + 1
        image = image.astype(np.uint8)

    if len(image.shape) == 2:
        image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=2)

    # get label
    with open(pathTxtLabel, 'r') as file:
        labels = file.read().splitlines()
    # plot
    for label in labels:
        label_parts = label.split(' ')
        points = label_parts[1:]
        for i in range(0, len(points), 2):
            point = points[i:i + 2]
            x = round(float(point[0]) * image.shape[1])
            y = round(float(point[1]) * image.shape[0])
            image[y, x] = (0, 255, 0)

    image = Image.fromarray(image)
    image.show("img and txtlabel")


# ---------------------------------------------------------
# check accuracy of txt label
# ---------------------------------------------------------
def checkLabelPath(pathImg, pathTxtLabel):

    # get image
    image = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)
    if image.dtype == np.float32:
        min = image.min()
        max = image.max()
        image = (image - min) * (255 - 1) / (max - min) + 1
        image = image.astype(np.uint8)

    if len(image.shape) == 2:
        image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=2)

    # get label
    with open(pathTxtLabel, 'r') as file:
        labels = file.read().splitlines()
    # plot
    for label in labels:
        label_parts = label.split(' ')
        points = label_parts[1:]
        for i in range(0, len(points), 2):
            point = points[i:i + 2]
            x = round(float(point[0]) * image.shape[1])
            y = round(float(point[1]) * image.shape[0])
            image[y, x] = (0, 255, 0)

    image = Image.fromarray(image)
    image.show("img and txtlabel")

# ---------------------------------------------------------
# check accuracy of mask label
# ---------------------------------------------------------
def checkImgMask(args):

    imgPath = os.path.join(args.trainPath, "images", "0.tif")
    maskPath = os.path.join(args.trainPath, "masklabels", "0.tif")

    # image
    image = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
    if image.dtype == np.float32:
        min = image.min()
        max = image.max()
        image = (image - min) * (255 - 1) / (max - min) + 1
        image = image.astype(np.uint8)

    if len(image.shape) == 2:
        image = np.repeat(np.expand_dims(image, axis=-1), 3, axis=2)

    img = Image.fromarray(image)
    img.show("image")

    # masklabel
    label = cv2.imread(maskPath, cv2.IMREAD_UNCHANGED)
    label[label>0] =255

    label = Image.fromarray(label)
    label.show("masklabels")


def main():
    args = get_parser().parse_args()
    args = argsPlus(args)

    npz2image(args)

    npz2txtlabel(args)
    checkLabel(args)

    # npz2masklabel(args)
    # checkImgMask(args)




if __name__ == "__main__":
    main()

    # newImgPath = "/home/pointseg/Yolov8/datasets/coco128-seg/images/train2017/000000000030.jpg"
    # newLabelPath = "/home/pointseg/Yolov8/datasets/coco128-seg/labels/train2017/000000000030.txt"
    # checkLabelPath(newImgPath, newLabelPath)