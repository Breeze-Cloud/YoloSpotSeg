import os
import cv2
import numpy as np
from typing import Union, List
from tqdm import tqdm
import tifffile as tiff
import shutil
from PIL import Image
from skimage.morphology import remove_small_objects

from ultralytics.yolo.data.augment import GradientSharp
import argparse
import xlsxwriter

def get_parser():
    parser = argparse.ArgumentParser("get root path")
    parser.add_argument("--npzPath", type=str, default="/home/pointseg/datasets/deepblink/row_npzfiles/suntag.npz")
    parser.add_argument("--outputDir", type=str, default="/home/pointseg/datasets/deepblink-reproduce/suntag")
    parser.add_argument("--ksize", type=int, default=5)
    parser.add_argument("--sigmax", type=float, default=0.5)
    return parser

def args_add(args):
    args.imgSavePath = os.path.join(args.outputDir, "images")
    args.labelSavePath = os.path.join(args.outputDir, "labels")
    return args



# load numpy data(train, valid, test) from npz
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

# print datasets information
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

# if path is not exist, make it
def check_path(dir_path):
    """
    Args:
        dir_path: path need to check if exists
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# delete all file and fold of one path
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

# convert npz to img
def npz2img(x, pathToSave):
    """

    Args:
        x: npz file
        pathToSave:  path to save data images, not labels
    Returns:

    """
    check_path(pathToSave)
    deleteAllFile(pathToSave)

    datasetsName =  pathToSave.split('/')[-3]


    for i in tqdm(range(x.shape[0])):
        img = x[i, :, :]
        if datasetsName == "smfish":
            img = img.astype(np.uint16)
        elif datasetsName == "suntag":
            img = img.astype(np.uint16)

        imgPath = os.path.join(pathToSave, f"{i}.tif")
        cv2.imwrite(imgPath, img)

# calculate pixel axis of one point base on gaussian diffusion
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

# convert npz to txtlabel
def npz2txtlabel(x, y, pathToSave, ksize, sigmax):
    """

    Args:
        x: npz filesrhnl
        y:  npz files
        pathToSave:  save txt label
    Returns:

    """

    check_path(pathToSave)
    deleteAllFile(pathToSave)


    kernal = (ksize, ksize)
    sigmaX = sigmax


    img = x[0, :, :]

    for i in tqdm(range(y.shape[0])):
        if i==9:
            a = 1
        label = y[i]
        if len(label) == 0:
            # imgPath = os.path.join(pathToSave.replace("labels", "images"), f"{i}.tif")
            # if os.path.exists(imgPath):
            #     os.remove(imgPath)
            labelPath = os.path.join(pathToSave, f"{i}.txt")
            with open(labelPath, "w") as f:
                pass
            continue



        labelPath = os.path.join(pathToSave, f"{i}.txt")

        point_num = label.shape[0]
        for j in range(point_num):
            x_axis = label[j][1]
            y_axis = label[j][0]
            # circle_points = get_circle_pixels((x_axis, y_axis), 3)
            outline_axis = get_gaussDiffusion_pixels((x_axis, y_axis), img.shape, kernal, sigmaX)
            point_class = 0

            save_str = "{}".format(point_class)

            for point in outline_axis:
                save_str = "{} {} {}".format(save_str, point[1] / img.shape[0], point[0] / img.shape[1])
            save_str = "{}\n".format(save_str)

            with open(labelPath, 'a') as file:
                file.write(save_str)

def npz2xylabel(x, y, pathToSave):

    check_path(pathToSave)
    deleteAllFile(pathToSave)


    img = x[0, :, :]

    for i in tqdm(range(y.shape[0])):
        if i==9:
            a = 1
        label = y[i]
        if len(label) == 0:
            imgPath = os.path.join(pathToSave.replace("labels", "images"), f"{i}.tif")
            if os.path.exists(imgPath):
                os.remove(imgPath)

        labelPath = os.path.join(pathToSave, f"{i}.xlsx")

        workbook = xlsxwriter.Workbook(labelPath)
        worksheet = workbook.add_worksheet()
        for row_num, row_data in enumerate(label):
            worksheet.write_row(row_num, 0, row_data)
        workbook.close()

# group npz2img and  npz2label
def npz2img_label(pathSrc, pathImg, pathLabel, ksize, sigmax):
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_npz(pathSrc)

    npz2img(x_train, os.path.join(pathImg, "train"))
    npz2img(x_valid, os.path.join(pathImg, "valid"))
    npz2img(x_test, os.path.join(pathImg, "test"))

    npz2txtlabel(x_train, y_train, os.path.join(pathLabel, "train"),  ksize, sigmax)
    npz2txtlabel(x_valid, y_valid, os.path.join(pathLabel, "valid"),  ksize, sigmax)
    npz2txtlabel(x_test, y_test, os.path.join(pathLabel, "test"),  ksize, sigmax)

    # npz2xylabel(x_train, y_train, os.path.join(pathLabel, "train"))
    # npz2xylabel(x_valid, y_valid, os.path.join(pathLabel, "valid"))
    # npz2xylabel(x_test, y_test, os.path.join(pathLabel, "test"))

# check accuracy of txt label
def checkLabel(pathImg, pathLabel):
    # get image
    image = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)
    if image.dtype == np.float32:
        min = image.min()
        max = image.max()
        image = (image - min) * (255 - 1) / (max - min) + 1
        image = image.astype(np.uint8)

    # get label
    with open(pathLabel, 'r') as file:
        labels = file.read().splitlines()
    # plot
    for label in labels:
        label_parts = label.split(' ')
        points = label_parts[1:]
        for i in range(0, len(points), 2):
            point = points[i:i + 2]
            x = round(float(point[0]) * image.shape[1])
            y = round(float(point[1]) * image.shape[0])
            image[y, x] = 255

    image = Image.fromarray(image)
    image.show("img")

def npz2masklabel(x, y, pathToSave):
    """

    Args:
        x: npz filesrhnl
        y:  npz files
        pathToSave:  save txt label
    Returns:

    """

    check_path(pathToSave)
    deleteAllFile(pathToSave)

    # get gauss params corresponding different dataset
    pathSplit = pathToSave.split('/')
    datasetName = pathSplit[-3]
    if datasetName == "particle_gradient":
        kernal = (9, 9)
        sigmaX = 2.5
    elif datasetName == "receptor_gradient":
        kernal = (5, 5)
        sigmaX = 0.9
    elif datasetName == "smfish":
        kernal = (5, 5)
        sigmaX = 0.9
    elif datasetName == "suntag":
        kernal = (5, 5)
        sigmaX = 0.9
    elif datasetName == "vesicle_gradient":
        kernal = (3, 3)
        sigmaX = 0.5
    else:
        raise KeyError("{} is not exist".format(datasetName))

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
        blur = cv2.GaussianBlur(mask, ksize=kernal, sigmaX=sigmaX)
        _, blur = cv2.threshold(blur, 1, 255, cv2.THRESH_BINARY)
        blur = blur.astype(np.uint8)
        blur[blur > 0] = 1

        tiff.imwrite(labelPath, blur)

def npz2saveMaskLabel(pathSrc, pathLabel):
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_npz(pathSrc)

    check_path(pathLabel)

    npz2masklabel(x_train, y_train, os.path.join(pathLabel, "train"))
    npz2masklabel(x_valid, y_valid, os.path.join(pathLabel, "valid"))
    npz2masklabel(x_test, y_test, os.path.join(pathLabel, "test"))

def checkLabel(pathImg, pathTxtLabel):

    # get image
    image = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)
    if image.dtype == np.float32:
        min = image.min()
        max = image.max()
        image = (image - min) * (255 - 1) / (max - min) + 1
        image = image.astype(np.uint8)

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
            if image.dtype == np.uint16:
                image[y, x] = 65535
            else:
                image[y, x] = 255

    image = Image.fromarray(image)
    image.show("img")


def main(args):
    npzDatasetsInfo(args.npzPath)
    npz2img_label(args.npzPath, args.imgSavePath, args.labelSavePath, args.ksize, args.sigmax)





if __name__ == "__main__":
    args = get_parser().parse_args()
    args = args_add(args)
    main(args)





    # path to row datasets
    # pathNpzRoot = "/home/pointseg/datasets/deepblink"
    # particle_npz_path = os.path.join(pathNpzRoot, "row_npzfiles", "particle.npz")
    # receptor_npz_path = os.path.join(pathNpzRoot, "row_npzfiles", "receptor.npz")
    # smfish_npz_path = os.path.join(pathNpzRoot, "row_npzfiles", "smfish.npz")
    # suntag_npz_path = os.path.join(pathNpzRoot, "row_npzfiles", "suntag.npz")
    # vesicle_npz_path = os.path.join(pathNpzRoot, "row_npzfiles", "vesicle.npz")

    # path to save images and labels
    # pathSaveRoot = "/home/pointseg/datasets/deepblink"
    #
    # particle_img_savePath = os.path.join(pathSaveRoot, "particle", "images")
    # particle_label_savePath = os.path.join(pathSaveRoot, "particle", "labels")
    #
    # receptor_img_savePath = os.path.join(pathSaveRoot, "receptor", "images")
    # receptor_label_savePath = os.path.join(pathSaveRoot, "receptor", "labels")
    #
    # smfish_img_savePath = os.path.join(pathSaveRoot, "smfish", "images")
    # smfish_label_savePath = os.path.join(pathSaveRoot, "smfish", "labels")
    #
    # suntag_img_savePath = os.path.join(pathSaveRoot, "suntag", "images")
    # suntag_label_savePath = os.path.join(pathSaveRoot, "suntag", "labels")
    #
    # vesicle_img_savePath = os.path.join(pathSaveRoot, "vesicle", "images")
    # vesicle_label_savePath = os.path.join(pathSaveRoot, "vesicle", "labels")

    # print information about datasets
    # npzDatasetsInfo(particle_npz_path)
    # npzDatasetsInfo(receptor_npz_path)
    # npzDatasetsInfo(smfish_npz_path)
    # npzDatasetsInfo(suntag_npz_path)
    # npzDatasetsInfo(vesicle_npz_path)

    # convert npz to images and labesl
    # npz2img_label(particle_npz_path, particle_img_savePath, particle_label_savePath)
    # npz2img_label(receptor_npz_path, receptor_img_savePath, receptor_label_savePath)
    # npz2img_label(smfish_npz_path, smfish_img_savePath, smfish_label_savePath)
    # npz2img_label(suntag_npz_path, suntag_img_savePath, suntag_label_savePath)
    # npz2img_label(vesicle_npz_path, vesicle_img_savePath, vesicle_label_savePath)

    # check txt label
    # pathImg = os.path.join(suntag_img_savePath, "train", "100.tif")
    # pathLabel = os.path.join(suntag_label_savePath, "train", "100.txt")
    # checkLabel(pathImg, pathLabel)


    # check
    # newImgPath = "/home/pointseg/Yolov8/datasets/coco128-seg/images/train2017/000000000049.jpg"
    # newLabelPath =  "/home/pointseg/Yolov8/datasets/coco128-seg/labels/train2017/000000000049.txt"

    # newImgPath = "/home/pointseg/datasets/deepblink/suntag/images/train/10.tif"
    # newLabelPath =  "/home/pointseg/datasets/deepblink/suntag/labels/train/10.txt"
    #
    # checkLabel(newImgPath, newLabelPath)
