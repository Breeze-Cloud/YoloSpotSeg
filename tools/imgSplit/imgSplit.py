import os
import argparse

import cv2
import numpy as np
from PIL import Image
import tifffile as tiff
from tqdm import tqdm

def checkpath(path):
    if not os.path.exists(path):
        os.makedirs(path)



def get_parser():
    parser = argparse.ArgumentParser("get root path")
    parser.add_argument("--pathRoot", type=str)
    parser.add_argument("--outputDir", type=str)
    parser.add_argument("--splitFactor", type=int)
    return parser


def argsAdd(args):
    pathRoot = args.pathRoot

    args.pathImg = os.path.join(pathRoot, "images")
    args.pathLabel = os.path.join(pathRoot, "labels")

    # datasetName = os.path.basename(pathRoot)
    # pathSaveRoot = os.path.join(os.path.dirname(pathRoot), f"{datasetName}Spl")

    datasetName = os.path.basename(pathRoot)
    pathSaveRoot = os.path.join(args.outputDir, f"{datasetName}Spl{args.splitFactor}")

    args.pathImgSpl = os.path.join(pathSaveRoot, "images")
    args.pathLabelSpl = os.path.join(pathSaveRoot, "labels")

    checkpath(args.pathImgSpl)
    checkpath(args.pathLabelSpl)

    return args


def parse_yolo_seg_line(line):
    parts = line.split()
    class_id = int(parts[0])
    polygon_points = [float(p) for p in parts[1:]]
    return class_id, polygon_points


def create_mask_from_polygons(image_shape, polygons):
    mask = np.zeros(image_shape[:2], dtype=np.uint8)
    points = np.array(polygons).reshape((-1, 2))
    points = (points * image_shape[1::-1]).astype(int)  # Scale to image dimensions
    cv2.fillPoly(mask, [points], color=255)
    return mask


def read_yolo_seg_labels(label_path, image_shape):
    masks = []
    with open(label_path, 'r') as file:
        for line in file:
            class_id, polygon_points = parse_yolo_seg_line(line)
            mask = create_mask_from_polygons(image_shape, polygon_points)
            masks.append(mask)
    return masks


def showImgMask(img, mask):
    img = Image.fromarray(img)
    img.show("img")

    new_mask = np.zeros_like(mask, dtype=np.uint8)
    new_mask[mask > 0] = 255
    new_mask = Image.fromarray(new_mask)
    new_mask.show("mask")

def fourier_interpolation(image, scale_factor):
    # 获取原始图像的尺寸
    h, w = image.shape[:2]

    # 将图像转换为灰度图像（如果是彩色图像）
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 转换到频域
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)  # 将低频分量移动到中心

    # 获取扩展后的尺寸
    new_h, new_w = int(h * scale_factor), int(w * scale_factor)

    # 创建零填充的频域图像
    fshift_padded = np.zeros((new_h, new_w), dtype=np.complex64)
    fshift_padded[:h, :w] = fshift

    # 将频域图像逆中心化
    f_ishift_padded = np.fft.ifftshift(fshift_padded)

    # 转换回空间域
    image_interpolated = np.fft.ifft2(f_ishift_padded)
    image_interpolated = np.abs(image_interpolated)  # 取绝对值

    # 将结果转换为无符号8位整数
    image_interpolated = (image_interpolated / np.max(image_interpolated) * 255).astype(np.uint8)

    return image_interpolated

def get_outlin_one_point(mask_one_point):
    _, one_point_mask = cv2.threshold(mask_one_point, 0, 255, cv2.THRESH_BINARY)
    cnt, hit = cv2.findContours(one_point_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = list(cnt)
    one_point_cnt = cnt[0]
    outline = []
    for boundary_pixel in one_point_cnt:
        temp = list(boundary_pixel[0])
        outline.append(temp)
    return outline


def saveMask2Txt(masks, img, saveFilePath):

    with open(saveFilePath, 'w') as file:
        for mask in masks:
            if not np.all(mask == 0):
                outline_axis = get_outlin_one_point(mask)
                point_class = 0

                save_str = "{}".format(point_class)

                for point in outline_axis:
                    save_str = "{} {} {}".format(save_str, point[0] / img.shape[1], point[1] / img.shape[0])
                save_str = "{}\n".format(save_str)
                file.write(save_str)


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



def display_segmentation(image_path, label_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("Image not found")
        return

    # 读取标签文件
    with open(label_path, 'r') as file:
        lines = file.readlines()

    # 遍历每个对象的标签
    for line in lines:
        # 分割标签和坐标
        parts = line.strip().split()
        class_index = int(parts[0])
        # 读取坐标点，每两个值代表一个点
        points = np.array([list(map(float, part.split(','))) for part in parts[1:]], dtype=np.float32)
        points = points * image.shape[0]

        # 将坐标转换为整数，并reshape成(-1, 1, 2)的形状
        points = points.astype(np.int32).reshape((-1, 1, 2))

        # 绘制轮廓，这里使用红色，线条宽度为2
        cv2.polylines(image, [points], isClosed=True, color=(0, 0, 255), thickness=2)

    # 显示图像
    image = Image.fromarray(image)
    image.show('Image with Segmentation')


# 使用函数
# display_segmentation('path_to_image.jpg', 'path_to_label.txt')




def main(args):
    datatype = ["train", "valid", "test"]
    for type in datatype:
        pathImg = os.path.join(args.pathImg, type)
        pathLabel = os.path.join(args.pathLabel, type)

        imgNameList = os.listdir(pathImg)
        num_imgSpl = 0
        for imgName in tqdm(imgNameList):
            imgPath = os.path.join(pathImg, imgName)
            txtPath = os.path.join(pathLabel, imgName.replace("tif", "txt"))

            img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
            masks = read_yolo_seg_labels(txtPath, img.shape)

            height, width = img.shape
            block_width = width // args.splitFactor
            block_height = height // args.splitFactor

            for i in range(args.splitFactor):  # 4行
                for j in range(args.splitFactor):  # 4列
                    # 计算当前块的左上角坐标和右下角坐标
                    left = j * block_width
                    top = i * block_height
                    right = (j + 1) * block_width
                    bottom = (i + 1) * block_height

                    if j==args.splitFactor-1:
                        right = width
                    if i==args.splitFactor-1:
                        bottom = height

                    # 切割图像块
                    imgblock = img[top:bottom, left:right]
                    maskblocks = [mask[top:bottom, left:right] for mask in masks]

                    newImgBlock = fourier_interpolation(imgblock, args.splitFactor)
                    newMaskBlocks =  [cv2.resize(maskblock, (width, height), interpolation=cv2.INTER_NEAREST) for maskblock in maskblocks]

                    checkpath(os.path.join(args.pathImgSpl, type))
                    checkpath(os.path.join(args.pathLabelSpl, type))

                    newImgPath = os.path.join(args.pathImgSpl, type, f'{num_imgSpl}.tif')
                    newLabelPath = os.path.join(args.pathLabelSpl, type, f'{num_imgSpl}.txt')

                    tiff.imwrite(newImgPath, newImgBlock)
                    saveMask2Txt(newMaskBlocks, newImgBlock, newLabelPath)

                    # checkLabel(newImgPath, newLabelPath)
                    # display_segmentation(newImgPath, newLabelPath)

                    num_imgSpl = num_imgSpl + 1



if __name__ == "__main__":
    args = get_parser().parse_args()
    args = argsAdd(args)
    main(args)

    # newImgPath = "/home/pointseg/Yolov8/datasets/coco128-seg/images/train2017/000000000030.jpg"
    # newLabelPath =  "/home/pointseg/Yolov8/datasets/coco128-seg/labels/train2017/000000000030.txt"

    # newImgPath = "/home/pointseg/datasets/deepblink/suntagSpl/images/train/136.tif"
    # newLabelPath =  "/home/pointseg/datasets/deepblink/suntagSpl/labels/train/136.txt"
    #
    # checkLabel(newImgPath, newLabelPath)
    # display_segmentation(newImgPath, newLabelPath)


