import os
from ultralytics import YOLO
import cv2
import tempfile
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import tifffile as tiff

def ImageInfo(imagePath):
    img = cv2.imread(imagePath, cv2.IMREAD_UNCHANGED)
    if len(img.shape)==2:
        channels = 1
        validChannels = 1
    else:
        channels = img.shape[2]
        validChannels = 0
        for i in range(channels):
            singleChannel = img[:, :, i]
            if np.any(singleChannel):
                validChannels = validChannels + 1

    dataType = str(img.dtype)
    min_value = img.min()
    max_value = img.max()
    return imagePath, channels, validChannels, dataType, min_value, max_value


def selectWeight(pthDropdown):
    if pthDropdown == "particlePth":
        pthPath = "/home/pointseg/Yolov8/runs/segment/particleSpl/train17/weights/best.pt"
    elif pthDropdown == "receptorPth":
        pthPath = "/home/pointseg/Yolov8/runs/segment/receptorSpl/train15/weights/best.pt"
    elif pthDropdown == "vesiclePth":
        pthPath = "/home/pointseg/Yolov8/runs/segment/vesicleSpl/train16//weights/best.pt"
    elif pthDropdown == "smfishPth":
        pthPath = "/home/pointseg/Yolov8/runs/segment/smfishSpl/train13/weights/best.pt"
    elif pthDropdown == "suntagPth":
        pthPath = "/home/pointseg/Yolov8/runs/segment/suntagSpl/train12/weights/best.pt"
    elif pthDropdown == "spotData":
        pthPath = "/home/pointseg/Yolov8/runs/segment/spotDataResize/train45/weights/best.pt"
    return pthPath

def polygon_area(xy):
    x = xy[:, 0]
    y = xy[:, 1]
    n = len(x)
    area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
    return area

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

def imgSplit(image, factor):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = img.shape
    block_width = width // factor
    block_height = height // factor

    for i in range(factor):
        for j in range(factor):
            # 计算当前块的左上角坐标和右下角坐标
            left = j * block_width
            top = i * block_height
            right = (j + 1) * block_width
            bottom = (i + 1) * block_height

            if j == factor - 1:
                right = width
            if i == factor - 1:
                bottom = height

            # 切割图像块
            imgblock = img[top:bottom, left:right]

            newImgBlock = fourier_interpolation(imgblock, factor)

            newImgPath = os.path.join("./temp", f'{i}{j}.tif')

            tiff.imwrite(newImgPath, newImgBlock)

def bboxMapping(bboxAnn, factor, i, j, rowImage):
    height, width = rowImage.shape[0], rowImage.shape[1],
    block_width = width // factor
    block_height = height // factor
    bboxAnnCpu = bboxAnn.cpu().numpy()
    bboxAnnCpu[:, 0] = bboxAnnCpu[:, 0] / factor + (j * block_height)
    bboxAnnCpu[:, 1] = bboxAnnCpu[:, 1] / factor + (i * block_width)
    bboxAnnCpu[:, 2] = bboxAnnCpu[:, 2] / factor + (j * block_height)
    bboxAnnCpu[:, 3] = bboxAnnCpu[:, 3] / factor + (i * block_width)
    return bboxAnnCpu

def maskMapping(maskAnn, factor, i, j, rowImage):
    height, width = rowImage.shape[0], rowImage.shape[1],
    block_width = width // factor
    block_height = height // factor

    maskNewAnn = []
    for mask in maskAnn:
        mask[:, 0::2] = mask[:, 0::2] * height  / factor + (j * block_height)
        mask[:, 1::2] = mask[:, 1::2] * width / factor + (i * block_width)
        maskNewAnn.append(mask)

    return maskNewAnn



def spotSeg(imagePath, pthDropdown, spotFactor):
    spotFactor = int(spotFactor)
    if spotFactor==1:
        image = cv2.imread(imagePath)
        # load model
        model = YOLO(selectWeight(pthDropdown))
        # predict
        predictResult = model.predict(source=imagePath, augment=False)

        # plot bbox
        imgBbox = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        bboxAnn = predictResult[0].boxes.boxes
        for bbox in bboxAnn:
            x_min, y_min, x_max, y_max, conf, cls = bbox
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

            cv2.rectangle(imgBbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        imgWithBbox = cv2.addWeighted(image, 0.6, imgBbox, 0.4, 0)

        # plot mask
        imgMask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        MaskAnn = predictResult[0].masks.segments
        for mask in MaskAnn:
            mask[:, 0] = mask[:, 0] * image.shape[1]
            mask[:, 1] = mask[:, 1] * image.shape[0]
            mask = mask.astype(np.int32)
            cv2.fillPoly(imgMask, [mask], (211, 178, 225))
            cv2.polylines(imgMask, [mask], isClosed=True, color=(255, 179, 179), thickness=2)  # 边界颜色为红色，厚度为5

        imgWithMask = cv2.addWeighted(image, 0.6, imgMask, 0.4, 0)

        # generate table
        colums = ["Order", "CenterX", "CenterY", "Area"]
        tablePd = pd.DataFrame(columns=colums)

        order = np.arange(0, len(MaskAnn)) + 1
        tablePd["Order"] = order

        x = [np.mean(mask[:, 0]) for mask in MaskAnn]
        y = [np.mean(mask[:, 1]) for mask in MaskAnn]
        tablePd["CenterX"] = x
        tablePd["CenterY"] = y

        area = [polygon_area(mask) for mask in MaskAnn]
        tablePd["Area"] = area

        # number
        num = len(MaskAnn)

        # Fluoescence intensity distribution
        average_intensities = []
        grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for mask in MaskAnn:
            mask = mask.astype(np.int32)
            imgMask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.fillPoly(imgMask, [mask], color=255)

            target_pixel = grayImg[imgMask>0]
            average_intensity = np.mean(target_pixel)
            average_intensities.append(average_intensity)

        fig, ax = plt.subplots()
        ax.hist(average_intensities, bins=50, color='#99d7a6', edgecolor='black', log=True)
        ax.set_xlabel('Fluoescence intensity')
        ax.set_ylabel('count')
        ax.set_title("Histogram of Fluoescence Intensity")
        fig.savefig("./Picture/Fluoescence.png")

        return imgWithBbox, imgWithMask, tablePd, num, fig

    else:
        image = cv2.imread(imagePath)
        imgSplit(image, spotFactor)
        model = YOLO(selectWeight(pthDropdown))
        predictResult = model.predict(source="./temp", augment=False)
        BboxAnnTotal = []
        MaskAnnTotal = []
        for blockResult in predictResult:
            imgName = os.path.basename(blockResult.path).replace('.tif', '')
            i = int(imgName[0]) # row
            j = int(imgName[1]) # colunm
            bboxAnnPred = blockResult.boxes.boxes
            bboxMapped = bboxMapping(bboxAnnPred, spotFactor, i, j, image)
            BboxAnnTotal.append(bboxMapped)

            maskAnnPred = blockResult.masks.segments
            maskMapped = maskMapping(maskAnnPred, spotFactor, i, j, image)
            MaskAnnTotal.append(maskMapped)

        # bbox
        imgBbox = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for BboxAnn in BboxAnnTotal:
            for bbox in BboxAnn:
                x_min, y_min, x_max, y_max, conf, cls = bbox
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                cv2.rectangle(imgBbox, (x_min, y_min), (x_max, y_max), (255, 103, 125), 2)

        imgWithBbox = cv2.addWeighted(image, 0.6, imgBbox, 0.4, 0)

        # imgWithBbox = Image.fromarray(imgWithBbox)
        # imgWithBbox.show()

        # mask
        imgMask = np.zeros((image.shape[0], image.shape[1], 3), dtype=np.uint8)
        for maskAnn in MaskAnnTotal:
            for mask in maskAnn:
                mask = mask.astype(np.int32)
                cv2.fillPoly(imgMask, [mask], (211, 178, 225))
                cv2.polylines(imgMask, [mask], isClosed=True, color=(255, 179, 179), thickness=2)  # 边界颜色为红色，厚度为5

        imgWithMask = cv2.addWeighted(image, 0.6, imgMask, 0.4, 0)

        # imgWithMask = Image.fromarray(imgWithMask)
        # imgWithMask.show()

        # number
        maskAnnNewFormat = []
        for maskAnn in MaskAnnTotal:
            for mask in maskAnn:
                maskAnnNewFormat.append(mask)
        num = len(maskAnnNewFormat)

        # generate table
        colums = ["Spot ID", "CenterX", "CenterY", "Area"]
        tablePd = pd.DataFrame(columns=colums)

        order = np.arange(0, num) + 1
        tablePd["Spot ID"] = order

        x = [np.mean(mask[:, 0]) for mask in maskAnnNewFormat]
        y = [np.mean(mask[:, 1]) for mask in maskAnnNewFormat]
        tablePd["CenterX"] = x
        tablePd["CenterY"] = y

        area = [polygon_area(mask) for mask in maskAnnNewFormat]
        tablePd["Area"] = area

        # Fluoescence intensity distribution
        average_intensities = []
        grayImg = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        for mask in maskAnnNewFormat:
            mask = mask.astype(np.int32)
            imgMask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cv2.fillPoly(imgMask, [mask], color=255)

            target_pixel = grayImg[imgMask > 0]
            average_intensity = np.mean(target_pixel)
            average_intensities.append(average_intensity)

        fig, ax = plt.subplots()
        ax.hist(average_intensities, bins=50, color='#99d7a6', edgecolor='black', log=True)
        ax.set_xlabel('Fluoescence Intensity')
        ax.set_ylabel('Number of Spots')
        ax.set_title("Histogram of Fluoescence Intensity")


        # plt.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)

        # plt.savefig("./Picture/Fluoescence.png", dpi=600)


        return imgWithBbox, imgWithMask, tablePd, num, fig


def spotLoadexample():
    img_path = "/home/pointseg/datasets/spotData/images/test/10.tif"
    weight = "spotData"
    factor = "4"
    return img_path, weight, factor



if __name__ == "__main__":
    img_path = "/home/pointseg/datasets/spotData/images/test/10.tif"

    spotSeg(img_path, "spotData", 4)


