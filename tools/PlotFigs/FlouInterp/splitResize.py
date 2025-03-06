import os
import cv2
import numpy as np
import tifffile as tiff



def splitImg(imgPath, splitFactor):
    num_imgSpl = 0
    img = cv2.imread(imgPath, cv2.IMREAD_UNCHANGED)
    height, width = img.shape
    block_width = width // splitFactor
    block_height = height // splitFactor

    for i in range(splitFactor):  # 4行
        for j in range(splitFactor):  # 4列
            # 计算当前块的左上角坐标和右下角坐标
            left = j * block_width
            top = i * block_height
            right = (j + 1) * block_width
            bottom = (i + 1) * block_height

            if j==splitFactor-1:
                right = width
            if i==splitFactor-1:
                bottom = height

            # 切割图像块
            imgblock = img[top:bottom, left:right]

            newImgPath = os.path.join("./imgData", f'receptor_1_4_split_{num_imgSpl}.tif')

            tiff.imwrite(newImgPath, imgblock)
            num_imgSpl = num_imgSpl + 1


def composeHeatMap(imgDir,factor):

    imgList = []
    for i in range(4):
        imgPath = os.path.join(imgDir, f"receptor_1_4_{i}.png")
        img = cv2.imread(imgPath)
        imgResize = cv2.resize(img, (320, 320), interpolation=cv2.INTER_AREA)
        imgList.append(imgResize)

    upImg = np.hstack((imgList[0], imgList[1]))
    downImg = np.hstack((imgList[2], imgList[3]))

    newImg = np.vstack((upImg, downImg))
    print(newImg.shape)

    cv2.imwrite(os.path.join(imgDir, "receptor_compose.png"), newImg)



if __name__ == "__main__":

    # imgPath = "./imgData/receptor_1_4.tif"
    # splitFactor = 2
    # splitImg(imgPath, splitFactor)

    composeHeatMap("./heatmapData", 2)