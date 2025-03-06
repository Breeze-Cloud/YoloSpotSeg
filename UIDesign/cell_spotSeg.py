from ultralytics import YOLO
import tempfile
import cv2
import tempfile
import numpy as np
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import os
from spotSeg import imgSplit, bboxMapping, maskMapping
import seaborn as sns
from spotSeg import polygon_area
from scipy.interpolate import make_interp_spline
import xlsxwriter

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
    if pthDropdown == "cellData":
        pthPath = "/home/pointseg/Yolov8/runs/segment/cellData/train43/weights/last.pt"
    return pthPath


def getSpotNumInCell(oneCellAnn, xMeanSpot, yMeanSpot, image):
    cellImgMask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    oneCellAnn = oneCellAnn.astype(np.int32)
    cv2.fillPoly(cellImgMask, [oneCellAnn], color=255)
    num = 0
    for xMean, yMean in zip(xMeanSpot, yMeanSpot):
        xMean = int(xMean)
        yMean = int(yMean)
        if cellImgMask[yMean, xMean] > 0:
            num = num + 1
    return num

def getSpotAccumulatedFIInCell(oneCellAnn, xMeanSpot, yMeanSpot, image, spotAverageIntensities):
    cellImgMask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    oneCellAnn = oneCellAnn.astype(np.int32)
    cv2.fillPoly(cellImgMask, [oneCellAnn], color=255)
    intensity = 0
    for xMean, yMean, spotIntens in zip(xMeanSpot, yMeanSpot, spotAverageIntensities):
        xMean = int(xMean)
        yMean = int(yMean)
        if cellImgMask[yMean, xMean] > 0:
            intensity = intensity + spotIntens
    return intensity

def getSpotAreaofOneCell(oneCellAnn, xMeanSpot, yMeanSpot, image, spotAnnTotal):
    cellImgMask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
    oneCellAnn = oneCellAnn.astype(np.int32)
    cv2.fillPoly(cellImgMask, [oneCellAnn], color=255)
    spotAreaAccumOfOneCell = 0
    for xMean, yMean, spotAnn in zip(xMeanSpot, yMeanSpot, spotAnnTotal):
        xMean = int(xMean)
        yMean = int(yMean)
        if cellImgMask[yMean, xMean] > 0:
            spotAreaAccumOfOneCell = spotAreaAccumOfOneCell + polygon_area(spotAnn)
    return spotAreaAccumOfOneCell


def cellSpotSeg(spotImagePath, spotPth, spotFactor, cellImagePath, cellPth):
    # -------------------------------------------- spot --------------------------------------------
    spotFactor = int(spotFactor)
    if spotFactor == 1:
        spotImage = cv2.imread(spotImagePath)
        spotModel = YOLO(selectWeight(spotPth))
        spotPredResult = spotModel.predict(source=spotImagePath, augment=False)

        # plot bbox
        spotImgBbox = np.zeros((spotImage.shape[0], spotImage.shape[1], 3), dtype=np.uint8)
        spotBboxAnn = spotPredResult[0].boxes.boxes
        for bbox in spotBboxAnn:
            x_min, y_min, x_max, y_max, conf, cls = bbox
            x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

            cv2.rectangle(spotImgBbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        spotImgWithBbox = cv2.addWeighted(spotImage, 0.7, spotImgBbox, 0.3, 0)

        # plot mask
        spotImgMask = np.zeros((spotImage.shape[0], spotImage.shape[1], 3), dtype=np.uint8)
        spotMaskAnn = spotPredResult[0].masks.segments
        for mask in spotMaskAnn:
            mask[:, 0] = mask[:, 0] * spotImage.shape[1]
            mask[:, 1] = mask[:, 1] * spotImage.shape[0]
            mask = mask.astype(np.int32)
            cv2.fillPoly(spotImgMask, [mask], (0, 255, 0))
            cv2.polylines(spotImgMask, [mask], isClosed=True, color=(0, 0, 255), thickness=2)  # 边界颜色为红色，厚度为5

        spotImgWithMask = cv2.addWeighted(spotImage, 0.7, spotImgMask, 0.3, 0)

        spotNum = len(spotMaskAnn)
    else:
        spotImage = cv2.imread(spotImagePath)
        imgSplit(spotImage, spotFactor)
        spotModel = YOLO(selectWeight(spotPth))
        spotPredResult = spotModel.predict(source="./temp", augment=False)
        BboxAnnTotal = []
        MaskAnnTotal = []
        for blockResult in spotPredResult:
            imgName = os.path.basename(blockResult.path).replace('.tif', '')
            i = int(imgName[0])  # row
            j = int(imgName[1])  # colunm
            bboxAnnPred = blockResult.boxes.boxes
            bboxMapped = bboxMapping(bboxAnnPred, spotFactor, i, j, spotImage)
            BboxAnnTotal.append(bboxMapped)

            maskAnnPred = blockResult.masks.segments
            maskMapped = maskMapping(maskAnnPred, spotFactor, i, j, spotImage)
            MaskAnnTotal.append(maskMapped)

        # bbox
        imgBbox = np.zeros((spotImage.shape[0], spotImage.shape[1], 3), dtype=np.uint8)
        for BboxAnn in BboxAnnTotal:
            for bbox in BboxAnn:
                x_min, y_min, x_max, y_max, conf, cls = bbox
                x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)
                cv2.rectangle(imgBbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

        spotImgWithBbox = cv2.addWeighted(spotImage, 0.7, imgBbox, 0.3, 0)

        # mask
        imgMask = np.zeros((spotImage.shape[0], spotImage.shape[1], 3), dtype=np.uint8)
        for maskAnn in MaskAnnTotal:
            for mask in maskAnn:
                mask = mask.astype(np.int32)
                cv2.fillPoly(imgMask, [mask], (0, 255, 0))
                cv2.polylines(imgMask, [mask], isClosed=True, color=(0, 0, 255), thickness=2)  # 边界颜色为红色，厚度为5

        spotImgWithMask = cv2.addWeighted(spotImage, 0.7, imgMask, 0.3, 0)

        spotMaskAnn = []
        for maskAnn in MaskAnnTotal:
            for mask in maskAnn:
                spotMaskAnn.append(mask)
        spotNum = len(spotMaskAnn)


    # -------------------------------------------- cell --------------------------------------------
    cellImage = cv2.imread(cellImagePath)
    cellModel = YOLO(selectWeight(cellPth))
    cellPredResult = cellModel.predict(source=cellImagePath, augment=False)

    # plot bbox
    cellImgBbox = np.zeros((cellImage.shape[0], cellImage.shape[1], 3), dtype=np.uint8)
    cellBboxAnn = cellPredResult[0].boxes.boxes
    for bbox in cellBboxAnn:
        x_min, y_min, x_max, y_max, conf, cls = bbox
        x_min, y_min, x_max, y_max = int(x_min), int(y_min), int(x_max), int(y_max)

        cv2.rectangle(cellImgBbox, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    cellImgWithBbox = cv2.addWeighted(cellImage, 0.7, cellImgBbox, 0.3, 0)

    # plot mask
    cellImgMask = np.zeros((cellImage.shape[0], cellImage.shape[1], 3), dtype=np.uint8)
    cellMaskAnn = cellPredResult[0].masks.segments
    for mask in cellMaskAnn:
        mask[:, 0] = mask[:, 0] * cellImage.shape[1]
        mask[:, 1] = mask[:, 1] * cellImage.shape[0]
        mask = mask.astype(np.int32)
        cv2.fillPoly(cellImgMask, [mask], (0, 255, 0))
        cv2.polylines(cellImgMask, [mask], isClosed=True, color=(0, 0, 255), thickness=2)  # 边界颜色为红色，厚度为5

    cellImgWithMask = cv2.addWeighted(cellImage, 0.7, cellImgMask, 0.3, 0)

    cellNum = len(cellMaskAnn)

    # -------------------------------------------- spot fluorescence intensity --------------------------------------------

    # Fluoescence intensity distribution
    spotAverageIntensities = []
    spotGrayImg = cv2.cvtColor(spotImage, cv2.COLOR_BGR2GRAY)
    for mask in spotMaskAnn:
        mask = mask.astype(np.int32)
        spotImgMask = np.zeros((spotGrayImg.shape[0], spotGrayImg.shape[1]), dtype=np.uint8)
        cv2.fillPoly(spotImgMask, [mask], color=255)

        target_pixel = spotGrayImg[spotImgMask > 0]
        average_intensity = np.mean(target_pixel)
        spotAverageIntensities.append(average_intensity)


    # -------------------------------------------- counting --------------------------------------------

    csColums = ["Cell ID", "SpotNum", "Accumulated fluorescence intensity", "mean fluorescence intensity", "cellArea", "density"]
    csTable = pd.DataFrame(columns=csColums)

    order = np.arange(0, cellNum) + 1
    csTable["Cell ID"] = order

    xMeanSpot = [np.mean(mask[:, 0]) for mask in spotMaskAnn]
    yMeanSpot = [np.mean(mask[:, 1]) for mask in spotMaskAnn]

    spotOfCellNum = [getSpotNumInCell(mask, xMeanSpot, yMeanSpot, cellImage) for mask in cellMaskAnn]
    csTable["SpotNum"] = spotOfCellNum

    cellAccumulatedFI = [getSpotAccumulatedFIInCell(mask, xMeanSpot, yMeanSpot, cellImage, spotAverageIntensities) for mask in cellMaskAnn]
    csTable["Accumulated fluorescence intensity"] = cellAccumulatedFI

    temp_result = []
    for c, s in zip(cellAccumulatedFI,spotOfCellNum):
        if s == 0:
            temp_result.append(None)
        else:
            temp_result.append(c/s)

    csTable["mean fluorescence intensity"] = np.array(temp_result)

    cellArea = [polygon_area(mask) for mask in cellMaskAnn]
    spotAccumAreaOfOneCell = [getSpotAreaofOneCell(mask, xMeanSpot, yMeanSpot, cellImage, spotMaskAnn) for mask in cellMaskAnn]
    density = np.array(spotAccumAreaOfOneCell)/np.array(cellArea)

    csTable["cellArea"] = cellArea
    csTable["density"] = density


    # -------------------------------------------- xiao ti qin --------------------------------------------
    # 设置 Nature 期刊风格参数
    plt.rcParams.update({
        'font.family': 'Arial',
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 11,
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'axes.linewidth': 0.8,
        'lines.linewidth': 1.2,
        'patch.linewidth': 0.8,
        'xtick.major.width': 0.8,
        'ytick.major.width': 0.8,
        'xtick.direction': 'out',
        'ytick.direction': 'out',
        'figure.dpi': 300,
        'savefig.dpi': 300,
        'legend.fontsize': 9,
        'legend.frameon': False
    })

    spotOfCellNum = np.array(spotOfCellNum)
    y_spotOfCellNumNo0 = spotOfCellNum[spotOfCellNum != 0]
    x_value = []
    for y in y_spotOfCellNumNo0:
        x_value.append("way")

    # 创建画布
    fig, ax = plt.subplots()  # Nature 单栏标准尺寸
    violin = sns.violinplot(
        x=x_value,
        y=y_spotOfCellNumNo0,
        palette=["#4E79A7"],  # Nature 经典配色
        linewidth=0.8,
        width=0.8,
        inner="box",  # 内部显示箱线图
        saturation=0.9,
        ax=ax,
        cut=0
    )


    spotOfCellNum = np.array(spotOfCellNum)
    y_spotOfCellNumNo0 = spotOfCellNum[spotOfCellNum!=0]

    # workbook = xlsxwriter.Workbook("./Table/spotnum.xlsx")
    # worksheet = workbook.add_worksheet()
    # for row, value in enumerate(y_spotOfCellNumNo0):
    #     worksheet.write(row, 0, value)
    # workbook.close()

    x_value = []
    for y in y_spotOfCellNumNo0:
        x_value.append("way1")

    # --- 坐标轴优化 ---
    ax.set_ylabel("Spot counts/cell", labelpad=8)

    # 刻度线样式
    ax.tick_params(axis='both', which='major', pad=4)
    plt.xticks(rotation=0)  # 保持水平标签

    # 科学刻度格式
    from matplotlib.ticker import FormatStrFormatter
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.0f'))  # 整数刻度

    # 清理边框
    sns.despine(offset=5, trim=True)

    # -------------------------------------------- density --------------------------------------------


    spotOfCellNum = np.array(spotOfCellNum)

    # spotNumDensity = [spotOfCellNum, density]
    #
    # workbook = xlsxwriter.Workbook("./Table/density.xlsx")
    # worksheet = workbook.add_worksheet()
    # for row_index, row in enumerate(spotNumDensity):
    #     for col_index, value in enumerate(row):
    #         worksheet.write(row_index, col_index, value)
    # workbook.close()


    # bar
    sorted_indices = np.argsort(spotOfCellNum)
    spotOfCellNum = spotOfCellNum[sorted_indices]
    density = density[sorted_indices]

    # nihe
    fit_degree = 2
    # 使用二次多项式拟合
    coefficients = np.polyfit(spotOfCellNum, density, fit_degree)
    trend_curve = np.polyval(coefficients, spotOfCellNum)

    fig1, ax1 = plt.subplots()  # Nature 单栏标准尺寸
    ax1.bar(spotOfCellNum, density, color='skyblue')

    # 使用插值方法生成平滑曲线

    ax1.plot(
        spotOfCellNum,
        trend_curve,
        color='tomato',
        marker='o',
        linewidth=2,
        markersize=8,
    )

    # ax1.tick_params(axis='both', which='both', labelbottom=False, labelleft=False)

    ax1.set_xlabel('Spot counts/cell')
    ax1.set_ylabel('Density(%)')
    ax1.legend()



    # fig1.savefig("./Picture/density.png", dpi=600)


    return spotImgWithBbox, spotImgWithMask, cellImgWithBbox, cellImgWithMask, spotNum, cellNum, csTable, fig


def cellSpotLoadExample():
    spotImagePath = "/home/pointseg/datasets/spotData/images/test/10.tif"
    cellImagePath = "/home/pointseg/datasets/cellData/images/test/10.tif"
    spotWeight = "spotData"
    factor = "4"
    cellWeight = "cellData"
    return spotImagePath, spotWeight, factor, cellImagePath, cellWeight


if __name__ == "__main__":
    spotImagePath = "/home/pointseg/datasets/spotData/images/test/110.tif"
    cellImagePath = "/home/pointseg/datasets/cellData/images/test/110.tif"

    cellSpotSeg(spotImagePath, "receptorPth", 4, cellImagePath, "cellData")








