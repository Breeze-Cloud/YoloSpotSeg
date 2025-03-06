import cv2
import numpy as np


def checkLabel(pathImg, pathTxtLabel, saveName):

    # get image
    image = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)
    if image.dtype != np.uint8:
        min = image.min()
        max = image.max()
        image = (image - min)  / (max - min) * 255
        image = image.astype(np.uint8)

    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # get label
    masks = np.zeros_like(image, dtype=np.uint8)

    with open(pathTxtLabel, 'r') as file:
        labels = file.read().splitlines()
    # plot
    for label in labels:
        label_parts = label.split(' ')
        points = label_parts[1:]
        # mask = np.zeros_like(image, dtype=np.uint8)
        xys = []
        # newPoints = []
        for i in range(0, len(points), 2):
            point = points[i:i + 2]
            y = round(float(point[0]) * image.shape[1])
            x = round(float(point[1]) * image.shape[0])
            masks[x, y] = (0, 255, 0)
            # xys.append([y, x])
            # newPoints.append(x)
            # newPoints.append(y)

        # xys = np.array(xys, dtype=np.int32)
        # cv2.fillPoly(masks, [xys], (0, 255, 0))

        # x_coords = newPoints[::2]  # 偶数索引为x坐标
        # y_coords = newPoints[1::2]  # 奇数索引为y坐标
        #
        # x_min, x_max = np.min(x_coords), np.max(x_coords)
        # y_min, y_max = np.min(y_coords), np.max(y_coords)
        #
        # cv2.rectangle(masks, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    new_img = cv2.addWeighted(image, 0.7, masks, 0.3, 0)


    cv2.imwrite(f"./{saveName}.png", new_img)




if __name__ == "__main__":
    imgPath = "/home/pointseg/datasets/deepblink/particle/images/train/0.tif"
    labelPath = "/home/pointseg/datasets/deepblink/particle/labels/train/0.txt"
    saveName = "particle"
    checkLabel(imgPath, labelPath, saveName)

    imgPath = "/home/pointseg/datasets/deepblink/receptor/images/train/0.tif"
    labelPath = "/home/pointseg/datasets/deepblink/receptor/labels/train/0.txt"
    saveName = "receptor"
    checkLabel(imgPath, labelPath, saveName)

    imgPath = "/home/pointseg/datasets/deepblink/vesicle/images/train/0.tif"
    labelPath = "/home/pointseg/datasets/deepblink/vesicle/labels/train/0.txt"
    saveName = "vesicle"
    checkLabel(imgPath, labelPath, saveName)

    imgPath = "/home/pointseg/datasets/deepblink/smfish/images/train/101.tif"
    labelPath = "/home/pointseg/datasets/deepblink/smfish/labels/train/101.txt"
    saveName = "smfish"
    checkLabel(imgPath, labelPath, saveName)

    imgPath = "/home/pointseg/datasets/deepblink/suntag/images/train/0.tif"
    labelPath = "/home/pointseg/datasets/deepblink/suntag/labels/train/0.txt"
    saveName = "suntag"
    checkLabel(imgPath, labelPath, saveName)