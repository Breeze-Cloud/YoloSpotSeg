import os
import cv2
import numpy as np
from PIL import Image
from numpy.matlib import zeros
from skimage.morphology import remove_small_objects


def loadImg(path):
    image = cv2.imread(pathImg, cv2.IMREAD_UNCHANGED)
    # if image.dtype == np.float32:
    #     min = image.min()
    #     max = image.max()
    #     image = (image - min) * (255 - 1) / (max - min) + 1
    #     image = image.astype(np.uint8)
    return image

def ScharrGradient(image):

    if image.dtype == np.uint8:
        # gradient x ,y
        grad_x = cv2.Scharr(image, cv2.CV_32F, 1, 0)  # 对x求一阶导
        grad_y = cv2.Scharr(image, cv2.CV_32F, 0, 1)  # 对y求一阶导
        gradx = cv2.convertScaleAbs(grad_x)  # 用convertScaleAbs()函数将其转回原来的uint8形式
        grady = cv2.convertScaleAbs(grad_y)
        gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)

        # remove noise
        _, ostu_bin = cv2.threshold(gradxy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ostu_bin = ostu_bin > 0
        ostu_bin = remove_small_objects(ostu_bin, min_size=5) # remove single pixel
        gradxy_remove_noise = np.zeros_like(gradxy)
        gradxy_remove_noise[ostu_bin == ostu_bin.max()] = gradxy[ostu_bin == ostu_bin.max()]

        kernel = np.ones((2, 2), np.uint8)
        grad_erode = cv2.erode(gradxy_remove_noise, kernel, iterations=1)


        _, thresh = cv2.threshold(grad_erode, 127, 255, cv2.THRESH_BINARY)
        mask = np.zeros((grad_erode.shape[0] + 2, grad_erode.shape[1] + 2), np.uint8)
        floodflags = 8
        floodflags |= cv2.FLOODFILL_MASK_ONLY
        floodflags |= (255 << 8)
        num, im, mask, rect = cv2.floodFill(thresh, mask, (0,0), 255, loDiff=(10,10,10), upDiff=(10,10,10), flags=floodflags)

        grad_hole = im
        grad_hole = Image.fromarray(grad_hole)
        grad_hole.save("./demoGrad/grad_hole.png")

        # add image
        im =np.zeros_like(image, dtype=np.uint16)
        im = im + image + gradxy_remove_noise
        image = np.clip(im, 0, 255).astype(np.uint8)

        # save image
        gradxy = Image.fromarray(gradxy)
        gradxy.save("./demoGrad/2-gradxy.png")

        gradxy_remove_noise = Image.fromarray(gradxy_remove_noise)
        gradxy_remove_noise.save("./demoGrad/3-gradxy_remove_noise.png")

        image = Image.fromarray(image)
        image.save("./demoGrad/4-image_gradxy.png")

        grad_erode = Image.fromarray(grad_erode)
        grad_erode.save("./demoGrad/grad_erode.png")

    elif image.dtype == np.float32:
        # grident
        grad_x32 = cv2.Scharr(image, cv2.CV_32F, 1, 0)  # 对x求一阶导
        grad_y32 = cv2.Scharr(image, cv2.CV_32F, 0, 1)  # 对y求一阶导
        gradxy32 = cv2.addWeighted(grad_x32, 0.5, grad_y32, 0.5, 0)
        cv2.imwrite("./demoGrad/2-gradxy32.tif", gradxy32)

        # dtype change
        min = image.min()
        max = image.max()
        img8 = (image - min) * 255 / (max - min)
        img8 = img8.astype(np.uint8)
        cv2.imwrite("./demoGrad/3-img8.tif", img8)

        # grident
        grad_x = cv2.Scharr(img8, cv2.CV_32F, 1, 0)  # 对x求一阶导
        grad_y = cv2.Scharr(img8, cv2.CV_32F, 0, 1)  # 对y求一阶导
        gradx = cv2.convertScaleAbs(grad_x)  # 用convertScaleAbs()函数将其转回原来的uint8形式
        grady = cv2.convertScaleAbs(grad_y)
        gradxy = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)
        cv2.imwrite("./demoGrad/4-gradxy.tif", gradxy)

        # remove noise
        _, ostu_bin = cv2.threshold(gradxy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        ostu_bin = ostu_bin > 0
        ostu_bin = remove_small_objects(ostu_bin, min_size=5)  # remove single pixel
        gradxy_remove_noise = np.zeros_like(gradxy32)
        gradxy_remove_noise[ostu_bin == ostu_bin.max()] = gradxy32[ostu_bin == ostu_bin.max()]
        cv2.imwrite("./demoGrad/5-gradxy_remove_noise.tif", gradxy_remove_noise)

        # add
        im = np.zeros_like(image, dtype=image.dtype)
        im = im + image + gradxy_remove_noise
        image = np.clip(im, min, max)
        cv2.imwrite("./demoGrad/6-image.tif", image)






pathSaveRoot = "/home/pointseg/datasets/deepblink"
img_savePath = os.path.join(pathSaveRoot, "particle", "images")
pathImg = os.path.join(img_savePath, "train", "0.tif")
img = loadImg(pathImg)
ScharrGradient(img)

cv2.imwrite("./demoGrad/1-image.tif", img)


