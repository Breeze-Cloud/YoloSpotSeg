import os
import cv2
import numpy as np

img_path = "/home/pointseg/datasets/deepblink/receptor/images/test/0.tif"
save_path = "/home/pointseg/deepblink/deepBlink-master/speed/0.png"


img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
img = img.astype(np.int16)

cv2.imwrite(save_path, img)



