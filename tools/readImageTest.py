import cv2
import numpy as np

path8Bit = "./demo.png"
path32Bit = "/home/pointseg/datasets/deepblink/smfish/images/train/0.tif"

# directly read
print("directly read")
img8 = cv2.imread(path8Bit)
print("img 8 bit dtype: ", img8.dtype)
# img32 = cv2.imread(path32Bit)   # can not read
# print("img 32 bit dtype: ", img32.dtype)

#  cv2.IMREAD_UNCHANGED read
print("cv2.IMREAD_UNCHANGED read")
img8_1 = np.repeat(np.expand_dims(cv2.imread(path8Bit, cv2.IMREAD_UNCHANGED), axis=-1), 3, axis=2)
print("img 8 bit dtype: ", img8_1.dtype)
img32 = cv2.imread(path32Bit, cv2.IMREAD_UNCHANGED)
print("img 32 bit dtype: ", img32.dtype)

