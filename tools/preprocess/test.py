import cv2
from PIL import Image
import numpy as np
from skimage.morphology import remove_small_objects
#
# pathgrad = "/home/pointseg/Datasets/receptor/gradData/test/images/61.tif"
# pathunet = "/home/pointseg/Datasets/receptor/unetData/test/images/pseudo_color_prediction/121.png"
#
# gradxy = cv2.imread(pathgrad, cv2.IMREAD_UNCHANGED)
#
# _, ostu_bin = cv2.threshold(gradxy, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# ostu_bin = ostu_bin > 0
# ostu_bin = remove_small_objects(ostu_bin, min_size=5)  # remove single pixel
# gradxy_remove_noise = np.zeros_like(gradxy)
# gradxy_remove_noise[ostu_bin == ostu_bin.max()] = gradxy[ostu_bin == ostu_bin.max()]
#
# # kernel = np.ones((2, 2), np.uint8)
# # grad_erode = cv2.GaussianBlur(gradxy_remove_noise, (9, 9), 0)
# high_pass_filtered_image = cv2.Laplacian(gradxy_remove_noise, ddepth=cv2.CV_64F)
# high_pass_filtered_image_8u = cv2.convertScaleAbs(high_pass_filtered_image)
#
#
# high_pass_filtered_image_8u = Image.fromarray(high_pass_filtered_image_8u)
# high_pass_filtered_image_8u.show()
#
#




# mask = cv2.imread(pathgrad, cv2.IMREAD_UNCHANGED)
#
# mask[mask > 0] = 255

# mask = Image.fromarray(mask)
# mask.show()

# unet = cv2.imread(pathunet, cv2.IMREAD_UNCHANGED)
#
#
# if len(unet.shape) == 3:
#     unet = unet[:, :, 0]
#
# filtGrad = np.zeros_like(grad)
# filtGrad[unet > 0] = grad[unet > 0]
# filtGrad = Image.fromarray(filtGrad)
# filtGrad.show()


path = "/home/pointseg/Datasets/receptor/gradData/train/masklabels/10.tif"
unet = cv2.imread(path, cv2.IMREAD_UNCHANGED)
unet[unet > 0] = 255
unet = Image.fromarray(unet)
unet.show()