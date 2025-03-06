import cv2

path = "/home/pointseg/datasets/deepblink/smfish/imagesCenter/train/1.png"

image = cv2.imread(path, cv2.IMREAD_UNCHANGED)


# 创建CLAHE对象，clipLimit和tileGridSize是CLAHE的两个重要参数
# clipLimit控制对比度，tileGridSize定义了局部区域的大小
clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(10,10))

# 应用CLAHE算法增强对比度
# 由于CLAHE函数期望8位图像，我们需要先将16位图像转换为8位
# 这可以通过归一化实现，但会丢失一些细节
image_8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
enhanced_image = clahe.apply(image_8bit)

# 显示原始图像和增强后的图像
cv2.imshow('Original Image', image_8bit)
cv2.imshow('Enhanced Image', enhanced_image)


cv2.waitKey(0)
cv2.destroyAllWindows()
