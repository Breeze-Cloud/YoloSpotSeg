import cv2
import numpy as np
import matplotlib.pyplot as plt


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



if __name__ == "__main__":
    # 读取图像
    image = cv2.imread('demo.png')

    # 执行傅里叶插值，将尺寸变为原来的2倍
    scale_factor = 2
    interpolated_image = fourier_interpolation(image, scale_factor)

    # 显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.title('Original Image')
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Interpolated Image')
    plt.imshow(interpolated_image, cmap='gray')
    plt.axis('off')

    plt.show()
