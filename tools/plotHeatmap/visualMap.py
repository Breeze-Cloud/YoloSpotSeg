import torch
import matplotlib.pyplot as plt
from ultralytics import YOLO
from PIL import Image
from torchvision.transforms import ToTensor, Resize, Compose
import cv2




#定义一个函数来提取并可视化特征图
def visualize_feature_maps(model, image):
    model.model.eval()
    feature_maps =[]
    # 定义钩子函数
    def hook(module, input, output):
        feature_maps.append(output)

    # 注册钩子到所有卷积层hooks =[]
    hooks = []
    for layer in model.model.modules():
        if isinstance(layer, torch.nn.Conv2d):
            hooks.append(layer.register_forward_hook(hook))

    # 前向传播以捕获特征图
    with torch.no_grad():
        _ = model(image)

    #取消注册钩子
    for hook in hooks:
        hook.remove()

    # 可视化特征图
    for i, fmap in enumerate(feature_maps):
        # if fmap.shape[1]<4:
        #     continue
        num_channels =fmap.shape[1]
        num_channels_to_show=fmap.shape[1]       #选择要可视化的通道数量(例如:4个通道)
        fig, axes = plt.subplots(1, num_channels_to_show)
        for j in range(num_channels_to_show):
            ax = axes[j]
            ax.imshow(fmap[0,j].cpu().numpy(),cmap='viridis')
            ax.axis('off')
        plt.suptitle(f'Feature Map {i + 1}',fontsize=20)
        plt.savefig(f"./headMap/{i}.png")

# 调用函数进行特征图可视化
# 加载模型
model = YOLO("/home/pointseg/Yolov8/runs/segment/receptorSpl/train15/weights/best.pt")
image_path = "/home/pointseg/datasets/deepblink/receptorSpl/images/test/0.tif"

image = cv2.imread(image_path)
image = Image.fromarray(image)
# 调整图像尺寸，使其符台模型的输入要求
resize =Compose([Resize((512,512)),ToTensor()])
image = resize(image).unsqueeze(0) # 转换为张量并增加批次维度
visualize_feature_maps(model, image)
