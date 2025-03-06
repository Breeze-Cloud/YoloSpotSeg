import json
import os
import shutil
import cv2
import numpy as np

# info ，license，categories 结构初始化；
# 在train.json,val.json,test.json里面信息是一致的；

# info，license暂时用不到
info = {
    "year": 2024,
    "version": '1.0',
    "date_created": 2024 - 12 - 7
}

licenses = {
    "id": 1,
    "name": "null",
    "url": "null",
}

# 自己的标签类别，跟yolo的数据集类别要对应好；
# id就是0，1，2，3 ...依次递增。 name就是标签名称，比如car、person。。。
categories = [
    {
        "id": 1,
        "name": 'spot',
        "supercategory": 'spot',
    },
]

# 初始化train,test、valid 数据字典
# info licenses categories 在train和test里面都是一致的；
train_data = {'info': info, 'licenses': licenses, 'categories': categories, 'images': [], 'annotations': []}
test_data = {'info': info, 'licenses': licenses, 'categories': categories, 'images': [], 'annotations': []}
valid_data = {'info': info, 'licenses': licenses, 'categories': categories, 'images': [], 'annotations': []}



def calculate_area(points):
    points = np.array(points)
    # 计算多边形面积
    return 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))

# image_path 对应yolov8的图像路径，比如images/train；
# label_path 对应yolov8的label路径，比如labels/train 跟images要对应；
def yolo_covert_coco_format(image_path, label_path):
    count = 0
    images = []
    annotations = []
    for index, img_file in enumerate(os.listdir(image_path)):
        if img_file.endswith('.tif'):
            image_info = {}
            img = cv2.imread(os.path.join(image_path, img_file), cv2.IMREAD_UNCHANGED)
            height, width = img.shape
            image_info['id'] = index
            image_info['file_name'] = img_file
            image_info['width'], image_info['height'] = width, height
        else:
            continue
        if image_info != {}:
            images.append(image_info)
        # 处理label信息-------
        label_file = os.path.join(label_path, img_file.replace('.tif', '.txt'))



        with open(label_file, 'r') as f:
            for idx, line in enumerate(f.readlines()):
                info_annotation = {}
                data = line.strip().split()
                class_id = int(data[0]) + 1  # According to COCO standards, numbering starts from one.
                polygon_points = [float(x) for x in data[1:]]
                num_points = len(polygon_points) // 2
                segmentation_per_line = []
                segmentation = []
                xmin, ymin = polygon_points[0 * 2] * width, polygon_points[0 * 2 + 1] * height
                xmax, ymax = xmin, ymin

                for i in range(num_points):
                    xmin = min(polygon_points[i * 2] * width, xmin)
                    xmax = max(polygon_points[i * 2] * width, xmax)

                    ymin = min(polygon_points[i * 2 + 1] * height, ymin)
                    ymax = max(polygon_points[i * 2 + 1] * height, ymax)

                    segmentation_per_line.append([polygon_points[i * 2] * width, polygon_points[i * 2 + 1] * height])
                    segmentation.append(polygon_points[i * 2] * width)
                    segmentation.append(polygon_points[i * 2 + 1] * height)
                w, h = xmax - xmin, ymax - ymin

                area = calculate_area(segmentation_per_line)
                segmentationList = []
                segmentationList.append(segmentation)
                # class_num, xs, ys, ws, hs = line.strip().split(' ')
                # class_id, xc, yc, w, h = int(class_num), float(xs), float(ys), float(ws), float(hs)
                # xmin = (xc - w / 2) * width
                # ymin = (yc - h / 2) * height
                # xmax = (xc + w / 2) * width
                # ymax = (yc + h / 2) * height
                # bbox_w = int(width * w)
                # bbox_h = int(height * h)
                img_copy = img[int(ymin):int(ymax), int(xmin):int(xmax)].copy()

                info_annotation["category_id"] = class_id  # 类别的id
                info_annotation['bbox'] = [xmin, ymin, w, h]  ## bbox的坐标
                info_annotation['area'] = area  ###area
                info_annotation['image_id'] = index  # bbox的id

                info_annotation['id'] = count  # bbox的id

                count = count + 1

                # cv2.imwrite(f"./temp/{info_annotation['id']}.jpg", img_copy)
                info_annotation['segmentation'] = segmentationList
                info_annotation['iscrowd'] = 0  # 单例
                annotations.append(info_annotation)
    return images, annotations


# key == train，test，val
# 对应要生成的json文件，比如instances_train.json，instances_test.json，instances_val.json
# 只是为了不重复写代码。。。。。
def gen_json_file(yolov8_data_path, coco_format_path, key):
    print('a1')
    # json path
    json_path = os.path.join(coco_format_path, f'annotations/instances_{key}.json')
    dst_path = os.path.join(coco_format_path, f'{key}')
    if not os.path.exists(os.path.dirname(json_path)):
        os.makedirs(os.path.dirname(json_path), exist_ok=True)
    data_path = os.path.join(yolov8_data_path, f'images/{key}')
    label_path = os.path.join(yolov8_data_path, f'labels/{key}')
    images, anns = yolo_covert_coco_format(data_path, label_path)
    print('a2')
    if key == 'train':
        train_data['images'] = images
        train_data['annotations'] = anns
        with open(json_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        # shutil.copy(data_path,'')
        print('a3')
    elif key == 'test':
        test_data['images'] = images
        test_data['annotations'] = anns
        with open(json_path, 'w') as f:
            json.dump(test_data, f, indent=2)
    elif key == 'val':
        valid_data['images'] = images
        valid_data['annotations'] = anns
        with open(json_path, 'w') as f:
            json.dump(valid_data, f, indent=2)
    else:
        print(f'key is {key}')
    print(f'generate {key} json success!')
    print('a4')
    return


if __name__ == '__main__':
    # 将下列两行代码路径替换为自己的数据集路径
    yolov8_data_path = '/home/pointseg/datasets/deepblink/receptor'  # 该路径为YOLO格式数据集根目录路径
    coco_format_path = '/home/pointseg/datasets/deeplink_coco/receptor'  # 该路径为存放coco格式数据集的根目录路径
    gen_json_file(yolov8_data_path, coco_format_path, key='train')
    gen_json_file(yolov8_data_path, coco_format_path, key='val')
    gen_json_file(yolov8_data_path, coco_format_path, key='test')
