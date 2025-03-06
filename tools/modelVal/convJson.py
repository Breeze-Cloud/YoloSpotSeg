import json
import os

pred_json_path = "/home/pointseg/Yolov8/tools/modelVal/runs/segment/val61/predictions.json"  # YOLOv8 预测结果路径

# 读取源文件
with open(pred_json_path, 'r') as file:
    data = json.load(file)

id_index = {95:1, 90:2, 85:3, 80:4, 75:5, 70:6, 65:7, 60:8, 55:9, 50:10, 40:11, 35:12, 30:13, 20:14, 115:15, 110:16, 105:17, 100:18, 10:19}

# 对每个字典进行处理
for item in data:
    # 如果字典中包含 'image_id' 键
    if 'image_id' in item:
        # 将 'image_id' 值转换为5位数形式，并在前面用0填充
        item['image_id'] = id_index[item['image_id']] # str(item['image_id']).zfill(5)
        item["category_id"] = 1

# 获取当前脚本所在目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建新文件的路径
new_file_path = os.path.join(current_dir, 'new_pre_file.json')

# 将修改后的内容写入到新文件中
with open(new_file_path, 'w') as file:
    json.dump(data, file, indent=4)