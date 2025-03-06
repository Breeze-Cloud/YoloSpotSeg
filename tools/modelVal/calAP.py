from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

# ================ 参数配置 ================
label_json_path = "/home/pointseg/cellDataMaskFormer/Mask2Former/datasets/cellDataCoco/annotations/celldata_instances_test.json"   # 标注标签文件路径
pred_json_path = "./new_pre_file.json"  # YOLOv8 预测结果路径
# =========================================


import argparse
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--anno_json', type=str, default=label_json_path,
                        help='training model path')
    parser.add_argument('--pred_json', type=str, default=pred_json_path, help='data yaml path')

    return parser.parse_known_args()[0]


if __name__ == '__main__':
    opt = parse_opt()
    anno_json = opt.anno_json
    pred_json = opt.pred_json

    anno = COCO(anno_json)  # init annotations api
    print(pred_json)
    pred = anno.loadRes(pred_json)  # init predictions api
    eval = COCOeval(anno, pred, 'bbox')
    eval.evaluate()
    eval.accumulate()
    eval.summarize()
    pass