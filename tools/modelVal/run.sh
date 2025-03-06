#!/bin/bash


#python modelval.py --datasets_yaml ../../ultralytics/datasets/particle_spl.yaml  \
#                     --pthPath /home/pointseg/Yolov8/runs/segment/particleSpl/train17/weights/best.pt  \
#                    --iou 0.5

#python modelval.py --datasets_yaml ../../ultralytics/datasets/receptor_spl.yaml  \
#                     --pthPath /home/pointseg/Yolov8/runs/segment/receptorSpl/train15/weights/best.pt   \
#                    --iou 0.6
#
#python modelval.py --datasets_yaml ../../ultralytics/datasets/vesicle_spl.yaml  \
#                     --pthPath /home/pointseg/Yolov8/runs/segment/vesicleSpl/train16/weights/best.pt   \
#                    --iou 0.6

#python modelval.py --datasets_yaml ../../ultralytics/datasets/smfish_spl.yaml  \
#                     --pthPath /home/pointseg/Yolov8/runs/segment/smfishSpl/train13/weights/best.pt   \
#                    --iou 0.6
#
#python modelval.py --datasets_yaml ../../ultralytics/datasets/suntag_spl.yaml  \
#                     --pthPath /home/pointseg/Yolov8/runs/segment/suntagSpl/train12/weights/best.pt   \
#                    --iou 0.7
#

#python modelval.py --datasets_yaml ../../ultralytics/datasets/BF-DAPI-Fluor.yaml  \
#                     --pthPath /home/pointseg/Yolov8/runs/segment/BF-DAPI-Fluor/train41/weights/best.pt   \
#                    --iou 0.6

python modelval.py --datasets_yaml ../../ultralytics/datasets/cellData.yaml  \
                     --pthPath /home/pointseg/Yolov8/runs/segment/train25/weights/best.pt   \
                    --iou 0.6



