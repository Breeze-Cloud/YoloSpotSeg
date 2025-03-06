#!/bin/bash


#python getF1py.py --datasets_yaml ../../ultralytics/datasets/particle_spl.yaml  \
#                     --pthPath /home/pointseg/Yolov8/runs/segment/particleSpl/train17/weights/best.pt  \
#                    --iou 0.5  \
#                    --factor  2 \
#                    --saveName  /home/pointseg/Yolov8/tools/predictgetF1/particle_spl.xlsx
#
#python getF1py.py --datasets_yaml ../../ultralytics/datasets/receptor_spl.yaml  \
#                     --pthPath /home/pointseg/Yolov8/runs/segment/receptorSpl/train15/weights/best.pt   \
#                    --iou 0.5 \
#                    --factor  4 \
#                    --saveName  /home/pointseg/Yolov8/tools/predictgetF1/receptor_spl.xlsx
#
python getF1py.py --datasets_yaml ../../ultralytics/datasets/vesicle_spl.yaml  \
                     --pthPath /home/pointseg/Yolov8/runs/segment/vesicleSpl/train16/weights/best.pt   \
                    --iou 0.5 \
                    --factor  4 \
                    --saveName  /home/pointseg/Yolov8/tools/predictgetF1/vesicle_spl.xlsx

#python getF1py.py --datasets_yaml ../../ultralytics/datasets/smfish_spl.yaml  \
#                     --pthPath /home/pointseg/Yolov8/runs/segment/smfishSpl/train13/weights/best.pt   \
#                    --iou 0.5 \
#                    --factor  2 \
#                    --saveName  /home/pointseg/Yolov8/tools/predictgetF1/smfish_spl.xlsx


#python getF1py.py --datasets_yaml ../../ultralytics/datasets/suntag_spl.yaml  \
#                     --pthPath  /home/pointseg/Yolov8/runs/segment/suntagSpl/train12/weights/best.pt  \
#                    --iou 0.5 \
#                    --factor  2 \
#                    --saveName  /home/pointseg/Yolov8/tools/predictgetF1/suntag_spl.xlsx


#python getF1py.py --datasets_yaml ../../ultralytics/datasets/BF-DAPI-Fluor.yaml  \
#                     --pthPath /home/pointseg/Yolov8/runs/segment/BF-DAPI-Fluor/train41/weights/best.pt   \
#                    --iou 0.6


