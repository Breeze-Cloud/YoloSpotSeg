#!/bin/bash

#python ./splitSNRImg.py --dataPathRoot /home/pointseg/datasets/deepblink/receptorSpl
#python ./splitSNRImg.py --dataPathRoot /home/pointseg/datasets/deepblink/vesicleSpl

python ./getF1SNR.py --datasets  receptor \
        --pthPath   /home/pointseg/Yolov8/runs/segment/receptorSpl/train15/weights/best.pt  \
        --savePath  /home/pointseg/Yolov8/tools/PlotFigs/result/yolov8

python ./getF1SNR.py --datasets  vesicle \
        --pthPath   /home/pointseg/Yolov8/runs/segment/vesicleSpl/train16/weights/best.pt  \
        --savePath  /home/pointseg/Yolov8/tools/PlotFigs/result/yolov8




