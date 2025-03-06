#!/bin/bash

python ./modelPreditc.py --dataName cellData \
     --dataPath /home/pointseg/datasets/cellData/images/test \
     --pthPath /home/pointseg/Yolov8/runs/segment/cellData/train43/weights/best.pt \
     --outputPath ./outputs

