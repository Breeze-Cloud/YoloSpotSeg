#!/bin/bash

#python ./plotResult.py --imgPath /home/pointseg/datasets/deepblink/receptorSpl/images/test/0.tif           \
#                      --gtpath /home/pointseg/datasets/deepblink/receptorSpl/labels/test/0.txt              \
#                      --predpath /home/pointseg/Yolov8/outputs/receptorPredict/receptor/labels/0.txt     \
#                      --savepath ./receptor
#
#python ./plotResult.py --imgPath /home/pointseg/datasets/deepblink/vesicleSpl/images/test/0.tif           \
#                      --gtpath /home/pointseg/datasets/deepblink/vesicleSpl/labels/test/0.txt              \
#                      --predpath /home/pointseg/Yolov8/outputs/vesiclePredict/vesicle/labels/0.txt     \
#                      --savepath ./vesicle
#
#
#python ./plotResult.py --imgPath /home/pointseg/datasets/deepblink/smfishSpl/images/test/0.tif           \
#                      --gtpath /home/pointseg/datasets/deepblink/smfishSpl/labels/test/0.txt              \
#                      --predpath /home/pointseg/Yolov8/outputs/smfishPredict/smfish/labels/0.txt     \
#                      --savepath ./smfish
#
#python ./plotResult.py --imgPath /home/pointseg/datasets/deepblink/suntagSpl/images/test/15.tif           \
#                      --gtpath /home/pointseg/datasets/deepblink/suntagSpl/labels/test/15.txt              \
#                      --predpath /home/pointseg/Yolov8/outputs/suntagPredict/suntag/labels/15.txt     \
#                      --savepath ./suntag

python ./plotResult.py --imgPath /home/pointseg/datasets/deepblink/receptorSpl/images/test/0.tif           \
                      --gtpath /home/pointseg/datasets/deepblink/receptorSpl/labels/test/0.txt              \
                      --predpath /home/pointseg/Yolov8/outputs/receptorPredict/receptor/labels/0.txt     \
                      --savepath ./BF-DAPI-Fluor