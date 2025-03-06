#!/bin/bash

# --pathRoot: path to dataset
# --outputDir: path to output directory
# --splitFactor: convert one image to splitFactor*splitFactor images

#python ./imgSplit.py --pathRoot /home/pointseg/datasets/deepblink/particle  \
#            --outputDir /home/pointseg/datasets/deepblinkSpl   \
#            --splitFactor 2

#python ./imgSplit.py --pathRoot /home/pointseg/datasets/deepblink/receptor  \
#            --outputDir /home/pointseg/datasets/deepblinkSpl   \
#            --splitFactor 2

#python ./imgSplit.py --pathRoot /home/pointseg/datasets/deepblink/vesicle  \
#            --outputDir /home/pointseg/datasets/deepblinkSpl   \
#            --splitFactor 2
#
#python ./imgSplit.py --pathRoot /home/pointseg/datasets/deepblink/smfish  \
#            --outputDir /home/pointseg/datasets/deepblinkSpl   \
#            --splitFactor 2
#
python ./imgSplit.py --pathRoot /home/pointseg/datasets/deepblink-reproduce/suntag  \
            --outputDir /home/pointseg/datasets/deepblink-reproduce   \
            --splitFactor 2
