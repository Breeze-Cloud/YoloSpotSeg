#!/bin/bash



#  "particle":
#        kernal = (9, 9)
#        sigmaX = 2.5
#  "receptor":
#        kernal = (5, 5)
#        sigmaX = 0.9
#   "smfish":
#        kernal = (5, 5)
#        sigmaX = 0.9
#   "suntag":
#        kernal = (5, 5)
#        sigmaX = 0.9
#    "vesicle":
#        kernal = (3, 3)
#        sigmaX = 0.5


python ./npz2imgtxt.py --npzPath /home/pointseg/datasets/deepblink/row_npzfiles/particle.npz  \
            --outputDir /home/pointseg/datasets/deepblink_imglabel/particle    \
            --ksize 9   \
            --sigmax 2.5

python ./npz2imgtxt.py --npzPath /home/pointseg/datasets/deepblink/row_npzfiles/receptor.npz  \
            --outputDir /home/pointseg/datasets/deepblink_imglabel/receptor    \
            --ksize 5   \
            --sigmax 0.9

python ./npz2imgtxt.py --npzPath /home/pointseg/datasets/deepblink/row_npzfiles/vesicle.npz  \
            --outputDir /home/pointseg/datasets/deepblink_imglabel/vesicle    \
            --ksize 5   \
            --sigmax 0.9

python ./npz2imgtxt.py --npzPath /home/pointseg/datasets/deepblink/row_npzfiles/smfish.npz  \
            --outputDir /home/pointseg/datasets/deepblink_imglabel/smfish    \
            --ksize 5   \
            --sigmax 0.9


python ./npz2imgtxt.py --npzPath /home/pointseg/datasets/deepblink/row_npzfiles/suntag.npz  \
            --outputDir /home/pointseg/datasets/deepblink_imglabel/suntag    \
            --ksize 5   \
            --sigmax 0.9



#    if datasetName == "particle":
#        kernal = (9, 9)
#        sigmaX = 2.5
#    elif datasetName == "receptor":
#        kernal = (5, 5)
#        sigmaX = 0.9
#    elif datasetName == "smfish":
#        kernal = (5, 5)
#        sigmaX = 0.9
#    elif datasetName == "suntag":
#        kernal = (5, 5)
#        sigmaX = 0.9
#    elif datasetName == "vesicle":
#        kernal = (3, 3)
#        sigmaX = 0.5
