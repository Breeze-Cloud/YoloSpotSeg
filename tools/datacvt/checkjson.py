import json
import numpy as np
from tqdm import tqdm

def frPyObjects(i, pyobj):
    # encode rle from a list of python objects
    if type(pyobj) == np.ndarray:
        print("{}, {}, {}".format(i, type(pyobj), len(pyobj[0])))
    elif type(pyobj) == list and len(pyobj[0]) == 4:
        print("{}, {}, {}".format(i, type(pyobj), len(pyobj[0])))
    elif type(pyobj) == list and len(pyobj[0]) > 4:
        print("{}, {}, {}".format(i, type(pyobj), len(pyobj[0])))
    elif type(pyobj) == list and type(pyobj) == dict and 'counts' in pyobj[0] and 'size' in pyobj[0]:
        print("{}, {}, {}".format(i, type(pyobj), len(pyobj[0])))
    # encode rle from single python object
    elif type(pyobj) == list and len(pyobj[0]) == 4:
        print("{}, {}, {}".format(i, type(pyobj), len(pyobj[0])))
    elif type(pyobj) == list and len(pyobj[0]) > 4:
        print("{}, {}, {}".format(i, type(pyobj), len(pyobj[0])))
    elif type(pyobj) == dict and 'counts' in pyobj and 'size' in pyobj:
        print("{}, {}, {}".format(i, type(pyobj), len(pyobj[0])))
    else:
        print("{}, {}, {},  ERROR".format(i, type(pyobj), len(pyobj[0])))
        raise Exception('input type is not supported.')

def check(pathJson):
    jsonfile = open(pathJson, "r")
    jsonObj = json.load(jsonfile)
    jsonfile.close()

    for i, instance in tqdm(enumerate(jsonObj["annotations"])):
        frPyObjects(i, instance["segmentation"])

if __name__ == "__main__":
    # pathTrainJson = "/home/cellseg/datasets/cellpose/annotations/instances_train2017.json"
    pathTrainJson = "/home/pointseg/datasets/deeplink_coco/receptor/annotations/instances_train.json"
    pathValJson = "/home/pointseg/datasets/deeplink_coco/receptor/annotations/instances_val.json"
    pathTestJson = "/home/pointseg/datasets/deeplink_coco/receptor/annotations/instances_test.json"

    check(pathTrainJson)
    check(pathValJson)
    check(pathTestJson)

# 之后，可以根据上述print()打印出来的不符合条件对象的id，来修改对应的segmentation列表（手动修改即可，不符合条件的数量一般很少）
#json_object["annotations"][1510]["segmentation"] = [[230.83333333333331, 773.8888888888889, 231.83333333333331, 773.8888888888889, 237.22222222222223, 770.5555555555555]]

# 将修改后的json文件重新写回到coco_instancestrain.json/coco_instancesval.json中即可
# val_json = open(JSON_LOC, "w")
# json.dump(json_object, val_json)
# val_json.close()

