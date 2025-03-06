import numpy as np


path = "/home/pointseg/datasets/deepblink/row_npzfiles/suntag.npz"


npz = np.load(path)

with np.load(path, allow_pickle=True) as data:
    t = data["y_test"]
    pass
