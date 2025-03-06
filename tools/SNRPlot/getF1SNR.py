from ultralytics import YOLO
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", type=str, default="receptor")
    parser.add_argument("--pthPath", type=str)
    parser.add_argument("--savePath", type=str)
    return parser

def get_F1(result):
    p = result.results_dict["metrics/precision(M)"]
    r = result.results_dict["metrics/recall(M)"]
    f1 = 2*p*r/(p+r)
    return f1


def SNRplot(F1List):
    x = ["SNR7", "SNR4", "SNR2", "SNR1"]

    plt.plot(x, F1List, marker='o', color="black")  # marker='o'表示用圆圈标记每个数据点
    plt.title(args.datasets)
    plt.ylim(0, 1)

    for i, (pos_x, pos_y) in enumerate(zip(x, F1List)):
        plt.text(pos_x, pos_y, f'{pos_y:.2f}', ha='center', va='bottom')

    plt.xlabel('SNR')
    plt.ylabel('F1')
    plt.grid(True)
    plt.savefig(args.datasets + "_f1.png")


if __name__ == "__main__":

    args = get_parser().parse_args()
    model = YOLO(args.pthPath)
    SNRYamlList = []

    SNR = [7, 4, 2, 1]
    for i in SNR:
        SNRYamlList.append(os.path.join("../../ultralytics/datasets",args.datasets + f"_spl_SNR{i}.yaml"))

    F1List = []
    for SNRYaml in SNRYamlList:
        result = model.val(SNRYaml)
        F1 = get_F1(result)
        F1List.append(F1)

    path = os.path.join(args.savePath, args.datasets + "_spl_SNR.npy")
    np.save(path, np.array(F1List))


    SNRplot(F1List)





