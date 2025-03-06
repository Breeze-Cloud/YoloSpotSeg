from ultralytics import YOLO
import argparse


def get_parser():
    parser = argparse.ArgumentParser("get datatsets")

    parser.add_argument("--datasets_yaml", type=str, default="receptor.yaml")
    parser.add_argument("--pthPath", type=str)

    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()

    model = YOLO(args.pthPath)

    result = model.val(args.datasets_yaml)


