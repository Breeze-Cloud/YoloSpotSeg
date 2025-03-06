from ultralytics import YOLO
import argparse
def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets_yaml", type=str)
    parser.add_argument("--pthPath", type=str)
    parser.add_argument("--iou", type=float)
    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()

    model = YOLO(args.pthPath)

    result = model.val(args.datasets_yaml, iou=args.iou, max_det=3000, save_json=True)

