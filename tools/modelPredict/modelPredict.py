from ultralytics import YOLO
import argparse


def get_parser():
    parser = argparse.ArgumentParser("")
    parser.add_argument("--dataName", type=str)
    parser.add_argument("--dataPath", type=str)
    parser.add_argument("--pthPath", type=str)
    parser.add_argument("--outputPath", type=str)
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    model = YOLO(args.pthPath)

    result = model.predict(source=args.dataPath,
                           save=False,
                           save_txt=False,
                           save_crop=False,
                           project=args.outputPath,
                           name=args.dataName)

