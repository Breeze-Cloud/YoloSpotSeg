from ultralytics import YOLO
import argparse


def get_parser():
    parser = argparse.ArgumentParser("get datatsets")

    parser.add_argument("--datasets_yaml", type=str, default="cellData.yaml")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--imgsz", type=int, default=1024)

    return parser

if __name__ == "__main__":

    args = get_parser().parse_args()

    model = YOLO('yolov8x-seg.yaml')
    # model = YOLO('yolov8x-seg.pt')





    # Train the model
    results = model.train(data=args.datasets_yaml, epochs=500, imgsz=args.imgsz, batch=args.batch_size)



