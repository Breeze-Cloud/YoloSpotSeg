from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("/home/pointseg/Yolov8/runs/segment/receptorSpl/train15/weights/best.pt")
    imgPath = "/home/pointseg/datasets/deepblink/receptor/images/test/0.tif"
    result = model.predict(source=imgPath, augment=False)


