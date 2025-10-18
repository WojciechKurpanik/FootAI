from ultralytics import YOLO

model = YOLO("yolo11n.pt")

model.train(
    data="../data/data.yaml",
    epochs=50,
    imgsz=960,
    batch=16,
    device="mps",
    workers=4,
    name="football_yolo11",
    project="models",
    pretrained=True
)
