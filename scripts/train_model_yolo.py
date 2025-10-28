from ultralytics import YOLO

def train():
    model = YOLO("yolo11l.pt")

    model.train(
        data="../data/data.yaml",
        epochs=150,
        imgsz=960,
        batch=8,
        device="mps",
        workers=0,
        patience=50,
        name="FootAI_yolo11l",
        project="models",
        pretrained=True
    )
