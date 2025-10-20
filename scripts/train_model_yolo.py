if __name__ == "__main__":
    from ultralytics import YOLO

    model = YOLO("yolo11l.pt")

    model.train(
        data="../data/data.yaml",
        epochs=150,
        imgsz=960,
        batch=8,
        device=0,
        workers=0,
        #patience=10,
        name="FootAI_yolo11l",
        project="models",
        pretrained=True
    )
