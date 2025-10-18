if __name__ == "__main__":
    from ultralytics import YOLO

    # Załaduj model YOLO11n
    model = YOLO("yolo11n.pt")

    # Trening modelu
    model.train(
        data="../data/data.yaml",
        epochs=100,
        imgsz=960,
        batch=16,
        device=0,
        workers=0,
        patience=10,
        name="FootAI_yolo11",
        project="models",
        pretrained=True
    )
