from ultralytics import YOLO

def train():
    model = YOLO("runs/detect/yolo11x/weights/last.pt")
    model.train(
        data="../data/data.yaml",
        epochs=200,
        imgsz=960,
        batch=6,
        # device="mps",  # apple silicon
        device=0, #windows z gpu
        lr0=0.0008,
        lrf=0.01,
        optimizer="AdamW",
        amp=True,
        warmup_epochs=3,
        freeze=0,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,
        patience=75,
        plots=True,
        save_period=25,
        cls=2.0,
        name="FootAI_yolo11x",
        project="models"
    )
