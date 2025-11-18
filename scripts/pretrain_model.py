from ultralytics import YOLO

def pretrain():
    model = YOLO("yolo11x.pt")
    model.train(
        data="../data/data.yaml",
        epochs=15,
        imgsz=960,
        batch=8,
        # device="mps", #apple silicon
        device=0, #windows z gpu
        lr0=0.001,
        optimizer="AdamW",
        amp=True,
        cos_lr=True,
        freeze=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=5,
        translate=0.05,
        scale=0.5,
        shear=2.0,
        mosaic=1.0,
        mixup=0.1,
        copy_paste=0.3,
        patience=10,
        plots=True,
        name="yolo11x",
        project="models",
        pretrained=True
    )