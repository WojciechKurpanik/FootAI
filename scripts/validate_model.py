from ultralytics import YOLO

def validate():
    model = YOLO("runs/detect/yolo11x_football_finetune/weights/best.pt")
    results = model.val(
        data="../data/data.yaml",
        imgsz=960,
        conf=0.05,
        #device="mps",  # apple silicon
        device=0, #windows z gpu
        plots=True
    )
    return results