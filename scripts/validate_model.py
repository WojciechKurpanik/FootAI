from ultralytics import YOLO

def validate():
    model = YOLO("models/FootAI_yolo11l_200epochs/weights/best.pt")
    results = model.val(
        data="../data/data_3classes.yaml",
        imgsz=960,
        conf=0.45,
        device="mps",  # apple silicon
        #device=0, #windows z gpu
        plots=True
    )
    return results

if __name__ == "__main__":
    validate()