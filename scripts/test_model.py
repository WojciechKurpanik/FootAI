from ultralytics import YOLO
import cv2
import os

def test():
    model = YOLO("models/FootAI_yolo11l/weights/best.pt")
    test_dir = "../data/test/images/"
    out_dir = "../outputs/detections/"
    os.makedirs(out_dir, exist_ok=True)

    for img_name in os.listdir(test_dir):
        if not img_name.endswith((".jpg", ".png")): continue
        img_path = os.path.join(test_dir, img_name)
        results = model.predict(source=img_path, imgsz=960, conf=0.25)
        res_plotted = results[0].plot()
        cv2.imwrite(os.path.join(out_dir, img_name), res_plotted)

    print("Detection saved in", out_dir)
