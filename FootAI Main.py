from ultralytics import YOLO
import supervision as sv
import cv2
import os
import numpy as np
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

MODEL_PATH = "scripts/models/FootAI_yolo11l/weights/best.pt"
RETINA_WEIGHTS = "scripts/models/RetinaNet/model_best.pth"
VIDEO_PATH = "0bfacc_0.mp4"
OUTPUT_PATH = "outputs/tracked_match_shapes_yolo.mp4"
CONF_THRESH = 0.35
RETINA_CONF_THRESH =0.35
TRACKER_TYPE = "bytetrack.yaml"

CLASS_COLORS = {
    "player": (0, 255, 0),
    "goalkeeper": (0, 255, 255),
    "referee": (255, 0, 0),
    "ball": (255, 255, 255)
}

os.makedirs("outputs", exist_ok=True)

model = YOLO(MODEL_PATH)
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
cfg.MODEL.RETINANET.NUM_CLASSES = 1
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
cfg.MODEL.WEIGHTS = RETINA_WEIGHTS
cfg.MODEL.RETINANET.SCORE_THRESH_TEST = RETINA_CONF_THRESH

cfg.TEST.DETECTIONS_PER_IMAGE = 1
cfg.INPUT.MIN_SIZE_TEST = 1080
cfg.INPUT.MAX_SIZE_TEST = 1920
cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[6, 8, 10, 12, 16, 20, 24]]
cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 0.7, 1.0, 1.5]]

predictor_retina = DefaultPredictor(cfg)
results = model.track(
    source=VIDEO_PATH,
    conf=CONF_THRESH,
    tracker=TRACKER_TYPE,
    stream=True,
    show=False,
    persist=True
)

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
width, height = int(cap.get(3)), int(cap.get(4))
out = cv2.VideoWriter(
    OUTPUT_PATH,
    cv2.VideoWriter_fourcc(*'mp4v'),
    fps,
    (width, height)
)

ellipse_annotator = sv.EllipseAnnotator(thickness=2)
triangle_annotator = sv.TriangleAnnotator()
triangle_annotator_ball_retina = sv.TriangleAnnotator(
    color=sv.Color.from_hex("#FFA500"))
for result in results:
    frame = result.orig_img.copy()

    if result.boxes is None or len(result.boxes) == 0:
        out.write(frame)
        continue

    boxes = result.boxes.xyxy.cpu().numpy()
    class_ids = result.boxes.cls.cpu().numpy().astype(int)
    track_ids = result.boxes.id.cpu().numpy().astype(int)
    class_names = result.names

    detections = sv.Detections(
        xyxy=boxes,
        class_id=class_ids,
        tracker_id=track_ids
    )

    mask_ellipse = [class_names[int(cls)] in ["player", "goalkeeper"] for cls in class_ids]
    if any(mask_ellipse):
        detections_ellipse = detections[mask_ellipse]
        frame = ellipse_annotator.annotate(frame, detections_ellipse)

    mask_triangle = [class_names[int(cls)] in ["referee", "ball"] for cls in class_ids]
    if any(mask_triangle):
        detections_triangle = detections[mask_triangle]
        frame = triangle_annotator.annotate(frame, detections_triangle)

    # Dodanie predykcji z RetinaNet dla piłki i narysowanie ich
    try:
        outputs = predictor_retina(frame)
        instances = outputs.get("instances", None) if isinstance(outputs, dict) else getattr(outputs, "instances", None)
        if instances is not None and len(instances) > 0:
            instances = instances.to("cpu")

            if hasattr(instances, "scores") and hasattr(instances, "pred_boxes"):
                scores = instances.scores.numpy()
                keep_mask = scores >= RETINA_CONF_THRESH
                if keep_mask.any():
                    boxes_ret = instances.pred_boxes.tensor.numpy()[keep_mask]
                    # mapowanie klasy "ball" do id używanego przez YOLO (jeśli istnieje)
                    try:
                        ball_yolo_cls = next((k for k, v in class_names.items() if v == "ball"), 0)
                    except Exception:
                        ball_yolo_cls = 0
                    class_ids_ret = np.full(len(boxes_ret), fill_value=ball_yolo_cls, dtype=int)
                    track_ids_ret = np.full(len(boxes_ret), -1, dtype=int)

                    detections_retina = sv.Detections(
                        xyxy=boxes_ret,
                        class_id=class_ids_ret,
                        tracker_id=track_ids_ret
                    )

                    frame = triangle_annotator_ball_retina.annotate(frame, detections_retina)
    except Exception as e:
        print("Retina prediction error:", e)

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video with tracking saved under: {OUTPUT_PATH}")
