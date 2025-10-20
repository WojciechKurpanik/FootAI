from ultralytics import YOLO
import supervision as sv
import cv2
import os

MODEL_PATH = "scripts/models/FootAI_yolo11l/weights/best.pt"
VIDEO_PATH = r"C:\Users\Windows 10\Downloads\DFL Bundesliga Data Shootout\test\test (23).mp4"
OUTPUT_PATH = "outputs/tracked_match_shapes_yolo.mp4"
CONF_THRESH = 0.35
TRACKER_TYPE = "bytetrack.yaml"

CLASS_COLORS = {
    "player": (0, 255, 0),
    "goalkeeper": (0, 255, 255),
    "referee": (255, 0, 0),
    "ball": (255, 255, 255)
}

os.makedirs("outputs", exist_ok=True)

model = YOLO(MODEL_PATH)

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

    out.write(frame)

cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Video with tracking saved under: {OUTPUT_PATH}")
