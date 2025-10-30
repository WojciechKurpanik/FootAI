import yaml
import cv2
import supervision as sv
from ultralytics import YOLO
from segmentation.frames import Frames
from tracking.tracking_objects import Tracker
import numpy as np
import os

class Analyze:
    def __init__(self, config_path: str):
        self.video_path = None
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_path = self.config["model_path"]
        self.confidence = self.config.get("confidence", 0.3)
        self.output_path = self.config["output_path"]
        self.tracker_config = self.config.get("tracker_config", "bytetrack.yaml")

        self.model = YOLO(self.model_path)

        self.CLASS_COLORS = {
            "player": sv.Color.from_hex("#00FF00"),
            "goalkeeper": sv.Color.from_hex("#00FFFF"),
            "referee": sv.Color.from_hex("#FF00FF"),
            "ball": sv.Color.from_hex("#FFFF00")
        }

        self.ellipse_annotators = {
            "player": sv.EllipseAnnotator(color=sv.Color.from_hex("#00FF00"), thickness=2),
            "goalkeeper": sv.EllipseAnnotator(color=sv.Color.from_hex("#00FFFF"), thickness=2)
        }

        self.triangle_annotators = {
            "referee": sv.TriangleAnnotator(color=sv.Color.from_hex("#FF00FF")),
            "ball": sv.TriangleAnnotator(color=sv.Color.from_hex("#FFFF00"))
        }

        self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.from_hex("#000000"))

    def run(self, video_path: str):
        self.video_path = video_path

        frames = Frames(self.video_path)
        tracker = Tracker(
            model_path=self.model_path,
            tracker_config=self.tracker_config,
            conf_threshold=self.confidence
        )

        results = tracker.track_video(self.video_path)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        out = cv2.VideoWriter(
            self.output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            frames.fps,
            (frames.width, frames.height)
        )

        for result in results:
            frame = result.orig_img.copy()
            detections = sv.Detections.from_ultralytics(result)

            if len(detections) == 0:
                out.write(frame)
                continue

            class_names = result.names

            mask_ellipse = [class_names[c] in ["player", "goalkeeper"] for c in detections.class_id]
            mask_triangle = [class_names[c] in ["ball", "referee"] for c in detections.class_id]

            if any(mask_ellipse):
                detections_ellipse = detections[np.array(mask_ellipse)]
                for cls in np.unique(detections_ellipse.class_id):
                    mask_cls = detections_ellipse.class_id == cls
                    frame = self.ellipse_annotators[class_names[cls]].annotate(
                        frame,
                        detections_ellipse[mask_cls]
                    )

            if any(mask_triangle):
                detections_triangle = detections[np.array(mask_triangle)]
                for cls in np.unique(detections_triangle.class_id):
                    mask_cls = detections_triangle.class_id == cls
                    frame = self.triangle_annotators[class_names[cls]].annotate(
                        frame,
                        detections_triangle[mask_cls]
                    )

            track_ids = detections.tracker_id if detections.tracker_id is not None else np.arange(len(detections))

            ball_mask = [class_names[c] == "ball" for c in detections.class_id]

            for i, is_ball in enumerate(ball_mask):
                if is_ball:
                    track_ids[i] = 0

            labels = [
                f"{class_names[c]} ID:{int(tid)}"
                for c, tid in zip(detections.class_id, track_ids)
            ]

            annotated = self.label_annotator.annotate(frame, detections, labels=labels)
            out.write(annotated)

        frames.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Result saved in: {self.output_path}")
