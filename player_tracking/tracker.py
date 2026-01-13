from ultralytics import YOLO

class Tracker:
    def __init__(self, model_path: str, conf_threshold: float = 0.45):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def track_video(self, video_path: str):
        return self.model.track(
            source=video_path,
            conf=self.conf_threshold,
            stream=True,
            persist=True,
            show=False,
            iou=0.5 #best 0.5
        )

