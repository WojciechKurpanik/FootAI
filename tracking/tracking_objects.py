from ultralytics import YOLO

class Tracker:
    def __init__(self, model_path: str, tracker_config: str = "bytetrack.yaml", conf_threshold: float = 0.35):
        self.model = YOLO(model_path)
        self.tracker_config = tracker_config
        self.conf_threshold = conf_threshold

    def track_video(self, video_path: str):
        return self.model.track(
            source=video_path,
            conf=self.conf_threshold,
            tracker=self.tracker_config,
            stream=True,
            persist=True,
            show=False
        )
