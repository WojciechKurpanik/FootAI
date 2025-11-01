import cv2
from logger.logger import logger

class Frames:

    def __init__(self, source_path: str):
        self.cap = cv2.VideoCapture(source_path)
        if not self.cap.isOpened():
            logger.error("Cannot open video file")
            raise ValueError(f"Cannot open file: {source_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def __iter__(self):
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            yield frame

    def release(self):
        self.cap.release()
