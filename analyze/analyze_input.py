import yaml
import cv2
import supervision as sv
from ultralytics import YOLO
from segmentation.frames import Frames
from tracking.tracking_objects import Tracker
from tracking.in2teams import In2Teams
import numpy as np
import os
from logger.logger import logger

class Analyze:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_path = self.config["model_path"]
        self.confidence = self.config.get("confidence", 0.3)
        self.output_path = self.config["output_path"]
        self.tracker_config = self.config.get("tracker_config", "bytetrack.yaml")

        # Inicjalizacja modelu YOLO
        self.model = YOLO(self.model_path)

        # Kolory i anotatory
        self.CLASS_COLORS = {
            "player": sv.Color.from_hex("#00FF00"),
            "goalkeeper": sv.Color.from_hex("#00FFFF"),
            "referee": sv.Color.from_hex("#FF00FF"),
            "ball": sv.Color.from_hex("#FFFF00"),
        }

        self.ellipse_annotators = {
            "player": sv.EllipseAnnotator(color=sv.Color.from_hex("#00FF00"), thickness=2),
            "goalkeeper": sv.EllipseAnnotator(color=sv.Color.from_hex("#00FFFF"), thickness=2),
        }

        self.triangle_annotators = {
            "referee": sv.TriangleAnnotator(color=sv.Color.from_hex("#FF00FF")),
            "ball": sv.TriangleAnnotator(color=sv.Color.from_hex("#FFFF00")),
        }

        self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.from_hex("#000000"))

    def run(self, video_path: str):
        self.video_path = video_path

        # --- Inicjalizacja narzędzi ---
        frames = Frames(self.video_path)
        tracker = Tracker(
            model_path=self.model_path,
            tracker_config=self.tracker_config,
            conf_threshold=self.confidence,
        )
        in2teams = In2Teams()

        results = tracker.track_video(self.video_path)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)

        out = cv2.VideoWriter(
            self.output_path + "_analyzed.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            frames.fps,
            (frames.width, frames.height),
        )

        logger.info("Starting video analyze...")

        for frame_idx, result in enumerate(results):
            frame = result.orig_img.copy()
            detections = sv.Detections.from_ultralytics(result)
            class_names = result.names

            # --- Pomiń, jeśli brak detekcji ---
            if len(detections) == 0:
                out.write(frame)
                continue

            # --- Wybierz zawodników (bez bramkarzy i sędziów) ---
            player_mask = np.array([class_names[c] == "player" for c in detections.class_id])
            players = detections[player_mask]
            track_ids = (
                detections.tracker_id if detections.tracker_id is not None else np.arange(len(detections))
            )

            # --- W pierwszej klatce ustal kolory drużyn ---
            if frame_idx == 0 and len(players) > 2:
                logger.info("Team's color initialization...")
                in2teams.assign_color_to_team(frame, players)

            # --- Anotacje graczy z kolorami drużyn ---
            for det_idx, bbox in enumerate(players.xyxy):
                player_id = int(track_ids[det_idx]) if track_ids is not None else det_idx
                team_id = in2teams.assign_player_to_team(frame, bbox, player_id)

                # Kolory drużyn (Team 1 = niebieski, Team 2 = czerwony)
                if team_id == 1:
                    color = sv.Color.from_hex("#0088FF")  # niebieski
                elif team_id == 2:
                    color = sv.Color.from_hex("#FF3333")  # czerwony
                else:
                    color = sv.Color.from_hex("#00FF00")  # zielony (np. jeśli KMeans nie przypisał)

                # naprawa: przekazujemy też class_id, żeby supervision nie rzucał błędem
                detections_for_player = sv.Detections(
                    xyxy=np.array([bbox]),
                    class_id=np.array([0]),  # placeholder (np. 0)
                    tracker_id=np.array([player_id])
                )

                # ustawiamy color_lookup na NONE, by wymusić użycie naszego koloru
                ellipse_annotator = sv.EllipseAnnotator(color=color, thickness=2)
                frame = ellipse_annotator.annotate(
                    frame,
                    detections_for_player
                )

                # etykieta gracza z ID i numerem drużyny
                label_text = f"Player {player_id} | Team {team_id}"
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(
                    frame,
                    label_text,
                    (x1, max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    color.as_bgr(),
                    1,
                    cv2.LINE_AA,
                )

            # --- Anotacje pozostałych klas (piłka, sędzia, bramkarz) ---
            mask_triangle = np.array([class_names[c] in ["ball", "referee"] for c in detections.class_id])
            if np.any(mask_triangle):
                detections_triangle = detections[mask_triangle]
                for cls in np.unique(detections_triangle.class_id):
                    mask_cls = detections_triangle.class_id == cls
                    frame = self.triangle_annotators[class_names[cls]].annotate(
                        frame, detections_triangle[mask_cls]
                    )

            # --- Zapisz klatkę ---
            out.write(frame)

        frames.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info(f"Result save in: {self.output_path}_analyzed.mp4")

