import yaml
import cv2
import supervision as sv
from ultralytics import YOLO
from segmentation.frames import Frames
from player_tracking.tracker import Tracker
import numpy as np
import os
from logger.logger import logger
from player_tracking.embedding_assigner import EmbeddingTeamAssigner
import torch
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.engine import DefaultPredictor
from detectron2.detectron2 import model_zoo
from pitch_keypoints_tracking.pitch_configuration import SoccerPitchConfiguration
from pitch_keypoints_tracking.view_transformation import ViewTransformer
from pitch_keypoints_tracking.team_heatmap import HeatmapGenerator

CONFIG = SoccerPitchConfiguration()

class Analyze:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_path = self.config["model_path"]
        self.confidence = self.config.get("confidence", 0.45)
        self.output_path = self.config["output_path"]
        self.tracker_config = self.config.get("tracker_config", "bytetrack.yaml")
        self.keypoints_model_path = self.config.get("keypoints_model_path",
                                                    "scripts/models/Keypoints_yolo11x/weights/best.pt")

        # Config dla RetinaNet - detekcja piłki
        self.retina_weights = self.config.get("retina_weights", "scripts/models/RetinaNet/model_best.pth")
        self.retina_conf_thresh = self.config.get("retina_conf_thresh", 0.35)
        self.retina_min_size = self.config.get("retina_min_size", 1080)
        self.retina_max_size = self.config.get("retina_max_size", 1920)

        # Inicjalizacja modelu YOLO
        self.model = YOLO(self.model_path)
        self.keypoints_model = YOLO(self.keypoints_model_path)

        self.ellipse_annotators = {
            "goalkeeper": sv.EllipseAnnotator(color=sv.Color.from_hex("#00FFFF"), thickness=2),
        }

        self.triangle_annotators = {
            "referee": sv.TriangleAnnotator(color=sv.Color.from_hex("#FF00FF")),
            "ball": sv.TriangleAnnotator(color=sv.Color.from_hex("#FFFF00")),
        }

        self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.from_hex("#000000"))

        # Prediktor RetinaNet
        try:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
            cfg.MODEL.RETINANET.NUM_CLASSES = 1
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "mps"
            cfg.MODEL.WEIGHTS = self.retina_weights
            cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.retina_conf_thresh

            cfg.TEST.DETECTIONS_PER_IMAGE = 5
            cfg.INPUT.MIN_SIZE_TEST = self.retina_min_size
            cfg.INPUT.MAX_SIZE_TEST = self.retina_max_size
            cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[6, 8, 10, 12, 16, 20, 24]]
            cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 0.7, 1.0, 1.5]]

            self.predictor_retina = DefaultPredictor(cfg)
        except Exception as e:
            logger.warning(f"RetinaNet initialization failed: {e}")
            self.predictor_retina = None

        # dodatkowy annotator dla predykcji RetinaNet (piłka) - pomarańczowy
        self.triangle_annotator_ball_retina = sv.TriangleAnnotator(color=sv.Color.from_hex("#FFA500"))

        self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.from_hex("#000000"))

    def _predict_ball_with_retina(self, frame, class_names):
        if self.predictor_retina is None:
            return None

        outputs = self.predictor_retina(frame)
        instances = outputs["instances"].to("cpu")
        pred_boxes = instances.pred_boxes.tensor.numpy()
        scores = instances.scores.numpy()
        pred_classes = instances.pred_classes.numpy()

        ball_detections = []
        for box, score, cls in zip(pred_boxes, scores, pred_classes):
            if score >= self.retina_conf_thresh and class_names[cls] == "ball":
                ball_detections.append(box)

        if len(ball_detections) == 0:
            return None

        ball_detections = np.array(ball_detections)
        detections = sv.Detections(
            xyxy=ball_detections,
            class_id=np.zeros(len(ball_detections), dtype=int),
            confidence=scores[:len(ball_detections)]
        )
        return detections

    def run(self, video_path: str):  # Embeddings
        self.video_path = video_path

        frames = Frames(self.video_path)
        tracker = Tracker(
            model_path=self.model_path,
            tracker_config=self.tracker_config,
            conf_threshold=self.confidence,
        )

        team_assigner = EmbeddingTeamAssigner(device='cuda', max_history=5)
        results = tracker.track_video(self.video_path)

        # --- Inicjalizacja dwóch heatmap generatorów, po jednej dla drużyny ---
        heatmap_gen_team0 = HeatmapGenerator()
        heatmap_gen_team1 = HeatmapGenerator()

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        out = cv2.VideoWriter(
            self.output_path + "_analyzed_embeddings.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            frames.fps,
            (frames.width, frames.height),
        )

        logger.info("Starting video analyze...")

        for frame_idx, frame_from_results in enumerate(results):
            frame = frame_from_results.orig_img.copy()
            result = self.keypoints_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(result)
            detections = sv.Detections.from_ultralytics(frame_from_results)
            class_names = frame_from_results.names

            if len(detections) == 0:
                out.write(frame)
                continue

            if keypoints.confidence is not None:
                mask_conf = keypoints.confidence[0] < 0.7
                keypoints.xy[0][mask_conf] = np.nan

            # --- Wybierz zawodników (bez bramkarzy i sędziów) ---
            player_mask = np.array([class_names[c] == "player" for c in detections.class_id])
            players = detections[player_mask]
            track_ids = detections.tracker_id if detections.tracker_id is not None else np.arange(len(players))

            # --- Inicjalizacja embeddings drużyn (pierwsza klatka) ---
            if frame_idx == 0 and len(players) > 2:
                team_assigner.initialize(frame, players, track_ids)

            # --- Anotacje graczy ---
            team1_players = []
            team2_players = []
            for det_idx, bbox in enumerate(players.xyxy):
                player_id = int(track_ids[det_idx])
                team_id = team_assigner.assign_player(frame, bbox, player_id)

                color = sv.Color.from_hex("#0088FF") if team_id == 1 else sv.Color.from_hex("#FF3333")

                detections_for_player = sv.Detections(
                    xyxy=np.array([bbox]),
                    class_id=np.array([0]),
                    tracker_id=np.array([player_id])
                )
                ellipse_annotator = sv.EllipseAnnotator(color=color, thickness=2)
                frame = ellipse_annotator.annotate(frame, detections_for_player)

                label_text = f"Player {player_id} | Team {team_id}"
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, label_text, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.as_bgr(), 1)

                if team_id == 1:
                    team1_players.append(player_id)
                else:
                    team2_players.append(player_id)

            # --- Bramkarze ---
            mask_goalkeeper = np.array([class_names[c] == "goalkeeper" for c in detections.class_id])
            if np.any(mask_goalkeeper):
                detections_goalkeeper = detections[mask_goalkeeper]
                for cls in np.unique(detections_goalkeeper.class_id):
                    mask_cls = detections_goalkeeper.class_id == cls
                    frame = self.ellipse_annotators[class_names[cls]].annotate(
                        frame, detections_goalkeeper[mask_cls]
                    )

            # --- Piłka i sędzia ---
            mask_triangle = np.array([class_names[c] in ["ball", "referee"] for c in detections.class_id])
            if np.any(mask_triangle):
                detections_triangle = detections[mask_triangle]
                for cls in np.unique(detections_triangle.class_id):
                    mask_cls = detections_triangle.class_id == cls
                    frame = self.triangle_annotators[class_names[cls]].annotate(
                        frame, detections_triangle[mask_cls]
                    )

            # --- Piłka z modelem Retina ---
            detections_retina = self._predict_ball_with_retina(frame, class_names)
            if detections_retina is not None and len(detections_retina) > 0:
                frame = self.triangle_annotator_ball_retina.annotate(frame, detections_retina)

            # --- Aktualizacja heatmapy i transformacja ---
            if len(team1_players) > 0 or len(team2_players) > 0:
                valid_kp_mask = ~np.isnan(keypoints.xy[0][:, 0])

                if valid_kp_mask.sum() >= 4:
                    # KROK 1: Przygotowanie danych do homografii
                    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
                    transformer = ViewTransformer(
                        source=keypoints.xy[0][mask].astype(np.float32),
                        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
                    )

                    # KROK 2: Pobranie pozycji GRACZY
                    xy_players_video = players.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

                    # KROK 3: Transformacja pozycji GRACZY
                    transformed_xy_players = transformer.transform_points(xy_players_video)

                    # KROK 4: Aktualizacja heatmapy dla obu drużyn
                    if len(transformed_xy_players) > 0:
                        team1_xy = np.array(
                            [transformed_xy_players[i] for i, pid in enumerate(track_ids) if pid in team1_players])
                        team2_xy = np.array(
                            [transformed_xy_players[i] for i, pid in enumerate(track_ids) if pid in team2_players])

                        if team1_xy.size > 0:
                            heatmap_gen_team1.update_heatmap_from_xy(team1_xy)

                        if team2_xy.size > 0:
                            heatmap_gen_team0.update_heatmap_from_xy(team2_xy)

            out.write(frame)

            if frame_idx == 50:
                break

            print("frame: ", frame_idx)

        # --- Koniec pętli: zapis wideo i heatmap ---
        frames.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info(f"Result saved in: {self.output_path}_analyzed_embeddings.mp4")

        # --- Zapis obu heatmap ---
        heatmap_gen_team0.save_heatmap_on_pitch(team_id=0, output_dir="outputs")
        heatmap_gen_team1.save_heatmap_on_pitch(team_id=1, output_dir="outputs")
        logger.info("Saved heatmaps for both teams.")