import yaml
import cv2
import supervision as sv
from ultralytics import YOLO
from segmentation.frames import Frames
from player_tracking.tracker import Tracker
from player_tracking.clustering_assigner import In2Teams
import numpy as np
import os
from logger.logger import logger
from player_tracking.fast_team_assigner import FastTeamAssigner
from player_tracking.heuristic_assigner import HeuristicTeamAssigner
from player_tracking.embedding_assigner import EmbeddingTeamAssigner

import torch
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.engine import DefaultPredictor
from detectron2.detectron2 import model_zoo

from pitch_keypoints_tracking import draw_pitch
from pitch_keypoints_tracking.pitch_configuration import SoccerPitchConfiguration
from pitch_keypoints_tracking.view_transformation import ViewTransformer
from pitch_keypoints_tracking.team_heatmap import HeatmapGenerator

import matplotlib.pyplot as plt

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
        heatmap_gen = HeatmapGenerator()

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

            # --- 1. ODDZIELNA DETEKCJA BOISKA ---
            # Uruchamiamy model keypointów na czystej klatce
            kp_result = self.keypoints_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(kp_result)

            # Filtrowanie słabych punktów (tak jak w Twoim teście)
            if keypoints.confidence is not None:
                mask_conf = keypoints.confidence[0] < 0.5  # próg pewności dla punktów
                keypoints.xy[0][mask_conf] = np.nan
            # ------------------------------------

            # --- Pomiń, jeśli brak detekcji GRACZY ---
            if len(detections) == 0:
                out.write(frame)
                continue

            # --- Wybierz zawodników ---
            player_mask = np.array([class_names[c] == "player" for c in detections.class_id])
            players = detections[player_mask]
            track_ids = (
                detections.tracker_id if detections.tracker_id is not None else np.arange(len(detections))
            )

            # --- W pierwszej klatce ustal kolory drużyn ---
            if frame_idx == 0 and len(players) > 2:
                logger.info("Team's color initialization...")
                in2teams.assign_color_to_team(frame, players)

            # --- Aktualizacja heatmapy i transformacja ---
            if len(players) > 0:
                # Sprawdzamy, czy model keypointów coś znalazł (liczymy punkty nie-NaN)
                # keypoints.xy[0] ma kształt (N, 2). Sprawdzamy czy nie są NaN.
                valid_kp_mask = ~np.isnan(keypoints.xy[0][:, 0])

                if valid_kp_mask.sum() >= 4:
                    # KROK 1: Przygotowanie danych do KALIBRACJI (Keypoints)
                    # To są punkty, które mówią "gdzie jest boisko na obrazie"
                    src_pts = keypoints.xy[0][valid_kp_mask].astype(np.float32)
                    dst_pts = np.array(CONFIG.vertices)[valid_kp_mask].astype(np.float32)

                    # KROK 2: Obliczenie macierzy transformacji
                    transformer = ViewTransformer(config=CONFIG)
                    transformer.fit(src_pts, dst_pts)

                    # KROK 3: Pobranie pozycji GRACZY (To jest to, czego szukasz!)
                    # Pobieramy środek dolnej krawędzi ramki gracza (czyli tam, gdzie stoją stopy)
                    xy_players_video = players.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

                    # KROK 4: Transformacja pozycji GRACZY
                    # Używamy macierzy (z keypointów) żeby przeliczyć pozycje (graczy)
                    transformed_xy_players = transformer.transform_points(xy_players_video)

                    # KROK 5: Aktualizacja heatmapy danymi GRACZY
                    # Upewnij się, że tutaj przekazujesz zmienną z kroku 4, a nie src_pts czy dst_pts!
                    heatmap_gen.update_heatmap_from_xy(transformed_xy_players)

                else:
                    if frame_idx % 20 == 0:
                        print(f"Frame {frame_idx}: Zbyt mało punktów boiska do kalibracji.")

            # --- Anotacje graczy i obiektów ---
            for det_idx, bbox in enumerate(players.xyxy):
                player_id = int(track_ids[det_idx])
                team_id = in2teams.assign_player_to_team(frame, bbox, player_id)
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
                cv2.putText(frame, label_text, (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.as_bgr(), 1, cv2.LINE_AA)

            # --- Bramkarze ---
            mask_goalkeeper = np.array([class_names[c] in ["goalkeeper"] for c in detections.class_id])
            if np.any(mask_goalkeeper):
                detections_goalkeeper = detections[mask_goalkeeper]
                for cls in np.unique(detections_goalkeeper.class_id):
                    mask_cls = detections_goalkeeper.class_id == cls
                    frame = self.ellipse_annotators[class_names[cls]].annotate(frame, detections_goalkeeper[mask_cls])

            # --- Piłka i sędzia ---
            mask_triangle = np.array([class_names[c] in ["ball", "referee"] for c in detections.class_id])
            if np.any(mask_triangle):
                detections_triangle = detections[mask_triangle]
                for cls in np.unique(detections_triangle.class_id):
                    mask_cls = detections_triangle.class_id == cls
                    frame = self.triangle_annotators[class_names[cls]].annotate(frame, detections_triangle[mask_cls])

                # Piłka z modelem Retina
                detections_retina = self._predict_ball_with_retina(frame, class_names)
                if detections_retina is not None and len(detections_retina) > 0:
                    frame = self.triangle_annotator_ball_retina.annotate(frame, detections_retina)

            out.write(frame)

            if frame_idx == 25:
                break

        # --- Zakończenie ---
        frames.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info(f"Result save in: {self.output_path}_analyzed.mp4")

        # --- Zapis heatmapy na obraz boiska ---
        heatmap_gen.save_heatmap_on_pitch()

    # def run(self, video_path: str): #FastTeamAssigner
    #     self.video_path = video_path
    #
    #     # --- Inicjalizacja narzędzi ---
    #     frames = Frames(self.video_path)
    #     tracker = Tracker(
    #         model_path=self.model_path,
    #         tracker_config=self.tracker_config,
    #         conf_threshold=self.confidence,
    #     )
    #     team_assigner = FastTeamAssigner()
    #
    #     results = tracker.track_video(self.video_path)
    #     os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
    #
    #     out = cv2.VideoWriter(
    #         self.output_path + "_analyzed_fast.mp4",
    #         cv2.VideoWriter_fourcc(*"mp4v"),
    #         frames.fps,
    #         (frames.width, frames.height),
    #     )
    #
    #     logger.info("Starting video analyze...")
    #
    #     # --- Pętla po klatkach ---
    #     for frame_idx, result in enumerate(results):
    #         frame = result.orig_img.copy()
    #         detections = sv.Detections.from_ultralytics(result)
    #         class_names = result.names
    #
    #         if len(detections) == 0:
    #             out.write(frame)
    #             continue
    #
    #         # --- Wyodrębnij graczy, bramkarzy, piłkę, sędziego ---
    #         player_mask = np.array([class_names[c] == "player" for c in detections.class_id])
    #         goalkeeper_mask = np.array([class_names[c] == "goalkeeper" for c in detections.class_id])
    #         ball_mask = np.array([class_names[c] == "ball" for c in detections.class_id])
    #         referee_mask = np.array([class_names[c] == "referee" for c in detections.class_id])
    #
    #         players = detections[player_mask]
    #         goalkeepers = detections[goalkeeper_mask]
    #
    #         # --- ID obiektów (dla consistency) ---
    #         track_ids = (
    #             detections.tracker_id if detections.tracker_id is not None else np.arange(len(detections))
    #         )
    #
    #         # --- Inicjalizacja kolorów drużyn (tylko raz) ---
    #         if frame_idx == 0 and len(players) > 2:
    #             logger.info("Team color initialization using FastTeamAssigner...")
    #             team_assigner.initialize(frame, players)
    #
    #         # --- Aktualizacja centroidów drużyn ---
    #         if len(players) > 0:
    #             team_assigner.update(frame, players)
    #
    #         # --- Anotowanie graczy ---
    #         for det_idx, bbox in enumerate(players.xyxy):
    #             player_id = int(track_ids[det_idx]) if track_ids is not None else det_idx
    #             team_id = team_assigner.assign_player_to_team(frame, bbox, player_id)
    #
    #             color = sv.Color.from_hex("#0088FF") if team_id == 1 else sv.Color.from_hex("#FF3333")
    #
    #             ellipse_annotator = sv.EllipseAnnotator(color=color, thickness=2)
    #             frame = ellipse_annotator.annotate(
    #                 frame,
    #                 sv.Detections(
    #                     xyxy=np.array([bbox]),
    #                     class_id=np.array([0]),
    #                     tracker_id=np.array([player_id])
    #                 )
    #             )
    #
    #             label_text = f"Player {player_id} | Team {team_id}"
    #             x1, y1, x2, y2 = map(int, bbox)
    #             cv2.putText(frame, label_text, (x1, max(15, y1 - 10)),
    #                         cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.as_bgr(), 1, cv2.LINE_AA)
    #
    #         # --- Bramkarze: przypisanie po centroidzie drużyny ---
    #         if len(goalkeepers) > 0:
    #             for g_idx, bbox in enumerate(goalkeepers.xyxy):
    #                 g_id = int(track_ids[g_idx]) if track_ids is not None else g_idx
    #                 team_id = team_assigner.assign_goalkeeper_to_team(bbox)  # 🔥 nowa metoda
    #
    #                 g_color = sv.Color.from_hex("#00FFFF") if team_id == 1 else sv.Color.from_hex("#FFAA33")
    #                 ellipse_annotator = sv.EllipseAnnotator(color=g_color, thickness=2)
    #                 frame = ellipse_annotator.annotate(
    #                     frame,
    #                     sv.Detections(
    #                         xyxy=np.array([bbox]),
    #                         class_id=np.array([1]),
    #                         tracker_id=np.array([g_id])
    #                     )
    #                 )
    #
    #                 label_text = f"GK {g_id} | Team {team_id}"
    #                 x1, y1, x2, y2 = map(int, bbox)
    #                 cv2.putText(frame, label_text, (x1, max(15, y1 - 10)),
    #                             cv2.FONT_HERSHEY_SIMPLEX, 0.5, g_color.as_bgr(), 1, cv2.LINE_AA)
    #
    #         # --- Piłka i sędzia (opcjonalnie) ---
    #         mask_triangle = np.logical_or(ball_mask, referee_mask)
    #         if np.any(mask_triangle):
    #             detections_triangle = detections[mask_triangle]
    #             for cls in np.unique(detections_triangle.class_id):
    #                 mask_cls = detections_triangle.class_id == cls
    #                 frame = self.triangle_annotators[class_names[cls]].annotate(
    #                     frame, detections_triangle[mask_cls]
    #                 )
    #
    #         out.write(frame)
    #
    #     frames.release()
    #     out.release()
    #     cv2.destroyAllWindows()
    #     logger.info(f"Result saved in: {self.output_path}_analyzed.mp4")

    # def run(self, video_path: str):  #Heuristic
    #     self.video_path = video_path
    #
    #     # --- Inicjalizacja narzędzi ---
    #     frames = Frames(self.video_path)
    #     tracker = Tracker(
    #         model_path=self.model_path,
    #         tracker_config=self.tracker_config,
    #         conf_threshold=self.confidence,
    #     )
    #
    #     team_assigner = HeuristicTeamAssigner(max_history=3, overlap_threshold=0.2)
    #
    #     results = tracker.track_video(self.video_path)
    #     os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
    #
    #     out = cv2.VideoWriter(
    #         self.output_path + "_analyzed_heuristic.mp4",
    #         cv2.VideoWriter_fourcc(*"mp4v"),
    #         frames.fps,
    #         (frames.width, frames.height),
    #     )
    #
    #     logger.info("Starting video analyze...")
    #
    #     for frame_idx, result in enumerate(results):
    #         frame = result.orig_img.copy()
    #         detections = sv.Detections.from_ultralytics(result)
    #         class_names = result.names
    #
    #         if len(detections) == 0:
    #             out.write(frame)
    #             continue
    #
    #         # --- zawodnicy bez bramkarzy i sędziów ---
    #         player_mask = np.array([class_names[c] == "player" for c in detections.class_id])
    #         players = detections[player_mask]
    #         track_ids = (
    #             detections.tracker_id if detections.tracker_id is not None else np.arange(len(detections))
    #         )
    #
    #         # --- Inicjalizacja kolorów drużyn (pierwsza klatka) ---
    #         if frame_idx == 0 and len(players) > 2:
    #             logger.info("Team color initialization...")
    #             team_assigner.initialize(frame, players)
    #
    #         # --- Anotacje graczy z kolorami drużyn ---
    #         for det_idx, bbox in enumerate(players.xyxy):
    #             player_id = int(track_ids[det_idx]) if track_ids is not None else det_idx
    #             team_id = team_assigner.assign_player_to_team(frame, bbox, player_id, players)
    #
    #             # Kolory drużyn
    #             if team_id == 1:
    #                 color = sv.Color.from_hex("#0088FF")
    #             elif team_id == 2:
    #                 color = sv.Color.from_hex("#FF3333")
    #
    #             detections_for_player = sv.Detections(
    #                 xyxy=np.array([bbox]),
    #                 class_id=np.array([0]),  # placeholder
    #                 tracker_id=np.array([player_id])
    #             )
    #
    #             ellipse_annotator = sv.EllipseAnnotator(color=color, thickness=2)
    #             frame = ellipse_annotator.annotate(frame, detections_for_player)
    #
    #             label_text = f"Player {player_id} | Team {team_id}"
    #             x1, y1, x2, y2 = map(int, bbox)
    #             cv2.putText(
    #                 frame,
    #                 label_text,
    #                 (x1, max(15, y1 - 10)),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.5,
    #                 color.as_bgr(),
    #                 1,
    #                 cv2.LINE_AA,
    #             )
    #
    #         # --- Bramkarze ---
    #         mask_goalkeeper = np.array([class_names[c] == "goalkeeper" for c in detections.class_id])
    #         if np.any(mask_goalkeeper):
    #             detections_goalkeeper = detections[mask_goalkeeper]
    #             for cls in np.unique(detections_goalkeeper.class_id):
    #                 mask_cls = detections_goalkeeper.class_id == cls
    #                 frame = self.ellipse_annotators[class_names[cls]].annotate(
    #                     frame, detections_goalkeeper[mask_cls]
    #                 )
    #
    #         # --- Piłka i sędzia ---
    #         mask_triangle = np.array([class_names[c] in ["ball", "referee"] for c in detections.class_id])
    #         if np.any(mask_triangle):
    #             detections_triangle = detections[mask_triangle]
    #             for cls in np.unique(detections_triangle.class_id):
    #                 mask_cls = detections_triangle.class_id == cls
    #                 frame = self.triangle_annotators[class_names[cls]].annotate(
    #                     frame, detections_triangle[mask_cls]
    #                 )
    #         #Piłka z modelem Retina
    #         detections_retina = self._predict_ball_with_retina(frame, class_names)
    #         if detections_retina is not None and len(detections_retina) > 0:
    #             frame = self.triangle_annotator_ball_retina.annotate(frame, detections_retina)
    #
    #         out.write(frame)
    #
    #     frames.release()
    #     out.release()
    #     cv2.destroyAllWindows()
    #     logger.info(f"Result saved in: {self.output_path}_analyzed.mp4")

    # def run(self, video_path: str): #Embeddings
    #     self.video_path = video_path
    #
    #     frames = Frames(self.video_path)
    #     tracker = Tracker(
    #         model_path=self.model_path,
    #         tracker_config=self.tracker_config,
    #         conf_threshold=self.confidence,
    #     )
    #
    #     team_assigner = EmbeddingTeamAssigner(device='cuda', max_history=5)
    #
    #     results = tracker.track_video(self.video_path)
    #
    #     os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
    #     out = cv2.VideoWriter(
    #         self.output_path + "_analyzed_embeddings.mp4",
    #         cv2.VideoWriter_fourcc(*"mp4v"),
    #         frames.fps,
    #         (frames.width, frames.height),
    #     )
    #
    #     logger.info("Starting video analyze...")
    #
    #     for frame_idx, result in enumerate(results):
    #         frame = result.orig_img.copy()
    #         detections = sv.Detections.from_ultralytics(result)
    #         class_names = result.names
    #
    #         if len(detections) == 0:
    #             out.write(frame)
    #             continue
    #
    #         # --- Wybierz zawodników (bez bramkarzy i sędziów) ---
    #         player_mask = np.array([class_names[c] == "player" for c in detections.class_id])
    #         players = detections[player_mask]
    #         track_ids = detections.tracker_id if detections.tracker_id is not None else np.arange(len(players))
    #
    #         # --- Inicjalizacja embeddings drużyn (pierwsza klatka) ---
    #         if frame_idx == 0 and len(players) > 2:
    #             team_assigner.initialize(frame, players, track_ids)
    #
    #         # --- Anotacje graczy ---
    #         for det_idx, bbox in enumerate(players.xyxy):
    #             player_id = int(track_ids[det_idx])
    #             team_id = team_assigner.assign_player(frame, bbox, player_id)
    #
    #             color = sv.Color.from_hex("#0088FF") if team_id == 1 else sv.Color.from_hex("#FF3333")
    #
    #             detections_for_player = sv.Detections(
    #                 xyxy=np.array([bbox]),
    #                 class_id=np.array([0]),
    #                 tracker_id=np.array([player_id])
    #             )
    #             ellipse_annotator = sv.EllipseAnnotator(color=color, thickness=2)
    #             frame = ellipse_annotator.annotate(frame, detections_for_player)
    #
    #             label_text = f"Player {player_id} | Team {team_id}"
    #             x1, y1, x2, y2 = map(int, bbox)
    #             cv2.putText(frame, label_text, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.as_bgr(), 1)
    #
    #         # --- Bramkarze ---
    #         mask_goalkeeper = np.array([class_names[c] == "goalkeeper" for c in detections.class_id])
    #         if np.any(mask_goalkeeper):
    #             detections_goalkeeper = detections[mask_goalkeeper]
    #             for cls in np.unique(detections_goalkeeper.class_id):
    #                 mask_cls = detections_goalkeeper.class_id == cls
    #                 frame = self.ellipse_annotators[class_names[cls]].annotate(
    #                     frame, detections_goalkeeper[mask_cls]
    #                 )
    #
    #         # --- Piłka i sędzia ---
    #         mask_triangle = np.array([class_names[c] in ["ball", "referee"] for c in detections.class_id])
    #         if np.any(mask_triangle):
    #             detections_triangle = detections[mask_triangle]
    #             for cls in np.unique(detections_triangle.class_id):
    #                 mask_cls = detections_triangle.class_id == cls
    #                 frame = self.triangle_annotators[class_names[cls]].annotate(
    #                     frame, detections_triangle[mask_cls]
    #                 )
    #
    #         #Piłka z modelem Retina
    #             detections_retina = self._predict_ball_with_retina(frame, class_names)
    #             if detections_retina is not None and len(detections_retina) > 0:
    #                 frame = self.triangle_annotator_ball_retina.annotate(frame, detections_retina)
    #
    #             out.write(frame)
    #
    #     frames.release()
    #     out.release()
    #     cv2.destroyAllWindows()
    #     logger.info(f"Result saved in: {self.output_path}_analyzed_embeddings.mp4")
