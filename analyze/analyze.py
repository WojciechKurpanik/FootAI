import time

import yaml
import cv2
import supervision as sv
from ultralytics import YOLO

from player_tracking.PassDetector import PassDetector
from segmentation.frames import Frames
from player_tracking.tracker import Tracker
import numpy as np
import os
from logger.logger import logger
from player_tracking.embedding_assigner import EmbeddingTeamAssigner
import torch
from pitch_keypoints_tracking.pitch_configuration import SoccerPitchConfiguration
from pitch_keypoints_tracking.view_transformation import ViewTransformer
from pitch_keypoints_tracking.team_heatmap import HeatmapGenerator
from player_tracking.PossessionCalculator import PossessionCalculator
from player_tracking.PassDetector import PassDetector

from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.engine import DefaultPredictor
from detectron2.detectron2 import model_zoo

CONFIG = SoccerPitchConfiguration()

class Analyze:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_path = self.config["model_path"]
        self.confidence = self.config.get("confidence", 0.45)
        self.output_path = self.config["output_path"]
        self.keypoints_model_path = self.config['keypoint_model_path']

        # Config dla RetinaNet - detekcja piłki
        self.retina_weights = self.config["retina_path"]
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

        # self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.from_hex("#000000"))

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

        # self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.from_hex("#000000"))

        from collections import defaultdict
        self.timing_stats = defaultdict(list)

    def _log_timing_stats(self):
        logger.info("Timing stats")
        for component, times in self.timing_stats.items():
            avg_time = np.mean(times) * 1000  # ms
            max_time = np.max(times) * 1000
            min_time = np.min(times) * 1000
            logger.info(
                f"{component:20s}: avg={avg_time:6.2f}ms, "
                f"min={min_time:6.2f}ms, max={max_time:6.2f}ms"
            )
    def _predict_ball_with_retina(self, frame, class_names):
        if self.predictor_retina is None:
            return None

        start = time.time()
        outputs = self.predictor_retina(frame)
        end = time.time()
        self.timing_stats['RetinaNet'].append(end - start)

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

    def possesion_ui(self, frame, possession_calc: PossessionCalculator):
        possession_pct = possession_calc.get_possession_percentage()
        team1_pct = possession_pct.get(1, 0)
        team2_pct = possession_pct.get(2, 0)

        # tło dla tekstu (półprzezroczyste)
        overlay = frame.copy()
        cv2.rectangle(overlay, (frame.shape[1] - 300, 10), (frame.shape[1] - 10, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        # tekst posiadania
        cv2.putText(frame, f"Team 1: {team1_pct:.1f}%", (frame.shape[1] - 290, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 136, 0), 2)
        cv2.putText(frame, f"Team 2: {team2_pct:.1f}%", (frame.shape[1] - 290, 75),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (51, 51, 255), 2)
        return frame
    def passess_ui(self, frame, pass_calc: PassDetector):
            team_stats = pass_calc.get_passes_by_team()

            # Tło dla tekstu
            overlay = frame.copy()
            cv2.rectangle(overlay, (10, 10), (300, 200), (0, 0, 0), -1)
            frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

            # Team 1 (niebieski)
            cv2.putText(frame, "Team 1 Passes:", (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 136, 0), 2)
            cv2.putText(frame, f"  Total: {team_stats[1]['total']}", (20, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"  Successful: {team_stats[1]['successful']}", (20, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"  Accuracy: {team_stats[1]['accuracy']:.1f}%", (20, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            # Team 2 (czerwony)
            cv2.putText(frame, "Team 2 Passes:", (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (51, 51, 255), 2)
            cv2.putText(frame, f"  Total: {team_stats[2]['total']}", (20, 155),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            cv2.putText(frame, f"  Successful: {team_stats[2]['successful']}", (20, 175),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(frame, f"  Accuracy: {team_stats[2]['accuracy']:.1f}%", (20, 195),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

            return frame

    def run(self, video_path: str):  # Embeddings
        self.video_path = video_path

        frames = Frames(self.video_path)
        tracker = Tracker(
            model_path=self.model_path,
            conf_threshold=self.confidence,
        )

        team_assigner = EmbeddingTeamAssigner(device='cuda', max_history=5)

        start = time.time()
        results = tracker.track_video(self.video_path)
        end = time.time()
        self.timing_stats['YOLO players init'].append(end - start)

        # --- Inicjalizacja dwóch heatmap generatorów, po jednej dla drużyny ---
        heatmap_gen_team0 = HeatmapGenerator()
        heatmap_gen_team1 = HeatmapGenerator()

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        out = cv2.VideoWriter(
            self.output_path + "_analyzed_embeddings_no_labels.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            frames.fps,
            (frames.width, frames.height),
        )

        logger.info("Starting video analyze...")

        possession_calc = PossessionCalculator(distance_threshold=60.0, fps=25)
        pass_calc = PassDetector(possession_calc,
        min_confirm_frames=3,       # 3 detekcje wymagane
        confirm_window_frames=10,   # w oknie 10 klatek (0.4s)
        max_flight_frames=75)       # max 3s lotu

        for frame_idx, frame_from_results in enumerate(results):
            frame = frame_from_results.orig_img.copy()

            start = time.time()
            result = self.keypoints_model(frame, verbose=False)[0]
            end = time.time()
            self.timing_stats['YOLO keypoints'].append(end - start)

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
                start = time.time()
                team_assigner.initialize(frame, players, track_ids)
                end = time.time()
                self.timing_stats['Embeddings init'].append(end - start)

            ball_position = None
            detections_retina = self._predict_ball_with_retina(frame, ["ball"])
            if detections_retina is not None and len(detections_retina) > 0:
                ball_bbox = detections_retina.xyxy[0]
                ball_position = ((ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2)

            # -----------------------------TEST HEURYSTYKA BRAMKARZE ---------------------------
            # Na początku każdej klatki - czyścimy pozycje
            team_assigner.clear_frame_positions()

            # --- Anotacje graczy ---
            team1_players = []
            team2_players = []
            player_bboxes = []
            player_teams = []
            player_ids = []

            for det_idx, bbox in enumerate(players.xyxy):
                player_id = int(track_ids[det_idx])

                start = time.time()
                team_id = team_assigner.assign_player(frame, bbox, player_id)
                end = time.time()
                self.timing_stats['Embeddings assign'].append(end - start)

                # DODAJ TO: aktualizuj pozycje dla późniejszego przypisania bramkarzy
                team_assigner.update_team_positions(team_id, bbox)

                player_bboxes.append(bbox)
                player_teams.append(team_id)
                player_ids.append(player_id)

                if team_id == 1:
                    color = sv.Color.from_hex("#0088FF")
                elif team_id == 2:
                    color = sv.Color.from_hex("#FF3333")
                else:
                    color = sv.Color.from_hex("#00FF00")

                detections_for_player = sv.Detections(
                    xyxy=np.array([bbox]),
                    class_id=np.array([0]),
                    tracker_id=np.array([player_id])
                )
                ellipse_annotator = sv.EllipseAnnotator(color=color, thickness=2)
                frame = ellipse_annotator.annotate(frame, detections_for_player)

                if team_id == 1:
                    team1_players.append(player_id)
                else:
                    team2_players.append(player_id)

            # --- Bramkarze - ZMODYFIKOWANA SEKCJA ---
            mask_goalkeeper = np.array([class_names[c] == "goalkeeper" for c in detections.class_id])
            if np.any(mask_goalkeeper):
                detections_goalkeeper = detections[mask_goalkeeper]

                for gk_idx, gk_bbox in enumerate(detections_goalkeeper.xyxy):
                    # Przypisz bramkarza do drużyny
                    start = time.time()
                    gk_team_id = team_assigner.assign_goalkeeper(gk_bbox)
                    end = time.time()
                    self.timing_stats['Goalkeeper assign'].append(end - start)

                    # Wybierz kolor na podstawie drużyny
                    if gk_team_id == 1:
                        gk_color = sv.Color.from_hex("#0088FF")
                    else:
                        gk_color = sv.Color.from_hex("#FF3333")

                    # Annotacja bramkarza
                    gk_detection = sv.Detections(
                        xyxy=np.array([gk_bbox]),
                        class_id=np.array([detections_goalkeeper.class_id[gk_idx]])
                    )
                    ellipse_annotator = sv.EllipseAnnotator(color=gk_color, thickness=2)
                    frame = ellipse_annotator.annotate(frame, gk_detection)

            # ----------------------------- KONIEC TESTU HEURYSTYKI BRAMKARZY ---------------------------

            # # --- Anotacje graczy ---
            # team1_players = []
            # team2_players = []
            # #zwracanie bboxów i drużyn graczy do posiadania piłki
            # player_bboxes = []
            # player_teams = []
            # #player ids do wykrywania podań
            # player_ids = []
            # for det_idx, bbox in enumerate(players.xyxy):
            #     player_id = int(track_ids[det_idx])
            #
            #     start = time.time()
            #     team_id = team_assigner.assign_player(frame, bbox, player_id)
            #     end = time.time()
            #     self.timing_stats['Embeddings assign'].append(end - start)
            #
            #     player_bboxes.append(bbox)
            #     player_teams.append(team_id)
            #     player_ids.append(player_id)
            #     if team_id == 1:
            #         color = sv.Color.from_hex("#0088FF")
            #     elif team_id == 2:
            #         color = sv.Color.from_hex("#FF3333")
            #     else:
            #         color = sv.Color.from_hex("#00FF00")
            #
            #     detections_for_player = sv.Detections(
            #         xyxy=np.array([bbox]),
            #         class_id=np.array([0]),
            #         tracker_id=np.array([player_id])
            #     )
            #     ellipse_annotator = sv.EllipseAnnotator(color=color, thickness=2)
            #     frame = ellipse_annotator.annotate(frame, detections_for_player)
            #
            #     # label_text = f"Player {player_id} | Team {team_id}"
            #     x1, y1, x2, y2 = map(int, bbox)
            #     # cv2.putText(frame, label_text, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.as_bgr(), 1)
            #
            #     if team_id == 1:
            #         team1_players.append(player_id)
            #     else:
            #         team2_players.append(player_id)
            #
            if len(player_bboxes) > 0:
                pass_calc.update(frame_idx, ball_position, np.array(player_bboxes), player_teams, player_ids)

            frame = self.possesion_ui(frame, possession_calc)
            frame = self.passess_ui(frame, pass_calc)
            #
            # # --- Bramkarze ---
            # mask_goalkeeper = np.array([class_names[c] == "goalkeeper" for c in detections.class_id])
            # if np.any(mask_goalkeeper):
            #     detections_goalkeeper = detections[mask_goalkeeper]
            #     for cls in np.unique(detections_goalkeeper.class_id):
            #         mask_cls = detections_goalkeeper.class_id == cls
            #         frame = self.ellipse_annotators[class_names[cls]].annotate(
            #             frame, detections_goalkeeper[mask_cls]
            #         )

            # --- Piłka i sędzia ---
            mask_referee = np.array([class_names[c] == "referee" for c in detections.class_id])
            if np.any(mask_referee):
                detections_referee = detections[mask_referee]
                frame = self.triangle_annotators["referee"].annotate(frame, detections_referee)

            # --- Piłka z modelem Retina ---
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


        # --- Koniec pętli: zapis wideo i heatmap ---
        frames.release()
        out.release()
        cv2.destroyAllWindows()
        self._log_timing_stats()
        logger.info(f"Result saved in: {self.output_path}_analyzed_embeddings.mp4")
        heatmap_gen_team0.save_heatmap_on_pitch(team_id=0, output_dir="outputs")
        heatmap_gen_team1.save_heatmap_on_pitch(team_id=1, output_dir="outputs")
        logger.info("Saved heatmaps for both teams.")
        logger.info(f"Ball Possesion - Team 1: {possession_calc.get_possession_percentage().get(1, 0):.2f}%")
        logger.info(f"Ball Possesion - Team 2: {possession_calc.get_possession_percentage().get(2, 0):.2f}%")
