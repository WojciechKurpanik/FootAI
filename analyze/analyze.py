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
import torch
from detectron2.detectron2.config import get_cfg
from detectron2.detectron2.engine import DefaultPredictor
from detectron2.detectron2 import model_zoo
from pitch_keypoints_tracking.pitch_configuration import SoccerPitchConfiguration
from pitch_keypoints_tracking.view_transformation import ViewTransformer
from pitch_keypoints_tracking.team_heatmap import HeatmapGenerator


class Analyze:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_path = self.config["model_path"]
        self.confidence = self.config.get("confidence", 0.45)
        self.output_path = self.config["output_path"]
        self.tracker_config = self.config.get("tracker_config", "bytetrack.yaml")

        # RetinaNet - detekcja piłki
        self.retina_weights = self.config.get("retina_weights", "scripts/models/RetinaNet/model_best.pth")
        self.retina_conf_thresh = self.config.get("retina_conf_thresh", 0.35)
        self.retina_min_size = self.config.get("retina_min_size", 1080)
        self.retina_max_size = self.config.get("retina_max_size", 1920)

        # YOLO keypoints boiska
        self.keypoint_model_path = self.config.get(
            "keypoint_model_path",
            "scripts/models/Keypoints_yolo11x/weights/best.pt"
        )
        self.keypoint_model = YOLO(self.keypoint_model_path)

        # YOLO główny (players, goalkeeper, referee, ball)
        self.model = YOLO(self.model_path)

        # Annotatory
        self.ellipse_annotators = {
            "goalkeeper": sv.EllipseAnnotator(color=sv.Color.from_hex("#00FFFF"), thickness=2),
        }
        self.triangle_annotators = {
            "referee": sv.TriangleAnnotator(color=sv.Color.from_hex("#FF00FF")),
            "ball": sv.TriangleAnnotator(color=sv.Color.from_hex("#FFFF00")),
        }
        self.triangle_annotator_ball_retina = sv.TriangleAnnotator(color=sv.Color.from_hex("#FFA500"))
        self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.from_hex("#000000"))

        # Inicjalizacja RetinaNet
        try:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
            cfg.MODEL.RETINANET.NUM_CLASSES = 1
            cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
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

    def _detect_ball(self, frame, class_names):
        """Najpierw RetinaNet, jeśli brak detekcji, YOLO"""
        detections = None

        if self.predictor_retina:
            outputs = self.predictor_retina(frame)
            instances = outputs["instances"].to("cpu")
            boxes = instances.pred_boxes.tensor.numpy()
            scores = instances.scores.numpy()
            if len(boxes) > 0:
                detections = sv.Detections(
                    xyxy=boxes,
                    class_id=np.zeros(len(boxes), dtype=int),
                    confidence=scores
                )

        # Fallback YOLO, jeśli RetinaNet nic nie znalazł
        if detections is None or len(detections) == 0:
            results = self.model(frame)[0]
            detections = sv.Detections.from_ultralytics(results)
            # filtruj tylko klasę piłki
            mask = np.array([results.names[c] == "ball" for c in detections.class_id])
            detections = detections[mask] if np.any(mask) else None

        return detections

    @staticmethod
    def _draw_keypoints(frame, kp_results):
        if kp_results.boxes is not None and len(kp_results.boxes) > 0:
            for box in kp_results.boxes:
                xywh = box.xywh.cpu().numpy()[0]
                x, y = int(xywh[0]), int(xywh[1])
                cv2.circle(frame, (x, y), 6, (0, 255, 255), -1)
        return frame

    def run(self, video_path: str):
        self.video_path = video_path
        frames = Frames(self.video_path)
        tracker = Tracker(
            model_path=self.model_path,
            tracker_config=self.tracker_config,
            conf_threshold=self.confidence,
        )
        in2teams = In2Teams()
        pitch_config = SoccerPitchConfiguration()
        SCALE = 0.1
        view_transformer = ViewTransformer(pitch_config, scale=SCALE)
        heatmap_gen = HeatmapGenerator()

        results = tracker.track_video(self.video_path)
        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        out = cv2.VideoWriter(
            self.output_path + "_analyzed.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            frames.fps,
            (frames.width, frames.height),
        )

        logger.info("Starting video analysis...")

        for frame_idx, result in enumerate(results):
            frame = result.orig_img.copy()
            detections = sv.Detections.from_ultralytics(result)
            class_names = result.names
            kp_results = self.keypoint_model(frame, verbose=False)[0]

            frame = self._draw_keypoints(frame, kp_results)

            if len(detections) == 0:
                out.write(frame)
                continue

            # Przydzielanie drużyn
            player_mask = np.array([class_names[c] == "player" for c in detections.class_id])
            players = detections[player_mask]
            track_ids = detections.tracker_id if detections.tracker_id is not None else np.arange(len(detections))

            current_frame_players = []
            for det_idx, bbox in enumerate(players.xyxy):
                pid = int(track_ids[det_idx])
                team_id = in2teams.assign_player_to_team(frame, bbox, pid)
                color = sv.Color.from_hex("#0088FF") if team_id == 1 else sv.Color.from_hex("#FF3333")

                # Annotacja
                det_player = sv.Detections(
                    xyxy=np.array([bbox]), class_id=np.array([0]), tracker_id=np.array([pid])
                )
                frame = sv.EllipseAnnotator(color=color, thickness=2).annotate(frame, det_player)
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, f"Player {pid} | Team {team_id}", (x1, max(15, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.as_bgr(), 1, cv2.LINE_AA)

                x_foot = (x1 + x2) / 2
                y_foot = y2
                current_frame_players.append((x_foot, y_foot, team_id))

            # Transformacja i update heatmapy
            transformed_points = view_transformer.transform_points(
                player_detections=current_frame_players, keypoint_detections=kp_results
            )
            heatmap_gen.update(transformed_points)

            # Bramkarze i sędziowie
            for cls_name in ["goalkeeper", "referee"]:
                mask = np.array([class_names[c] == cls_name for c in detections.class_id])
                if np.any(mask):
                    det_cls = detections[mask]
                    annotator = self.ellipse_annotators["goalkeeper"] if cls_name == "goalkeeper" else self.triangle_annotators["referee"]
                    frame = annotator.annotate(frame, det_cls)

            # Piłka
            detections_ball = self._detect_ball(frame, class_names)
            if detections_ball is not None and len(detections_ball) > 0:
                frame = self.triangle_annotator_ball_retina.annotate(frame, detections_ball)

            out.write(frame)

        frames.release()
        out.release()
        cv2.destroyAllWindows()

        # Zapis heatmap
        logger.info("Generating heatmaps...")
        heatmap_gen.save_heatmaps(output_dir=os.path.dirname(self.output_path))
        logger.info(f"Result saved in: {self.output_path}_analyzed.mp4")


# import yaml
# import cv2
# import supervision as sv
# from ultralytics import YOLO
# from segmentation.frames import Frames
# from player_tracking.tracker import Tracker
# from player_tracking.clustering_assigner import In2Teams
# import numpy as np
# import os
# from logger.logger import logger
# from player_tracking.fast_team_assigner import FastTeamAssigner
# from player_tracking.heuristic_assigner import HeuristicTeamAssigner
# from player_tracking.embedding_assigner import EmbeddingTeamAssigner
#
# import torch
# from detectron2.detectron2.config import get_cfg
# from detectron2.detectron2.engine import DefaultPredictor
# from detectron2.detectron2 import model_zoo
#
# from pitch_keypoints_tracking import draw_pitch
# from pitch_keypoints_tracking.pitch_configuration import SoccerPitchConfiguration
# from pitch_keypoints_tracking.view_transformation import ViewTransformer
# from pitch_keypoints_tracking.team_heatmap import HeatmapGenerator
#
#
# class Analyze:
#     def __init__(self, config_path: str):
#         with open(config_path, "r") as f:
#             self.config = yaml.safe_load(f)
#
#         self.model_path = self.config["model_path"]
#         self.confidence = self.config.get("confidence", 0.45)
#         self.output_path = self.config["output_path"]
#         self.tracker_config = self.config.get("tracker_config", "bytetrack.yaml")
#
#         # Config dla RetinaNet - detekcja piłki
#         self.retina_weights = self.config.get("retina_weights", "scripts/models/RetinaNet/model_best.pth")
#         self.retina_conf_thresh = self.config.get("retina_conf_thresh", 0.35)
#         self.retina_min_size = self.config.get("retina_min_size", 1080)
#         self.retina_max_size = self.config.get("retina_max_size", 1920)
#
#         # ### --- NOWE: KONFIGURACJA MODELU KEYPOINTS ---
#         # Ścieżka do modelu boiska (dodaj w pliku config.yaml klucz 'keypoint_model_path')
#         self.keypoint_model_path = self.config.get("keypoint_model_path", "scripts/models/Keypoints_yolo11x/weights/best.pt")
#         self.keypoint_model = YOLO(self.keypoint_model_path)
#         # ### -------------------------------------------
#
#         # Inicjalizacja modelu YOLO
#         self.model = YOLO(self.model_path)
#
#         self.ellipse_annotators = {
#             "goalkeeper": sv.EllipseAnnotator(color=sv.Color.from_hex("#00FFFF"), thickness=2),
#         }
#
#         self.triangle_annotators = {
#             "referee": sv.TriangleAnnotator(color=sv.Color.from_hex("#FF00FF")),
#             "ball": sv.TriangleAnnotator(color=sv.Color.from_hex("#FFFF00")),
#         }
#
#         self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.from_hex("#000000"))
#
#         # Prediktor RetinaNet
#         try:
#             cfg = get_cfg()
#             cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_50_FPN_3x.yaml"))
#             cfg.MODEL.RETINANET.NUM_CLASSES = 1
#             cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "mps"
#             cfg.MODEL.WEIGHTS = self.retina_weights
#             cfg.MODEL.RETINANET.SCORE_THRESH_TEST = self.retina_conf_thresh
#
#             cfg.TEST.DETECTIONS_PER_IMAGE = 5
#             cfg.INPUT.MIN_SIZE_TEST = self.retina_min_size
#             cfg.INPUT.MAX_SIZE_TEST = self.retina_max_size
#             cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[6, 8, 10, 12, 16, 20, 24]]
#             cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 0.7, 1.0, 1.5]]
#
#             self.predictor_retina = DefaultPredictor(cfg)
#         except Exception as e:
#             logger.warning(f"RetinaNet initialization failed: {e}")
#             self.predictor_retina = None
#
#         # dodatkowy annotator dla predykcji RetinaNet (piłka) - pomarańczowy
#         self.triangle_annotator_ball_retina = sv.TriangleAnnotator(color=sv.Color.from_hex("#FFA500"))
#
#         self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.from_hex("#000000"))
#
#     def _predict_ball_with_retina(self, frame, class_names):
#         if self.predictor_retina is None:
#             return None
#
#         outputs = self.predictor_retina(frame)
#         instances = outputs["instances"].to("cpu")
#         pred_boxes = instances.pred_boxes.tensor.numpy()
#         scores = instances.scores.numpy()
#         pred_classes = instances.pred_classes.numpy()
#
#         ball_detections = []
#         for box, score, cls in zip(pred_boxes, scores, pred_classes):
#             if score >= self.retina_conf_thresh and class_names[cls] == "ball":
#                 ball_detections.append(box)
#
#         if len(ball_detections) == 0:
#             return None
#
#         ball_detections = np.array(ball_detections)
#         detections = sv.Detections(
#             xyxy=ball_detections,
#             class_id=np.zeros(len(ball_detections), dtype=int),
#             confidence=scores[:len(ball_detections)]
#         )
#         return detections
#
#     def run(self, video_path: str):  # In2Teams - KMeans przydzielanie druyn
#
#         VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
#             color=sv.Color.from_hex('#FFD700'),
#             text_color=sv.Color.from_hex('#FFFFFF'),
#             border_radius=5,
#             text_thickness=1,
#             text_scale=0.5,
#             text_padding=5,
#         )
#
#         self.video_path = video_path
#
#         # --- Inicjalizacja narzędzi ---
#         frames = Frames(self.video_path)
#         tracker = Tracker(
#             model_path=self.model_path,
#             tracker_config=self.tracker_config,
#             conf_threshold=self.confidence,
#         )
#         in2teams = In2Teams()
#
#         # ### --- NOWE: INICJALIZACJA NARZĘDZI HEATMAPY ---
#         pitch_config = SoccerPitchConfiguration()
#         SCALE = 0.1  # 1px = 10cm
#
#         view_transformer = ViewTransformer(pitch_config, scale=SCALE)
#         heatmap_gen = HeatmapGenerator()
#         # ### -------------------------------------------
#
#         results = tracker.track_video(self.video_path)
#         os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
#
#         out = cv2.VideoWriter(
#             self.output_path + "_analyzed.mp4",
#             cv2.VideoWriter_fourcc(*"mp4v"),
#             frames.fps,
#             (frames.width, frames.height),
#         )
#
#         logger.info("Starting video analyze...")
#
#         for frame_idx, result in enumerate(results):
#             frame = result.orig_img.copy()
#             detections = sv.Detections.from_ultralytics(result)
#             class_names = result.names
#
#             # --- do keypoint'ów
#             kp_results = self.keypoint_model(frame, verbose=False)[0]
#             keypoints_detections = sv.KeyPoints.from_ultralytics(kp_results)
#
#             # --- Pomiń, jeśli brak detekcji ---
#             if len(detections) == 0 or len(keypoints_detections) == 0:
#                 out.write(frame)
#                 continue
#
#             # --- Wybierz zawodników (bez bramkarzy i sędziów) ---
#             player_mask = np.array([class_names[c] == "player" for c in detections.class_id])
#             players = detections[player_mask]
#             track_ids = (
#                 detections.tracker_id if detections.tracker_id is not None else np.arange(len(detections))
#             )
#
#             # --- W pierwszej klatce ustal kolory drużyn ---
#             if frame_idx == 0 and len(players) > 2:
#                 logger.info("Team's color initialization...")
#                 in2teams.assign_color_to_team(frame, players)
#
#             # ### --- NOWE: LISTA POZYCJI W BIEŻĄCEJ KLATCE ---
#             current_frame_players = []
#             # ### -------------------------------------------
#
#             # --- Anotacje graczy z kolorami drużyn ---
#             for det_idx, bbox in enumerate(players.xyxy):
#                 player_id = int(track_ids[det_idx]) if track_ids is not None else det_idx
#                 team_id = in2teams.assign_player_to_team(frame, bbox, player_id)
#
#                 # Kolory drużyn (Team 1 = niebieski, Team 2 = czerwony)
#                 if team_id == 1:
#                     color = sv.Color.from_hex("#0088FF")  # niebieski
#                 elif team_id == 2:
#                     color = sv.Color.from_hex("#FF3333")  # czerwony
#                 else:
#                     color = sv.Color.from_hex("#00FF00")
#
#                 # naprawa: przekazujemy też class_id, żeby supervision nie rzucał błędem
#                 detections_for_player = sv.Detections(
#                     xyxy=np.array([bbox]),
#                     class_id=np.array([0]),
#                     tracker_id=np.array([player_id])
#                 )
#
#                 ellipse_annotator = sv.EllipseAnnotator(color=color, thickness=2)
#                 frame = ellipse_annotator.annotate(
#                     frame,
#                     detections_for_player
#                 )
#
#                 # etykieta gracza z ID i numerem drużyny
#                 label_text = f"Player {player_id} | Team {team_id}"
#                 x1, y1, x2, y2 = map(int, bbox)
#                 cv2.putText(
#                     frame,
#                     label_text,
#                     (x1, max(15, y1 - 10)),
#                     cv2.FONT_HERSHEY_SIMPLEX,
#                     0.5,
#                     color.as_bgr(),
#                     1,
#                     cv2.LINE_AA,
#                 )
#
#                 # ### --- NOWE: ZBIERANIE DANYCH DO HEATMAPY ---
#                 x_foot = (x1 + x2) / 2
#                 y_foot = y2
#                 current_frame_players.append((x_foot, y_foot, team_id))
#                 # ### ----------------------------------------
#
#             # ### --- NOWE: DETEKCJA BOISKA I AKTUALIZACJA HEATMAPY ---
#             # 1. Detekcja punktów kluczowych (boiska)
#             # -----------------------------------------
#             # DEBUG: Rysowanie wykrytych keypoints YOLO
#             # -----------------------------------------
#             if kp_results.boxes is not None and len(kp_results.boxes) > 0:
#                 for i, box in enumerate(kp_results.boxes):
#                     cls_id = int(box.cls.item())
#                     xywh = box.xywh.cpu().numpy()[0]
#                     x_center, y_center = int(xywh[0]), int(xywh[1])
#
#                     # rysowanie punktu
#                     cv2.circle(frame, (x_center, y_center), 6, (0, 255, 255), -1)
#
#                     # etykieta: klasa + pozycja
#                     cv2.putText(
#                         frame,
#                         f"KP {cls_id}",
#                         (x_center + 5, y_center - 5),
#                         cv2.FONT_HERSHEY_SIMPLEX,
#                         0.5,
#                         (0, 255, 255),
#                         1,
#                         cv2.LINE_AA
#                     )
#
#                 print("YOLO keypoints detected:", len(kp_results.boxes))
#                 print("Classes:", kp_results.boxes.cls.tolist())
#             else:
#                 print("YOLO keypoints detected: 0")
#
#             # 2. Transformacja perspektywy graczy z tej klatki
#             transformed_points = view_transformer.transform_points(
#                 player_detections=current_frame_players,
#                 keypoint_detections=kp_results
#             )
#
#             # 3. Dodanie punktów do generatora
#             heatmap_gen.update(transformed_points)
#             # ### -----------------------------------------------------
#
#             mask_goalkeeper = np.array([class_names[c] in ["goalkeeper"] for c in detections.class_id])
#             if np.any(mask_goalkeeper):
#                 detections_goalkeeper = detections[mask_goalkeeper]
#                 for cls in np.unique(detections_goalkeeper.class_id):
#                     mask_cls = detections_goalkeeper.class_id == cls
#                     frame = self.ellipse_annotators[class_names[cls]].annotate(
#                         frame, detections_goalkeeper[mask_cls]
#                     )
#
#             mask_triangle = np.array([class_names[c] in ["ball", "referee"] for c in detections.class_id])
#             if np.any(mask_triangle):
#                 detections_triangle = detections[mask_triangle]
#                 for cls in np.unique(detections_triangle.class_id):
#                     mask_cls = detections_triangle.class_id == cls
#                     frame = self.triangle_annotators[class_names[cls]].annotate(
#                         frame, detections_triangle[mask_cls]
#                     )
#
#                 # Piłka z modelem Retina
#                 detections_retina = self._predict_ball_with_retina(frame, class_names)
#                 if detections_retina is not None and len(detections_retina) > 0:
#                     frame = self.triangle_annotator_ball_retina.annotate(frame, detections_retina)
#
#                 out.write(frame)
#
#             if frame_idx == 25:
#                 break
#
#         frames.release()
#         out.release()
#         cv2.destroyAllWindows()
#
#         # ### --- NOWE: ZAPIS HEATMAP PO ZAKOŃCZENIU ANALIZY ---
#         logger.info("Generating heatmaps...")
#         output_dir = os.path.dirname(self.output_path)
#         heatmap_gen.save_heatmaps(output_dir=output_dir)
#         # ### --------------------------------------------------
#
#         logger.info(f"Result save in: {self.output_path}_analyzed.mp4")

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
