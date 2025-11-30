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
from tracking.FastTeamAssigner import FastTeamAssigner
from tracking.HeuristicTeamAssigner import HeuristicTeamAssigner
from tracking.EmbeddingTeamAssigner import EmbeddingTeamAssigner
from tracking.BallInterpolator import BallInterpolator
from tracking.PossesionCalculator import PossessionCalculator

# detectron2
import torch
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2 import model_zoo

class Analyze:
    def __init__(self, config_path: str):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model_path = self.config["model_path"]
        self.confidence = self.config.get("confidence", 0.45)
        self.output_path = self.config["output_path"]
        self.tracker_config = self.config.get("tracker_config", "bytetrack.yaml")

        #Config dla RetinaNet - detekcja piłki
        self.retina_weights = self.config.get("retina_weights", "scripts/models/RetinaNet/model_final1.pth")
        self.retina_conf_thresh = self.config.get("retina_conf_thresh", 0.35)
        self.retina_min_size = self.config.get("retina_min_size", 1080)
        self.retina_max_size = self.config.get("retina_max_size", 1920)

        # Inicjalizacja modelu YOLO
        self.model = YOLO(self.model_path)

        self.ellipse_annotators = {
            "goalkeeper": sv.EllipseAnnotator(color=sv.Color.from_hex("#00FFFF"), thickness=2),
        }

        self.triangle_annotators = {
            "referee": sv.TriangleAnnotator(color=sv.Color.from_hex("#FF00FF")),
            "ball": sv.TriangleAnnotator(color=sv.Color.from_hex("#FFFF00")),
        }

        self.label_annotator = sv.LabelAnnotator(text_color=sv.Color.from_hex("#000000"))

        #Prediktor RetinaNet
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
            # domyślne anchor'y można nadpisać w configie
            cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[6, 8, 10, 12, 16, 20, 24]]
            cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 0.7, 1.0, 1.5]]

            self.predictor_retina = DefaultPredictor(cfg)
        except Exception as e:
            logger.warning(f"RetinaNet initialization failed: {e}")
            self.predictor_retina = None

        # dodatkowy annotator dla predykcji RetinaNet (piłka) - pomarańczowy
        self.triangle_annotator_ball_retina = sv.TriangleAnnotator(color=sv.Color.from_hex("#FFA500"))

        #dla ball interpolation - brązowy
        #self.triangle_annotator_ball_interpolated = sv.TriangleAnnotator(color=sv.Color.from_hex("#8B4513"))

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
    # def run(self, video_path: str):  #In2Teams - KMeans przydzielanie druyn
    #     self.video_path = video_path
    #
    #     # --- Inicjalizacja narzędzi ---
    #     frames = Frames(self.video_path)
    #     tracker = Tracker(
    #         model_path=self.model_path,
    #         tracker_config=self.tracker_config,
    #         conf_threshold=self.confidence,
    #     )
    #     in2teams = In2Teams()
    #
    #     results = tracker.track_video(self.video_path)
    #     os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
    #
    #     out = cv2.VideoWriter(
    #         self.output_path + "_analyzed.mp4",
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
    #         # --- Pomiń, jeśli brak detekcji ---
    #         if len(detections) == 0:
    #             out.write(frame)
    #             continue
    #
    #         # --- Wybierz zawodników (bez bramkarzy i sędziów) ---
    #         player_mask = np.array([class_names[c] == "player" for c in detections.class_id])
    #         players = detections[player_mask]
    #         track_ids = (
    #             detections.tracker_id if detections.tracker_id is not None else np.arange(len(detections))
    #         )
    #
    #         # --- W pierwszej klatce ustal kolory drużyn ---
    #         if frame_idx == 0 and len(players) > 2:
    #             logger.info("Team's color initialization...")
    #             in2teams.assign_color_to_team(frame, players)
    #
    #         # --- Anotacje graczy z kolorami drużyn ---
    #         for det_idx, bbox in enumerate(players.xyxy):
    #             player_id = int(track_ids[det_idx]) if track_ids is not None else det_idx
    #             team_id = in2teams.assign_player_to_team(frame, bbox, player_id)
    #
    #             # Kolory drużyn (Team 1 = niebieski, Team 2 = czerwony)
    #             if team_id == 1:
    #                 color = sv.Color.from_hex("#0088FF")  # niebieski
    #             elif team_id == 2:
    #                 color = sv.Color.from_hex("#FF3333")  # czerwony
    #             else:
    #                 color = sv.Color.from_hex("#00FF00")
    #
    #             # naprawa: przekazujemy też class_id, żeby supervision nie rzucał błędem
    #             detections_for_player = sv.Detections(
    #                 xyxy=np.array([bbox]),
    #                 class_id=np.array([0]),  # placeholder (np. 0)
    #                 tracker_id=np.array([player_id])
    #             )
    #
    #             # ustawiamy color_lookup na NONE, by wymusić użycie naszego koloru
    #             ellipse_annotator = sv.EllipseAnnotator(color=color, thickness=2)
    #             frame = ellipse_annotator.annotate(
    #                 frame,
    #                 detections_for_player
    #             )
    #
    #             # etykieta gracza z ID i numerem drużyny
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
    #
    #         mask_goalkeeper = np.array([class_names[c] in ["goalkeeper"] for c in detections.class_id])
    #         if np.any(mask_goalkeeper):
    #             detections_goalkeeper = detections[mask_goalkeeper]
    #             for cls in np.unique(detections_goalkeeper.class_id):
    #                 mask_cls = detections_goalkeeper.class_id == cls
    #                 frame = self.ellipse_annotators[class_names[cls]].annotate(
    #                     frame, detections_goalkeeper[mask_cls]
    #                 )
    #
    #         mask_triangle = np.array([class_names[c] in ["ball", "referee"] for c in detections.class_id])
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
    #     logger.info(f"Result save in: {self.output_path}_analyzed.mp4")

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
    #                 team_id = team_assigner.assign_goalkeeper_to_team(bbox)  #  nowa metoda
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
    #     ball_interp = BallInterpolator(smoothing=3, bbox_size=28)
    #     possession_calc = PossessionCalculator(distance_threshold=80.0, fps=frames.fps)
    #
    #     frames_buffer = []
    #     ball_detected_frames = set()
    #
    #     for frame_idx, result in enumerate(results):
    #         frame = result.orig_img.copy()
    #         detections = sv.Detections.from_ultralytics(result)
    #         class_names = result.names
    #
    #         if len(detections) == 0:
    #             frames_buffer.append(frame)
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
    #         # zbierz pozycje piłki PRZED pętlą po graczach
    #         ball_position = None
    #         ball_mask = np.array([class_names[c] == "ball" for c in detections.class_id])
    #         if np.any(ball_mask):
    #             ball_bbox = detections[ball_mask].xyxy[0]
    #             ball_position = ((ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2)
    #
    #         # zbierz graczy z drużynami
    #         player_bboxes = []
    #         player_teams = []
    #
    #         # --- Anotacje graczy z kolorami drużyn ---
    #         for det_idx, bbox in enumerate(players.xyxy):
    #             player_id = int(track_ids[det_idx]) if track_ids is not None else det_idx
    #             team_id = team_assigner.assign_player_to_team(frame, bbox, player_id, players)
    #
    #             # dodaj do list dla possession_calc
    #             player_bboxes.append(bbox)
    #             player_teams.append(team_id)
    #
    #             # Kolory drużyn
    #             if team_id == 1:
    #                 color = sv.Color.from_hex("#0088FF")
    #             elif team_id == 2:
    #                 color = sv.Color.from_hex("#FF3333")
    #             else:
    #                 color = sv.Color.from_hex("#00FF00")
    #
    #             detections_for_player = sv.Detections(
    #                 xyxy=np.array([bbox]),
    #                 class_id=np.array([0]),
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
    #         # aktualizuj posiadanie PO zebraniu wszystkich graczy
    #         if len(player_bboxes) > 0:
    #             possession_calc.update(frame_idx, ball_position, np.array(player_bboxes), player_teams)
    #
    #         # wyświetl posiadanie w czasie rzeczywistym (prawy górny róg)
    #         possession_pct = possession_calc.get_possession_percentage()
    #         team1_pct = possession_pct.get(1, 0)
    #         team2_pct = possession_pct.get(2, 0)
    #
    #         # tło dla tekstu (półprzezroczyste)
    #         overlay = frame.copy()
    #         cv2.rectangle(overlay, (frame.shape[1] - 300, 10), (frame.shape[1] - 10, 100), (0, 0, 0), -1)
    #         frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
    #
    #         # tekst posiadania
    #         cv2.putText(frame, f"Team 1: {team1_pct:.1f}%", (frame.shape[1] - 290, 40),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 136, 0), 2)
    #         cv2.putText(frame, f"Team 2: {team2_pct:.1f}%", (frame.shape[1] - 290, 75),
    #                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (51, 51, 255), 2)
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
    #         # dodaj YOLO detekcje piłki do interpolatora
    #         if np.any(ball_mask):
    #             ball_detected_frames.add(frame_idx)
    #             detections_ball = detections[ball_mask]
    #             for bbox in detections_ball.xyxy:
    #                 ball_interp.add_detection(frame_idx, tuple(map(float, bbox)))
    #
    #         # Piłka z modelem Retina
    #         detections_retina = self._predict_ball_with_retina(frame, class_names)
    #         if detections_retina is not None and len(detections_retina) > 0:
    #             ball_detected_frames.add(frame_idx)
    #             frame = self.triangle_annotator_ball_retina.annotate(frame, detections_retina)
    #             for bbox in detections_retina.xyxy:
    #                 ball_interp.add_detection(frame_idx, tuple(map(float, bbox)))
    #
    #         frames_buffer.append(frame)
    #
    #     # --- PO PĘTLI: Interpolacja i zapis ---
    #     logger.info("Interpolating ball positions...")
    #     positions = ball_interp.interpolate()
    #
    #     # wyświetl finalne statystyki w logach
    #     possession_pct = possession_calc.get_possession_percentage()
    #     possession_time = possession_calc.get_possession_time()
    #     logger.info(f"Team 1 possession: {possession_pct.get(1, 0):.1f}% ({possession_time.get(1, 0):.1f}s)")
    #     logger.info(f"Team 2 possession: {possession_pct.get(2, 0):.1f}% ({possession_time.get(2, 0):.1f}s)")
    #
    #     for f_idx, frame in enumerate(frames_buffer):
    #         # brązowe bboxy tylko dla klatek bez detekcji YOLO/Retina
    #         if f_idx in positions and f_idx not in ball_detected_frames:
    #             center = positions[f_idx]
    #             bbox = ball_interp.get_bbox(center)
    #             detections_obj = sv.Detections(
    #                 xyxy=np.array([bbox]),
    #                 class_id=np.array([0]),
    #                 confidence=np.array([1.0])
    #             )
    #             frame = self.triangle_annotator_ball_interpolated.annotate(frame, detections_obj)
    #         out.write(frame)
    #
    #     frames.release()
    #     out.release()
    #     cv2.destroyAllWindows()
    #     logger.info(f"Result saved in: {self.output_path}_analyzed_heuristic.mp4")

    def run(self, video_path: str): #Embeddings
        self.video_path = video_path

        frames = Frames(self.video_path)
        tracker = Tracker(
            model_path=self.model_path,
            tracker_config=self.tracker_config,
            conf_threshold=self.confidence,
        )

        team_assigner = EmbeddingTeamAssigner(device='cuda', max_history=5)

        results = tracker.track_video(self.video_path)

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        out = cv2.VideoWriter(
            self.output_path + "_analyzed_embeddings.mp4",
            cv2.VideoWriter_fourcc(*"mp4v"),
            frames.fps,
            (frames.width, frames.height),
        )

        logger.info("Starting video analyze...")



        possession_calc = PossessionCalculator(distance_threshold=60.0, fps=25)

        ball_detected_frames = set()

        for frame_idx, result in enumerate(results):
            frame = result.orig_img.copy()
            detections = sv.Detections.from_ultralytics(result)
            class_names = result.names

            if len(detections) == 0:
                out.write(frame)
                continue

            # --- Wybierz zawodników (bez bramkarzy i sędziów) ---
            player_mask = np.array([class_names[c] == "player" for c in detections.class_id])
            players = detections[player_mask]
            track_ids = detections.tracker_id if detections.tracker_id is not None else np.arange(len(players))

            # --- Inicjalizacja embeddings drużyn (pierwsza klatka) ---
            if frame_idx == 0 and len(players) > 2:
                team_assigner.initialize(frame, players, track_ids)


            # DODANE: zbierz graczy z drużynami
            player_bboxes = []
            player_teams = []

            # ZMIANA: Pobierz pozycję piłki TYLKO z RetinaNet (nie YOLO)
            ball_position = None
            detections_retina = self._predict_ball_with_retina(frame, ["ball"])
            if detections_retina is not None and len(detections_retina) > 0:
                ball_bbox = detections_retina.xyxy[0]
                ball_position = ((ball_bbox[0] + ball_bbox[2]) / 2, (ball_bbox[1] + ball_bbox[3]) / 2)
                # ZMIANA: usunięto dodawanie do ball_interp

            # --- Anotacje graczy ---
            for det_idx, bbox in enumerate(players.xyxy):
                player_id = int(track_ids[det_idx]) if track_ids is not None else det_idx
                team_id = team_assigner.assign_player(frame, bbox, player_id)

                # DODANE: dodaj do list dla possession_calc
                player_bboxes.append(bbox)
                player_teams.append(team_id)

                # Kolory drużyn
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

                label_text = f"Player {player_id} | Team {team_id}"
                x1, y1, x2, y2 = map(int, bbox)
                cv2.putText(frame, label_text, (x1, max(15, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color.as_bgr(), 1, cv2.LINE_AA)

            # DODANE: aktualizuj posiadanie PO zebraniu wszystkich graczy
            if len(player_bboxes) > 0:
                possession_calc.update(frame_idx, ball_position, np.array(player_bboxes), player_teams)

            # DODANE: wyświetl posiadanie w czasie rzeczywistym (prawy górny róg)
            frame = self.possesion_ui(frame, possession_calc)

            # --- Bramkarze ---
            mask_goalkeeper = np.array([class_names[c] == "goalkeeper" for c in detections.class_id])
            if np.any(mask_goalkeeper):
                detections_goalkeeper = detections[mask_goalkeeper]
                for cls in np.unique(detections_goalkeeper.class_id):
                    mask_cls = detections_goalkeeper.class_id == cls
                    frame = self.ellipse_annotators[class_names[cls]].annotate(
                        frame, detections_goalkeeper[mask_cls]
                    )

            # --- Sędzia ---
            mask_referee = np.array([class_names[c] == "referee" for c in detections.class_id])
            if np.any(mask_referee):
                detections_referee = detections[mask_referee]
                frame = self.triangle_annotators["referee"].annotate(frame, detections_referee)



            # DODANE: Piłka z modelem Retina
            if detections_retina is not None and len(detections_retina) > 0:
                frame = self.triangle_annotator_ball_retina.annotate(frame, detections_retina)

            out.write(frame)


        # wyświetl finalne statystyki w logach
        possession_pct = possession_calc.get_possession_percentage()
        possession_time = possession_calc.get_possession_time()
        logger.info(f"Team 1 possession: {possession_pct.get(1, 0):.1f}% ({possession_time.get(1, 0):.1f}s)")
        logger.info(f"Team 2 possession: {possession_pct.get(2, 0):.1f}% ({possession_time.get(2, 0):.1f}s)")


        frames.release()
        out.release()
        cv2.destroyAllWindows()
        logger.info(f"Result saved in: {self.output_path}_analyzed_embeddings.mp4")
