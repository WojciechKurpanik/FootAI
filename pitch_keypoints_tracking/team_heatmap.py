import numpy as np
import cv2
import os
import supervision as sv
from pitch_keypoints_tracking.pitch_configuration import SoccerPitchConfiguration


class HeatmapGenerator:
    def __init__(self,
                 config: SoccerPitchConfiguration = SoccerPitchConfiguration(),
                 radar_size=(700, 400),
                 heatmap_size=(200, 100),
                 num_cols=8,
                 num_rows=6):

        self.config = config
        self.radar_w, self.radar_h = radar_size
        self.heatmap_w, self.heatmap_h = heatmap_size

        self.COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']

        self.num_cols = num_cols
        self.num_rows = num_rows

        self.sector_map = np.zeros((self.num_rows, self.num_cols), dtype=np.int32)

    # -----------------------------------------------------------
    #  Stare API — zachowane
    # -----------------------------------------------------------

    def update_heatmap_from_xy(self, transformed_xy: np.ndarray):
        if transformed_xy.size == 0:
            return self.sector_map

        norm_x = np.clip(transformed_xy[:, 0] / self.config.length, 0, 1)
        norm_y = np.clip(transformed_xy[:, 1] / self.config.width, 0, 1)

        sx = (norm_x * self.num_cols).astype(int)
        sy = (norm_y * self.num_rows).astype(int)

        sx = np.clip(sx, 0, self.num_cols - 1)
        sy = np.clip(sy, 0, self.num_rows - 1)

        for x, y in zip(sx, sy):
            self.sector_map[y, x] += 1

        return self.sector_map

    def render_pitch(self):
        pitch = np.ones((self.radar_h, self.radar_w, 3), dtype=np.uint8) * 50

        vertices = np.array(self.config.vertices, dtype=np.float32)
        vertices[:, 0] = vertices[:, 0] / self.config.length * self.radar_w
        vertices[:, 1] = vertices[:, 1] / self.config.width * self.radar_h

        for edge in self.config.edges:
            pt1 = tuple(vertices[edge[0] - 1].astype(int))
            pt2 = tuple(vertices[edge[1] - 1].astype(int))
            cv2.line(pitch, pt1, pt2, (200, 200, 200), 2)

        return pitch

    # -----------------------------------------------------------
    #  Główna metoda aktualizująca
    # -----------------------------------------------------------

    def update_heatmap(self, detections: sv.Detections, keypoints: sv.KeyPoints):
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)

        if mask.sum() < 3:
            return self.sector_map

        src_pts = keypoints.xy[0][mask].astype(np.float32)
        dst_pts = np.array(self.config.vertices)[mask].astype(np.float32)

        transformer = sv.ViewTransformer(source=src_pts, target=dst_pts)

        xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_xy = transformer.transform_points(xy)

        norm_x = np.clip(transformed_xy[:, 0] / self.config.length, 0, 1)
        norm_y = np.clip(transformed_xy[:, 1] / self.config.width, 0, 1)

        sx = (norm_x * self.num_cols).astype(int)
        sy = (norm_y * self.num_rows).astype(int)

        sx = np.clip(sx, 0, self.num_cols - 1)
        sy = np.clip(sy, 0, self.num_rows - 1)

        for x, y in zip(sx, sy):
            self.sector_map[y, x] += 1

        return self.sector_map

    # -----------------------------------------------------------
    #  Render radaru (sektory + gracze)
    # -----------------------------------------------------------

    def render_heatmap(self, detections: sv.Detections, keypoints: sv.KeyPoints, color_lookup: np.ndarray):
        radar = self.render_pitch()
        cell_w = self.radar_w // self.num_cols
        cell_h = self.radar_h // self.num_rows

        # Normalizacja w sektorach
        max_val = self.sector_map.max() if self.sector_map.max() > 0 else 1
        norm = (self.sector_map / max_val * 180).astype(np.uint8)

        norm_color = cv2.applyColorMap(norm, cv2.COLORMAP_JET)

        # Nałożenie sektorów
        for y in range(self.num_rows):
            for x in range(self.num_cols):
                x1 = x * cell_w
                y1 = y * cell_h
                x2 = x1 + cell_w
                y2 = y1 + cell_h

                color = tuple(int(c) for c in norm_color[y, x])
                overlay = radar.copy()

                cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)

                radar = cv2.addWeighted(radar, 0.80, overlay, 0.20, 0)

        # --------------------------------------------------
        #  Rysowanie pozycji graczy (bez zmian)
        # --------------------------------------------------
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)

        if mask.sum() >= 3:
            src_pts = keypoints.xy[0][mask].astype(np.float32)
            dst_pts = np.array(self.config.vertices)[mask].astype(np.float32)
            transformer = sv.ViewTransformer(source=src_pts, target=dst_pts)

            xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
            transformed_xy = transformer.transform_points(xy)

            transformed_xy[:, 0] = transformed_xy[:, 0] / self.config.length * self.radar_w
            transformed_xy[:, 1] = transformed_xy[:, 1] / self.config.width * self.radar_h

            for i, team_id in enumerate(color_lookup):
                pt = tuple(transformed_xy[i].astype(int))
                color = tuple(int(c) for c in sv.Color.from_hex(self.COLORS[team_id]).as_bgr())
                cv2.circle(radar, pt, 6, color, -1)

        return radar

    # -----------------------------------------------------------
    #  Zapis heatmapy na pełnym boisku
    # -----------------------------------------------------------

    def save_heatmap_on_pitch(self, team_id: int, output_dir: str = "outputs"):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pitch_path = os.path.join(script_dir, "pitch.png")
        pitch_img = cv2.imread(pitch_path)
        output_path = os.path.join(output_dir, f"heatmap_team{team_id}.png")

        if pitch_img is None:
            raise FileNotFoundError(f"Pitch image not found at {pitch_path}")

        pitch_h, pitch_w = pitch_img.shape[:2]

        if self.sector_map.max() == 0:
            print("WARNING: Heatmap is empty. Saving original pitch.")
            cv2.imwrite(output_path, pitch_img)
            return

        max_val = self.sector_map.max()
        norm = (self.sector_map / max_val * 180).astype(np.uint8)

        heatmap_resized = cv2.resize(norm, (pitch_w, pitch_h), interpolation=cv2.INTER_NEAREST)
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (21, 21), 0)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        alpha = heatmap_resized.astype(float) / 255.0
        alpha = np.clip(alpha * 1.2, 0, 0.5)
        alpha = cv2.merge([alpha, alpha, alpha])

        overlay = pitch_img.astype(float) * (1 - alpha) + heatmap_color.astype(float) * alpha
        overlay = overlay.astype(np.uint8)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, overlay)
        print(f"Heatmap saved on pitch in: {output_path}")
