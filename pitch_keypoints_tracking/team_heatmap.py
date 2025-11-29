import numpy as np
import cv2
import os
import supervision as sv
from pitch_keypoints_tracking.pitch_configuration import SoccerPitchConfiguration

class HeatmapGenerator:
    def __init__(self, config: SoccerPitchConfiguration = SoccerPitchConfiguration(),
                 radar_size=(700, 400), heatmap_size=(200, 100)):
        self.config = config
        self.radar_w, self.radar_h = radar_size
        self.heatmap_w, self.heatmap_h = heatmap_size
        self.COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
        self.heatmap = np.zeros((self.heatmap_h, self.heatmap_w), dtype=np.float32)

    def update_heatmap_from_xy(self, transformed_xy: np.ndarray):
        if transformed_xy.size == 0:
            return self.heatmap

        # Skalowanie: (pozycja_w_cm / dlugosc_boiska_cm * szerokosc_obrazka_heatmapy)
        xy_scaled = np.zeros_like(transformed_xy, dtype=int)

        # Oś X
        xy_scaled[:, 0] = np.clip((transformed_xy[:, 0] / self.config.length * self.heatmap_w).astype(int), 0,
                                  self.heatmap_w - 1)
        # Oś Y
        xy_scaled[:, 1] = np.clip((transformed_xy[:, 1] / self.config.width * self.heatmap_h).astype(int), 0,
                                  self.heatmap_h - 1)

        for pt in xy_scaled:
            self.heatmap[pt[1], pt[0]] += 1  # pt[1] to wiersz (y), pt[0] to kolumna (x)

        return self.heatmap

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

    def update_heatmap(self, detections: sv.Detections, keypoints: sv.KeyPoints):
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        if mask.sum() < 3:
            return self.heatmap

        src_pts = keypoints.xy[0][mask].astype(np.float32)
        dst_pts = np.array(self.config.vertices)[mask].astype(np.float32)
        transformer = sv.ViewTransformer(source=src_pts, target=dst_pts)

        xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_xy = transformer.transform_points(xy)

        transformed_xy[:, 0] = np.clip((transformed_xy[:, 0] / self.config.length * self.heatmap_w).astype(int), 0, self.heatmap_w - 1)
        transformed_xy[:, 1] = np.clip((transformed_xy[:, 1] / self.config.width * self.heatmap_h).astype(int), 0, self.heatmap_h - 1)

        for pt in transformed_xy:
            self.heatmap[pt[1], pt[0]] += 1

        return self.heatmap

    def render_heatmap(self, detections: sv.Detections, keypoints: sv.KeyPoints, color_lookup: np.ndarray):
        mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
        if mask.sum() < 3:
            return np.zeros((self.radar_h, self.radar_w, 3), dtype=np.uint8)

        src_pts = keypoints.xy[0][mask].astype(np.float32)
        dst_pts = np.array(self.config.vertices)[mask].astype(np.float32)
        transformer = sv.ViewTransformer(source=src_pts, target=dst_pts)

        xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        transformed_xy = transformer.transform_points(xy)

        transformed_xy[:, 0] = transformed_xy[:, 0] / self.config.length * self.radar_w
        transformed_xy[:, 1] = transformed_xy[:, 1] / self.config.width * self.radar_h

        radar = self.render_pitch()

        # Dodaj heatmapę
        heatmap_resized = cv2.resize(self.heatmap, (self.radar_w, self.radar_h))
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)
        radar = cv2.addWeighted(radar, 0.5, heatmap_color, 0.5, 0)

        for i, team_id in enumerate(color_lookup):
            pt = tuple(transformed_xy[i].astype(int))
            color = tuple(int(c) for c in sv.Color.from_hex(self.COLORS[team_id]).as_bgr())
            cv2.circle(radar, pt, 6, color, -1)

        return radar

    def save_heatmap_on_pitch(self, output_path: str = "outputs/heatmap_on_pitch.png"):
        """
        Zapisuje heatmapę zawodników na obrazie boiska z użyciem maski przezroczystości.
        """
        script_dir = os.path.dirname(os.path.abspath(__file__))
        pitch_path = os.path.join(script_dir, "pitch.png")
        pitch_img = cv2.imread(pitch_path)

        if pitch_img is None:
            # Fallback jeśli nie ma pliku - czarne tło o wymiarach z configu (skalowane)
            # Ale zakładamy, że plik jest, skoro widziałeś niebieskie tło.
            raise FileNotFoundError(f"Pitch image not found at {pitch_path}")

        pitch_h, pitch_w = pitch_img.shape[:2]

        # 1. Sprawdź czy mamy jakiekolwiek dane
        if self.heatmap.max() == 0:
            print("WARNING: Heatmap is empty (max value is 0). Saving original pitch.")
            cv2.imwrite(output_path, pitch_img)
            return

        # 2. Normalizacja i skalowanie heatmapy do wymiarów obrazu boiska
        heatmap_norm = self.heatmap / self.heatmap.max()
        heatmap_norm = (heatmap_norm * 255).astype(np.uint8)

        # Resize heatmapy do wielkości obrazka pitch.png
        heatmap_resized = cv2.resize(heatmap_norm, (pitch_w, pitch_h))

        # Rozmycie, aby punkty wyglądały jak chmury ciepła
        heatmap_resized = cv2.GaussianBlur(heatmap_resized, (21, 21), 0)

        # 3. Generowanie kolorów (JET: 0=Blue, 255=Red)
        heatmap_color = cv2.applyColorMap(heatmap_resized, cv2.COLORMAP_JET)

        # 4. KLUCZOWE: Maska przezroczystości (Alpha Channel)
        # Tworzymy maskę na podstawie intensywności.
        # Im wyższa wartość w heatmap_resized, tym bardziej widoczny kolor.
        # Dzielimy przez 255, żeby mieć zakres 0.0 - 1.0
        alpha = heatmap_resized.astype(float) / 255.0

        # Opcjonalnie: Zwiększamy "siłę" widoczności, ale przycinamy do 1.0
        # Mnożnik (np. 1.5) sprawia, że słabsze punkty są lepiej widoczne
        alpha = np.clip(alpha * 1.5, 0, 0.8)  # 0.8 to max opacity (żeby nie zakryć linii boiska całkowicie)

        # Rozszerzamy alpha do 3 kanałów (BGR)
        alpha = cv2.merge([alpha, alpha, alpha])

        # 5. Mieszanie obrazów
        # Wzór: Output = Pitch * (1 - alpha) + Heatmap * alpha
        # Tam gdzie alpha=0 (brak graczy), widzimy tylko Pitch.
        pitch_float = pitch_img.astype(float)
        heatmap_float = heatmap_color.astype(float)

        overlay = pitch_float * (1.0 - alpha) + heatmap_float * alpha
        overlay = overlay.astype(np.uint8)

        # Zapis do pliku
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, overlay)
        print(f"Heatmap saved on pitch in: {output_path}")

