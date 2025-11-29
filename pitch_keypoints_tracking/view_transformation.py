import numpy as np
import cv2
from pitch_keypoints_tracking.pitch_configuration import SoccerPitchConfiguration


class ViewTransformer:
    def __init__(self, config: SoccerPitchConfiguration = None):
        """
        View transformer używający keypointów boiska do rzutów top-down.
        """
        self.config = config
        self.homography = None

    def fit(self, src_points: np.ndarray, dst_points: np.ndarray):
        """
        Oblicza homografię na podstawie DOPASOWANYCH par punktów.

        Args:
            src_points: Punkty wykryte na obrazie (z detekcji YOLO/KeyPoints)
            dst_points: Odpowiadające im punkty na modelu boiska (z configu)
        """
        src_points = np.array(src_points, dtype=np.float32)
        dst_points = np.array(dst_points, dtype=np.float32)

        # Sprawdzenie czy mamy wystarczającą liczbę punktów (minimum 4 do homografii)
        if src_points.shape[0] < 4 or dst_points.shape[0] < 4:
            # print("[ViewTransformer] Za mało punktów do obliczenia homografii.")
            self.homography = None
            return

        # RANSAC do odporności na błędy detekcji
        self.homography, status = cv2.findHomography(src_points, dst_points, cv2.RANSAC)

        if self.homography is None:
            print("[ViewTransformer] Homografia nie została wyliczona.")

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Transformuje punkty graczy na widok top-down.
        points: Nx2 array [x, y]
        """
        if self.homography is None or len(points) == 0:
            return np.empty((0, 2))

        # Zamiana na współrzędne jednorodne [x, y, 1]
        points_hom = np.concatenate([points, np.ones((points.shape[0], 1))], axis=1)

        # Mnożenie macierzowe: H * P
        # (Używamy transpozycji .T, aby dopasować wymiary, a potem wracamy)
        transformed_hom = (self.homography @ points_hom.T).T

        # Normalizacja (podział przez 'z')
        # Zabezpieczenie przed dzieleniem przez zero (bardzo rzadkie, ale możliwe)
        z_values = transformed_hom[:, 2:3]
        z_values[z_values == 0] = 1e-6  # mała liczba zamiast zera

        transformed = transformed_hom[:, :2] / z_values
        return transformed