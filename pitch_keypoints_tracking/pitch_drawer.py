import cv2
import numpy as np
from .pitch_configuration import SoccerPitchConfiguration

def draw_pitch(config: SoccerPitchConfiguration, scale=0.1):
    """
    Rysuje boisko na podstawie konfiguracji Roboflow.
    scale=0.1 oznacza, że 1000 cm (10m) to 100 pikseli.
    """
    img_width = int(config.length * scale)
    img_height = int(config.width * scale)

    image = np.zeros((img_height, img_width, 3), dtype=np.uint8)
    image[:] = (50, 170, 50)  # Kolor trawy

    vertices = config.vertices

    # Rysowanie krawędzi zdefiniowanych w configu
    for start_idx, end_idx in config.edges:

        pt1 = vertices[start_idx - 1]
        pt2 = vertices[end_idx - 1]

        # Skalowanie do pikseli
        p1_px = (int(pt1[0] * scale), int(pt1[1] * scale))
        p2_px = (int(pt2[0] * scale), int(pt2[1] * scale))

        cv2.line(image, p1_px, p2_px, (255, 255, 255), 2)

    return image