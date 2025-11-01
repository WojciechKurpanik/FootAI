import cv2
import numpy as np
from sklearn.cluster import KMeans
from logger.logger import logger

class In2Teams:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}

    def dominant_color(self, crop_img):
        #zwraca dominujący kolor w obrazie (HSV) gracza.
        if crop_img is None or crop_img.size == 0:
            return np.array([0, 0, 0])

        crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        pixels = crop.reshape(-1, 3)

        # jeśli bardzo mało pikseli — zwróć średnią
        if len(pixels) < 10:
            return np.mean(pixels, axis=0)

        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        kmeans.fit(pixels)
        labels = kmeans.labels_.reshape(crop.shape[:2])

        # wybieramy klastry w rogach — tło
        corner_clusters = [
            labels[0, 0], labels[0, -1],
            labels[-1, 0], labels[-1, -1]
        ]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        player_color = kmeans.cluster_centers_[player_cluster]
        return player_color

    def assign_color_to_team(self, frame, detections):
        """
        Uczy się kolorów dwóch drużyn z pierwszej klatki.
        detections: obiekt sv.Detections zawierający bboxy zawodników.
        """
        player_colors = []

        for bbox in detections.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]

            # bierzemy tylko górną połowę (koszulka)
            top_half = crop[0:int(crop.shape[0] / 2), :]
            color = self.dominant_color(top_half)
            player_colors.append(color)

        if len(player_colors) < 2:
            logger.warning("Not enough players detected")
            return

        # Uczymy się 2 klastrów = 2 drużyny
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

        logger.info("Team's colors were defined")

    def assign_player_to_team(self, frame, bbox, player_id):

        #przypisuje zawodnika do jednej z drużyn w oparciu o jego kolor.

        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]

        if crop.size == 0:
            return 0

        top_half = crop[0:int(crop.shape[0] / 2), :]
        player_color = self.dominant_color(top_half)

        if not hasattr(self, "kmeans"):
            logger.warning("Teams were not initialized yet")
            return 0

        # przypisanie do najbliższego centroidu (drużyny)
        team_id = int(self.kmeans.predict([player_color])[0]) + 1
        self.player_team_dict[player_id] = team_id
        logger.info(f"Player {player_id} was assigned to team {team_id}")
        return team_id
