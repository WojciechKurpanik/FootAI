import numpy as np
import cv2

class FastTeamAssigner:
    """
    Szybki assigner drużyn bazujący na kolorach dominujących i centroidach graczy.
    Obsługuje też przypisanie bramkarzy na podstawie najbliższego centroidu drużyny.
    """

    def __init__(self):
        self.team_colors = []
        self.team_centroids = []

    def _extract_dominant_color(self, frame, bbox, k=3):
        """
        Wyznacza dominujący kolor w obrębie bounding boxa (uśredniony klaster KMeans).
        """
        x1, y1, x2, y2 = map(int, bbox)
        player_crop = frame[y1:y2, x1:x2]
        if player_crop.size == 0:
            return np.array([0, 0, 0])

        # przeskalowanie dla wydajności
        small = cv2.resize(player_crop, (20, 40), interpolation=cv2.INTER_LINEAR)
        data = small.reshape((-1, 3)).astype(np.float32)

        # proste centroidy zamiast KMeans dla prędkości
        dominant = np.mean(data, axis=0)
        return dominant

    def initialize(self, frame, players):
        """
        Inicjalizacja drużyn – ustala dwa dominujące kolory i centroidy.
        """
        colors = []
        centroids = []

        for bbox in players.xyxy:
            colors.append(self._extract_dominant_color(frame, bbox))
            centroids.append([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])

        colors = np.array(colors)
        centroids = np.array(centroids)

        # prosty podział KMeans na 2 drużyny
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(colors)
        labels = kmeans.labels_

        self.team_colors = [np.mean(colors[labels == i], axis=0) for i in range(2)]
        self.team_centroids = [np.mean(centroids[labels == i], axis=0) for i in range(2)]

    def update(self, frame, players):
        """
        Aktualizacja centroidów drużyn na podstawie nowych pozycji graczy.
        """
        centroids = np.array([[(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2] for bbox in players.xyxy])

        if len(centroids) < 2:
            return

        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(centroids)
        labels = kmeans.labels_

        self.team_centroids = [np.mean(centroids[labels == i], axis=0) for i in range(2)]

    def assign_player_to_team(self, frame, bbox, player_id=None):
        """
        Przypisuje gracza do drużyny o kolorze najbardziej podobnym do jego stroju.
        """
        color = self._extract_dominant_color(frame, bbox)
        distances = [np.linalg.norm(color - team_color) for team_color in self.team_colors]
        return int(np.argmin(distances)) + 1

    def assign_goalkeeper_to_team(self, bbox):
        """
        Przypisuje bramkarza do najbliższego centroidu drużyny.
        """
        if not self.team_centroids:
            return 0

        g_centroid = np.array([(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2])
        distances = [np.linalg.norm(g_centroid - c) for c in self.team_centroids]
        return int(np.argmin(distances)) + 1
