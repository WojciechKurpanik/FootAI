import torch
import torch.nn as nn
import torchvision.transforms as T
import torchvision.models as models
import numpy as np
from collections import deque
from logger.logger import logger
from sklearn.cluster import KMeans


class EmbeddingTeamAssigner:
    """
    Assignowanie drużyn na podstawie embeddings wizualnych graczy.
    - Oblicza embedding dla każdego gracza
    - Porównuje z centroidami drużyn
    - Wykorzystuje historię dla stabilizacji przypisania
    """

    def __init__(self, device='cuda', max_history=50):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.max_history = max_history
        self.team_centroids = {1: None, 2: None}  # centroid embeddings drużyn
        self.player_history = {}  # player_id -> deque[team_id]
        self.team_positions = {1: [], 2: []}  # pozycje graczy w każdej drużynie (dla bieżącej klatki)

        # model feature extractor
        self.model = models.resnet50(weights=True)
        self.model.fc = nn.Identity()  # odcinamy classifier
        self.model.eval().to(self.device)

        # transformacje obrazu
        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
        ])

    @torch.no_grad()
    def get_embedding(self, crop):
        """
        crop: np.ndarray (H,W,3)
        returns: np.ndarray 2048D embedding
        """
        x = self.transform(crop).unsqueeze(0).to(self.device)
        emb = self.model(x).squeeze(0).cpu().numpy()
        emb /= np.linalg.norm(emb) + 1e-6  # normalizacja L2
        return emb

    def initialize(self, frame, players, track_ids):
        """
        Pierwsza klatka: ustalenie centroidów drużyn
        Zakładamy, że mamy przynajmniej 2 graczy z każdej drużyny.
        """
        embeddings = []
        for bbox in players.xyxy:
            x1, y1, x2, y2 = map(int, bbox)
            crop = frame[y1:y2, x1:x2]
            embeddings.append(self.get_embedding(crop))

        embeddings = np.array(embeddings)

        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(embeddings)
        labels = kmeans.labels_

        # centroidy drużyn
        for i in range(2):
            self.team_centroids[i + 1] = np.mean(embeddings[labels == i], axis=0)

        # inicjalizacja historii
        for idx, tid in enumerate(labels):
            pid = int(track_ids[idx])
            self.player_history[pid] = deque([tid + 1], maxlen=self.max_history)

        logger.info("Initialized embedding centroids for both teams")

    def assign_player(self, frame, bbox, player_id):
        """
        Przypisuje gracza do drużyny na podstawie embeddings + history
        """
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        emb = self.get_embedding(crop)

        # odległość do centroidów drużyn
        dists = []
        for tid in [1, 2]:
            if self.team_centroids[tid] is None:
                dists.append(np.inf)
            else:
                dists.append(np.linalg.norm(emb - self.team_centroids[tid]))

        team_id = int(np.argmin(dists)) + 1

        # dodajemy do historii
        if player_id not in self.player_history:
            self.player_history[player_id] = deque(maxlen=self.max_history)
        self.player_history[player_id].append(team_id)

        # majority vote z ostatnich max_history ramek
        votes = list(self.player_history[player_id])
        final_team = max(set(votes), key=votes.count)

        # aktualizacja centroidu drużyny (prosty running average)
        alpha = 0.05
        self.team_centroids[final_team] = (1 - alpha) * self.team_centroids[final_team] + alpha * emb

        return final_team

    def clear_frame_positions(self):
        """
        Czyści pozycje graczy przed przetworzeniem nowej klatki.
        Wywołaj na początku każdej klatki.
        """
        self.team_positions = {1: [], 2: []}

    def update_team_positions(self, team_id, bbox):
        """
        Aktualizuje listę pozycji graczy w drużynie (dla bieżącej klatki).
        Wywołaj po assign_player dla każdego gracza w klatce.
        """
        x1, y1, x2, y2 = map(int, bbox)
        center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])
        self.team_positions[team_id].append(center)

    def assign_goalkeeper(self, bbox):
        """
        Przypisuje bramkarza do drużyny na podstawie odległości od centroidu drużyny.

        Args:
            bbox: bounding box bramkarza [x1, y1, x2, y2]

        Returns:
            team_id: int (1 lub 2) - drużyna do której przypisano bramkarza
        """
        x1, y1, x2, y2 = map(int, bbox)
        gk_center = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

        min_dist = np.inf
        assigned_team = 1

        for team_id in [1, 2]:
            if not self.team_positions[team_id]:
                continue

            # centroid pozycji graczy w drużynie
            positions = np.array(self.team_positions[team_id])
            team_centroid = np.mean(positions, axis=0)

            # odległość euklidesowa od bramkarza do centroidu drużyny
            dist = np.linalg.norm(gk_center - team_centroid)

            if dist < min_dist:
                min_dist = dist
                assigned_team = team_id

        logger.debug(f"Goalkeeper assigned to team {assigned_team} (distance: {min_dist:.2f})")
        return assigned_team