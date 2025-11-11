import numpy as np
import cv2
from collections import deque
from sklearn.cluster import KMeans

class HeuristicTeamAssigner:
    """
    Assignowanie drużyn z użyciem heurystyk:
    - ignoruje klatki z overlapami bboxów
    - głosowanie na podstawie 3 ostatnich próbek koloru ("best 2 of 3")
    """

    def __init__(self, max_history=5, overlap_threshold=0.2):
        self.team_colors = []
        self.player_history = {}  # player_id -> deque[team_id]
        self.overlap_threshold = overlap_threshold
        self.max_history = max_history

    def _extract_dominant_color(self, frame, bbox):
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return np.array([0, 0, 0])
        crop_small = cv2.resize(crop, (20, 40))
        return np.mean(crop_small.reshape(-1, 3), axis=0)

    def _iou(self, boxA, boxB):
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    def initialize(self, frame, players):
        colors = [self._extract_dominant_color(frame, bbox) for bbox in players.xyxy]
        if len(colors) < 2:
            return
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10).fit(colors)
        self.team_colors = [np.mean(np.array(colors)[kmeans.labels_ == i], axis=0) for i in range(2)]

    def assign_player_to_team(self, frame, bbox, player_id, players):
        # Pomijamy gracza, jeśli ma overlap z kimś innym
        for other_bbox in players.xyxy:
            if np.array_equal(bbox, other_bbox):
                continue
            if self._iou(bbox, other_bbox) > self.overlap_threshold:
                # nie przypisujemy tej klatki
                if player_id in self.player_history and len(self.player_history[player_id]) > 0:
                    # ostatnia znana drużyna (jeśli była)
                    return self.player_history[player_id][-1]
                return 0  # niepewność

        # jeśli brak overlapu — klasyczne przypisanie po kolorze
        color = self._extract_dominant_color(frame, bbox)
        dists = [np.linalg.norm(color - c) for c in self.team_colors]
        team_id = int(np.argmin(dists)) + 1

        # dodajemy do historii i głosujemy
        if player_id not in self.player_history:
            self.player_history[player_id] = deque(maxlen=self.max_history)
        self.player_history[player_id].append(team_id)

        # głosowanie większościowe
        votes = list(self.player_history[player_id])
        final_team = max(set(votes), key=votes.count)
        return final_team
