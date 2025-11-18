import cv2
import numpy as np
from sklearn.cluster import KMeans
from logger.logger import logger

class In2Teams:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def dominant_color(self, crop_img):
        #wyznacza dominujący kolor w półgórnej części wykadrowanego obrazu zawodnika.
        crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
        pixels = crop.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, n_init=10, random_state=0)
        kmeans.fit(pixels)

        labels = kmeans.labels_
        clustered_image = labels.reshape(crop.shape[0], crop.shape[1])
        corner_clusters = [clustered_image[0, 0], clustered_image[0, -1], clustered_image[-1, 0], clustered_image[-1, -1]]
        non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
        player_cluster = 1 - non_player_cluster

        return kmeans.cluster_centers_[player_cluster]

    def assign_color_to_team(self, frame, players):
        #ustala kolory drużyn na podstawie pierwszych wykrytych zawodników.
        player_colors = []
        for bbox in players.xyxy[:10]:
            cropped_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
            if cropped_img.size == 0:
                continue
            player_color = self.dominant_color(cropped_img[0:int(cropped_img.shape[0]/2), :])
            player_colors.append(player_color)

        if len(player_colors) < 2:
            logger.warning("Not enough players to assign color")
            return

        kmeans = KMeans(n_clusters=2, init='k-means++', n_init=5, random_state=0)
        kmeans.fit(player_colors)

        self.kmeans = kmeans
        self.team_colors[1] = kmeans.cluster_centers_[0]
        self.team_colors[2] = kmeans.cluster_centers_[1]

    def assign_player_to_team(self, frame, bbox, player_id, role="player", players_xyxy=None, players_team_ids=None):
        """
        Przypisuje zawodnika lub bramkarza do drużyny.
        Bramkarz przypisywany jest do drużyny, której zawodnicy są najbliżej (średnia pozycja).
        """
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        cropped_img = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        if cropped_img.size == 0:
            logger.warning("Invalid image")
            return None

        if role == "player":
            # piłkarz – przypisanie po kolorze stroju
            player_color = self.dominant_color(cropped_img[0:int(cropped_img.shape[0]/2), :])
            if self.kmeans is None:
                return None
            team_id = int(self.kmeans.predict([player_color])[0]) + 1

        elif role == "goalkeeper":
            # bramkarz – przypisanie do drużyny na podstawie średniej pozycji zawodników
            if players_xyxy is None or players_team_ids is None:
                # fallback: przypisz po lewej/prawej połowie boiska
                mid_x = (bbox[0] + bbox[2]) / 2
                frame_mid = frame.shape[1] / 2
                team_id = 1 if mid_x < frame_mid else 2
            else:
                # oblicz środek bramkarza
                gk_center = np.array([(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2])
                team_centers = {}
                # oblicz średnią pozycję zawodników dla każdej drużyny
                for t in [1, 2]:
                    team_players = [np.array([(x[0]+x[2])/2, (x[1]+x[3])/2])
                                    for x, tid in zip(players_xyxy, players_team_ids) if tid == t]
                    if len(team_players) == 0:
                        team_centers[t] = np.array([gk_center[0], gk_center[1]])  # fallback
                    else:
                        team_centers[t] = np.mean(team_players, axis=0)
                # wybierz drużynę, której środek jest najbliżej bramkarza
                distances = {t: np.linalg.norm(gk_center - c) for t, c in team_centers.items()}
                team_id = min(distances, key=distances.get)

        else:
            # fallback
            team_id = None

        self.player_team_dict[player_id] = team_id
        logger.info("Player {} assigned to team {}".format(player_id, team_id))
        return team_id