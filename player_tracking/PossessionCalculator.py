from collections import defaultdict
from typing import Dict, List, Tuple, Optional,NamedTuple
import numpy as np

class PossessionInfo(NamedTuple):
    team_id: Optional[int]
    player_id: Optional[int]

class PossessionCalculator:
    """
    Liczy posiadanie piłki dla każdej drużyny na podstawie odległości graczy od piłki.
    Gdy piłka leci (nikt nie jest blisko), posiadanie przypisywane jest ostatniej drużynie.
    """

    def __init__(self, distance_threshold: float = 60.0, fps: float = 25.0):
        self. distance_threshold = distance_threshold
        self.fps = fps
        self.possession_per_frame: Dict[int, Optional[int]] = {}
        self.frames_count = defaultdict(int)
        self. last_possessing_team: Optional[int] = None


    def _calculate_distance(self, point1: Tuple[float, float], bbox: np.ndarray) -> float:
        """Oblicza odległość euklidesową między punktem (piłka) a centrum bbox (gracz)"""
        ball_x, ball_y = point1
        x1, y1, x2, y2 = bbox
        player_x = (x1 + x2) / 2.0
        player_y = (y1 + y2) / 2.0
        return np.sqrt((ball_x - player_x)**2 + (ball_y - player_y)**2)

    def update(
        self,
        frame_idx: int,
        ball_position: Optional[Tuple[float, float]],
        players_bboxes: np.ndarray,
        team_ids: List[int],
        player_ids: List[int],
    ):


        if ball_position is None or len(players_bboxes) == 0:
            if self.last_possessing_team is not None:
                possession= PossessionInfo(self.last_possessing_team,None)
                self.possession_per_frame[frame_idx] = possession
                self.frames_count[self.last_possessing_team] += 1
            else:
                possession= PossessionInfo(None,None)
                self.possession_per_frame[frame_idx] = None
            return possession

        valid_indices = [i for i, tid in enumerate(team_ids) if tid in [1, 2]]
        if len(valid_indices) == 0:
            if self.last_possessing_team is not None:
                possession= PossessionInfo(self.last_possessing_team,None)
                self.possession_per_frame[frame_idx] = possession
                self.frames_count[self.last_possessing_team] += 1
            else:
                possession= PossessionInfo(None,None)
                self. possession_per_frame[frame_idx] = None
            return possession

        valid_bboxes = [players_bboxes[i] for i in valid_indices]
        valid_team_ids = [team_ids[i] for i in valid_indices]
        valid_player_ids = [player_ids[i] for i in valid_indices]


        distances = [self._calculate_distance(ball_position, bbox) for bbox in valid_bboxes]
        min_idx = int(np.argmin(distances))
        min_distance = distances[min_idx]

        if min_distance < self.distance_threshold:
            team_id = valid_team_ids[min_idx]
            player_id= valid_player_ids[min_idx]
            possession=PossessionInfo(team_id,player_id)
            self.possession_per_frame[frame_idx] = possession
            self.frames_count[team_id] += 1
            self.last_possessing_team = team_id
            return possession
        else:
            if self.last_possessing_team is not None:
                possession= PossessionInfo(self.last_possessing_team,None)
                self.possession_per_frame[frame_idx] = possession
                self.frames_count[self.last_possessing_team] += 1
            else:
                possession = PossessionInfo(None, None)
                self.possession_per_frame[frame_idx] = possession

            return possession

    def get_possession_percentage(self) -> Dict[int, float]:
        """Zwraca procent posiadania dla każdej drużyny."""
        total_frames = sum(self.frames_count.values())
        if total_frames == 0:
            return {1: 0.0, 2: 0.0}
        return {
            1: (self. frames_count.get(1, 0) / total_frames) * 100.0,
            2: (self.frames_count.get(2, 0) / total_frames) * 100.0
        }

    def get_possession_time(self) -> Dict[int, float]:
        """Zwraca czas posiadania dla każdej drużyny (w sekundach)."""
        return {
            1: self. frames_count.get(1, 0) / self.fps,
            2: self.frames_count.get(2, 0) / self.fps
        }