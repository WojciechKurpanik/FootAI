from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np

from player_tracking.PossessionCalculator import PossessionCalculator


class PassType(Enum):
    SUCCESSFUL = "successful"
    INTERCEPTED = "intercepted"


class PassState(Enum):
    NO_POSSESSION = "no_possession"
    PLAYER_HAS_BALL = "player_has_ball"
    BALL_IN_FLIGHT = "ball_in_flight"
    WAITING_CONFIRMATION = "waiting_confirmation"


@dataclass
class PassEvent:
    start_frame: int
    end_frame: int
    from_player_id: int
    to_player_id: int
    from_team_id: int
    to_team_id: int
    pass_type: PassType
    flight_frames: int


class PassDetector:
    """
    Robust pass detector:
    - Toleruje brak detekcji piłki podczas potwierdzania
    - Używa "okna czasowego" zamiast ciągłych klatek
    """

    def __init__(
        self,
        possession_calculator: PossessionCalculator,
        min_confirm_frames: int = 3,       # Ile razy gracz musi być widziany z piłką
        confirm_window_frames: int = 10,   # W jakim oknie czasowym
        max_flight_frames: int = 75,
    ):
        self. possession_calc = possession_calculator
        self.min_confirm_frames = min_confirm_frames
        self.confirm_window_frames = confirm_window_frames
        self.max_flight_frames = max_flight_frames

        self.passes: List[PassEvent] = []
        self.state = PassState.NO_POSSESSION

        # Lot piłki
        self.flight_start_frame: int = 0
        self.flight_from_player: Optional[int] = None
        self.flight_from_team: Optional[int] = None

        # Potwierdzenie odbioru - historia w oknie
        self.candidate_player: Optional[int] = None
        self.candidate_team: Optional[int] = None
        self.candidate_start_frame: int = 0
        self.candidate_detections: List[int] = []  # Klatki gdy widziany z piłką

        # Aktualny posiadacz
        self.current_player: Optional[int] = None
        self.current_team: Optional[int] = None

    def _count_detections_in_window(self, current_frame: int) -> int:
        """Ile razy kandydat był widziany z piłką w ostatnim oknie."""
        window_start = current_frame - self. confirm_window_frames
        return sum(1 for f in self.candidate_detections if f >= window_start)

    def update(
        self,
        frame_idx: int,
        ball_position: Optional[Tuple[float, float]],
        players_bboxes: np.ndarray,
        team_ids: List[int],
        player_ids: List[int]
    ) -> Optional[PassEvent]:

        possession = self.possession_calc.update(
            frame_idx, ball_position, players_bboxes, team_ids, player_ids
        )

        player_id = possession.player_id
        team_id = possession.team_id
        pass_event = None

        # === MASZYNA STANÓW ===

        if self.state == PassState.NO_POSSESSION:
            if player_id is not None:
                self._start_confirmation(player_id, team_id, frame_idx)

        elif self.state == PassState. PLAYER_HAS_BALL:
            if player_id is None:
                # Piłka opuściła gracza → start lotu
                self._start_flight(frame_idx)

            elif player_id != self.current_player:
                # Bezpośrednie przejęcie → lot + potwierdzenie
                self._start_flight(frame_idx)
                self._start_confirmation(player_id, team_id, frame_idx)

        elif self. state == PassState. BALL_IN_FLIGHT:
            flight_duration = frame_idx - self.flight_start_frame

            if player_id is not None:
                # Ktoś blisko piłki → zacznij potwierdzać
                self._start_confirmation(player_id, team_id, frame_idx)

            elif flight_duration > self.max_flight_frames:
                self._reset()

        elif self.state == PassState.WAITING_CONFIRMATION:
            window_duration = frame_idx - self. candidate_start_frame

            if player_id == self.candidate_player:
                # Ten sam gracz z piłką - dodaj do historii
                self. candidate_detections. append(frame_idx)

                detections = self._count_detections_in_window(frame_idx)
                if detections >= self.min_confirm_frames:
                    # POTWIERDZONY!
                    pass_event = self._confirm_pass(frame_idx)

            elif player_id is not None and player_id != self.candidate_player:
                # Inny gracz - może to on jest odbiorcą?
                # Sprawdź czy poprzedni kandydat miał wystarczająco
                detections = self._count_detections_in_window(frame_idx)
                if detections >= self. min_confirm_frames:
                    pass_event = self._confirm_pass(frame_idx)
                    # I zacznij potwierdzać nowego
                    self._start_flight(frame_idx)
                    self._start_confirmation(player_id, team_id, frame_idx)
                else:
                    # Poprzedni nie miał wystarczająco - zamień kandydata
                    self._start_confirmation(player_id, team_id, frame_idx)

            elif player_id is None:
                # Piłka zniknęła - ale NIE resetuj!
                # Sprawdź czy nie przekroczono okna
                if window_duration > self.confirm_window_frames:
                    detections = self._count_detections_in_window(frame_idx)
                    if detections >= self.min_confirm_frames:
                        pass_event = self._confirm_pass(frame_idx)
                    else:
                        # Wróć do lotu
                        self. state = PassState. BALL_IN_FLIGHT

        return pass_event

    def _start_flight(self, frame_idx: int):
        self.state = PassState. BALL_IN_FLIGHT
        self. flight_start_frame = frame_idx
        self.flight_from_player = self.current_player
        self.flight_from_team = self.current_team

    def _start_confirmation(self, player_id: int, team_id: int, frame_idx: int):
        self.state = PassState. WAITING_CONFIRMATION
        self.candidate_player = player_id
        self.candidate_team = team_id
        self. candidate_start_frame = frame_idx
        self.candidate_detections = [frame_idx]

    def _confirm_pass(self, frame_idx: int) -> Optional[PassEvent]:
        pass_event = None

        if self. flight_from_player is not None and self. flight_from_player != self.candidate_player:
            pass_type = (
                PassType.SUCCESSFUL
                if self.candidate_team == self.flight_from_team
                else PassType. INTERCEPTED
            )

            flight_frames = self.candidate_start_frame - self.flight_start_frame

            pass_event = PassEvent(
                start_frame=self.flight_start_frame,
                end_frame=frame_idx,
                from_player_id=self. flight_from_player,
                to_player_id=self.candidate_player,
                from_team_id=self. flight_from_team,
                to_team_id=self.candidate_team,
                pass_type=pass_type,
                flight_frames=max(0, flight_frames)
            )
            self.passes. append(pass_event)

        # Przejdź do stanu posiadania
        self.state = PassState. PLAYER_HAS_BALL
        self.current_player = self.candidate_player
        self.current_team = self.candidate_team

        return pass_event

    def _reset(self):
        self.state = PassState.NO_POSSESSION
        self.flight_from_player = None
        self.flight_from_team = None
        self.candidate_player = None
        self.candidate_detections = []
        self.current_player = None
        self.current_team = None

    def get_pass_stats(self) -> Dict:
        successful = [p for p in self.passes if p.pass_type == PassType. SUCCESSFUL]
        intercepted = [p for p in self.passes if p.pass_type == PassType. INTERCEPTED]

        return {
            "total_passes": len(self.passes),
            "successful": len(successful),
            "intercepted": len(intercepted),
            "accuracy": len(successful) / len(self. passes) * 100 if self.passes else 0.0
        }

    def get_passes_by_team(self) -> Dict[int, Dict]:
        stats = {
            1: {"successful": 0, "intercepted": 0, "total": 0, "accuracy": 0.0},
            2: {"successful": 0, "intercepted": 0, "total": 0, "accuracy": 0.0}
        }

        for p in self. passes:
            if p.from_team_id in stats:
                if p. pass_type == PassType.SUCCESSFUL:
                    stats[p.from_team_id]["successful"] += 1
                else:
                    stats[p.from_team_id]["intercepted"] += 1

        for team_id in stats:
            total = stats[team_id]["successful"] + stats[team_id]["intercepted"]
            stats[team_id]["total"] = total
            stats[team_id]["accuracy"] = (
                stats[team_id]["successful"] / total * 100 if total > 0 else 0.0
            )

        return stats

    def get_all_passes(self) -> List[PassEvent]:
        return self.passes