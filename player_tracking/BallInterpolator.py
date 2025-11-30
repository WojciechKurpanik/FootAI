# python
import numpy as np
from collections import OrderedDict, deque
from typing import Dict, Tuple, List, Union, Optional

class BallInterpolator:
    """
    Prosta interpolacja pozycji piłki.
    - add_detection(frame_idx, bbox_or_center)
    - interpolate(start_frame=None, end_frame=None) -> Dict[frame_idx, (x,y)]
    - get_bbox(center) -> (x1,y1,x2,y2)
    bbox_or_center: (x,y) lub (x1,y1,x2,y2)
    """

    def __init__(self, smoothing: int = 1, bbox_size: int = 24):
        self.detections: Dict[int, Tuple[float, float]] = {}
        self.smoothing = max(1, int(smoothing))
        self.bbox_size = int(bbox_size)

    def _to_center(self, v: Union[Tuple[float, float], Tuple[float, float, float, float]]):
        if len(v) == 2:
            return float(v[0]), float(v[1])
        x1, y1, x2, y2 = v
        return float((x1 + x2) / 2.0), float((y1 + y2) / 2.0)

    def add_detection(self, frame_idx: int, bbox_or_center: Union[Tuple[float, float], Tuple[float, float, float, float]]):
        """Dodaj wykrycie piłki dla klatki."""
        c = self._to_center(bbox_or_center)
        self.detections[int(frame_idx)] = c

    def interpolate(self, start_frame: Optional[int] = None, end_frame: Optional[int] = None) -> Dict[int, Tuple[float, float]]:
        """
        Zwraca słownik frame_idx -> (x,y) dla wszystkich klatek od start_frame do end_frame.
        Jeśli start_frame/end_frame nie podane, używa min/max znanych detekcji.
        Braki poza zakresem znanych detekcji wypełniane są najbliższym znanym punktem.
        Po interpolacji opcjonalne wygładzenie ruchem średniej.
        """
        if not self.detections:
            return {}

        items = sorted(self.detections.items())
        frames = [f for f, _ in items]
        pts = [p for _, p in items]

        s = start_frame if start_frame is not None else frames[0]
        e = end_frame if end_frame is not None else frames[-1]
        s, e = int(s), int(e)
        total_len = e - s + 1
        xs = np.full(total_len, np.nan, dtype=float)
        ys = np.full(total_len, np.nan, dtype=float)

        # fill known
        for f, (x, y) in items:
            if s <= f <= e:
                idx = f - s
                xs[idx] = x
                ys[idx] = y

        # linear interpolation for nan segments
        def interp_axis(axis):
            n = axis.size
            not_nan = ~np.isnan(axis)
            if not_np := np.any(not_nan):
                idxs = np.where(not_nan)[0]
                vals = axis[not_nan]
                # before first known -> copy first
                if idxs[0] > 0:
                    axis[:idxs[0]] = vals[0]
                # after last known -> copy last
                if idxs[-1] < n-1:
                    axis[idxs[-1]+1:] = vals[-1]
                # interpolate internal gaps
                for i in range(len(idxs)-1):
                    a, b = idxs[i], idxs[i+1]
                    if b - a > 1:
                        axis[a:b+1] = np.linspace(axis[a], axis[b], b - a + 1)
            return axis

        xs = interp_axis(xs)
        ys = interp_axis(ys)

        # smoothing (moving average)
        if self.smoothing > 1:
            k = self.smoothing
            pad = k // 2
            xs_p = np.pad(xs, (pad, pad), mode='edge')
            ys_p = np.pad(ys, (pad, pad), mode='edge')
            xs = np.convolve(xs_p, np.ones(k)/k, mode='valid')
            ys = np.convolve(ys_p, np.ones(k)/k, mode='valid')

        result: Dict[int, Tuple[float, float]] = {}
        for i in range(total_len):
            frame_id = s + i
            result[frame_id] = (float(xs[i]), float(ys[i]))

        return result

    def get_bbox(self, center: Tuple[float, float], size: Optional[int] = None) -> Tuple[int, int, int, int]:
        """Zwraca kwadratowy bbox wokół środka: (x1,y1,x2,y2)"""
        if size is None:
            size = self.bbox_size
        cx, cy = center
        half = size / 2.0
        x1 = int(round(cx - half))
        y1 = int(round(cy - half))
        x2 = int(round(cx + half))
        y2 = int(round(cy + half))
        return x1, y1, x2, y2
