# Tracks how reliable MediaPipe's pose data is right now. When a player
# turns away from the camera or gets partially blocked, MediaPipe still
# returns coordinates but the visibility scores drop — detectors have no
# way to know that on their own, so this gives them a quality score to
# check before trusting a frame.
#
# Not a fix, just a gate: main.py/pipeline_b.py skip detector updates
# when score < QUALITY_THRESHOLD instead of firing events on bad data.

import numpy as np
from collections import deque

# Joints weighted by how much detectors actually rely on them —
# wrists/knees matter most (strokes, lunges), shoulders/hips give
# general orientation confidence
_WEIGHTED_JOINTS = {
    11: 1.0,  # left shoulder
    12: 1.0,  # right shoulder
    23: 1.2,  # left hip
    24: 1.2,  # right hip
    25: 1.3,  # left knee
    26: 1.3,  # right knee
    15: 1.4,  # left wrist
    16: 1.4,  # right wrist
}
_TOTAL_WEIGHT = sum(_WEIGHTED_JOINTS.values())

QUALITY_THRESHOLD = 0.45   # below this, detectors get suppressed
_EMA_ALPHA = 0.25          # higher = more responsive, lower = more stable


class PoseQuality:
    def __init__(self):
        self.score:  float = 1.0    # smoothed 0-1 quality
        self.raw:    float = 1.0    # unsmoothed this-frame score
        self.status: str   = "good" # "good" / "degraded" / "poor"
        self._history: deque = deque(maxlen=30)   # for a mini quality graph

    def update(self, landmarks) -> float:
        """
        landmarks: MediaPipe pose_landmarks[0] list (33 landmarks).
        Returns the smoothed quality score.
        """
        weighted_sum = 0.0
        for idx, weight in _WEIGHTED_JOINTS.items():
            vis = getattr(landmarks[idx], 'visibility', 1.0)
            weighted_sum += vis * weight

        self.raw   = weighted_sum / _TOTAL_WEIGHT
        self.score = (1 - _EMA_ALPHA) * self.score + _EMA_ALPHA * self.raw
        self._history.append(self.score)

        if self.score >= 0.70:
            self.status = "good"
        elif self.score >= QUALITY_THRESHOLD:
            self.status = "degraded"
        else:
            self.status = "poor"

        return self.score

    @property
    def suppress_detectors(self) -> bool:
        """True when quality is too low to trust detector output."""
        return self.score < QUALITY_THRESHOLD

    @property
    def status_color(self) -> tuple:
        """BGR colour matching the current status."""
        if self.status == "good":     return (90, 255, 140)   # green
        if self.status == "degraded": return (0, 215, 255)    # yellow
        return (45, 45, 230)                                   # red

    def reset(self) -> None:
        self.score = 1.0
        self.raw   = 1.0
        self.status = "good"
        self._history.clear()