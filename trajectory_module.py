# trajectory_module.py
#
# REALISM FIXES


import cv2
import numpy as np
from collections import deque

from action_module import SPEED_SHUFFLE


# Prediction horizon — 0.5s is the outer edge of useful linear prediction
# for human locomotion. Beyond this the person could have changed direction
# multiple times.
MAX_PREDICT_SEC = 0.50

# Average velocity is computed over this many recent frames. 4 is enough
# to smooth single-frame jitter without lagging behind a real direction
# change that takes 2-3 frames to commit to.
N_VELOCITY_FRAMES = 4

# Number of prediction dots to draw. Spread over MAX_PREDICT_SEC, so at
# 30fps this is one dot per ~3 frames — readable without clutter.
N_DOTS = 6


class TrajectoryPredictor:
    def __init__(self):
        # Stores (timestamp_sec, cx_px, cy_px) tuples
        self._buf: deque = deque(maxlen=60)   # ~2s at 30fps

    def _avg_velocity(self) -> tuple:
        """
        Returns average (vx, vy) in px/sec over the last N_VELOCITY_FRAMES
        pairs of consecutive buffer entries. Returns (0,0) if not enough data.
        """
        pts = list(self._buf)
        if len(pts) < 2:
            return (0.0, 0.0)

        recent = pts[-min(N_VELOCITY_FRAMES + 1, len(pts)):]
        vx_sum, vy_sum, count = 0.0, 0.0, 0
        for i in range(1, len(recent)):
            t0, x0, y0 = recent[i - 1]
            t1, x1, y1 = recent[i]
            dt = max(t1 - t0, 1e-3)
            vx_sum += (x1 - x0) / dt
            vy_sum += (y1 - y0) / dt
            count += 1

        if count == 0:
            return (0.0, 0.0)
        return (vx_sum / count, vy_sum / count)

    def update(self, cx: float, cy: float, timestamp: float) -> None:
        """Call once per frame with the primary player's hip-centre position."""
        self._buf.append((timestamp, cx, cy))

    def draw(self, img: np.ndarray, speed: float,
             w: int, h: int,
             trail_color: tuple = (120, 120, 120),
             dot_color:   tuple = (0, 200, 255)) -> np.ndarray:
        """
        Draws:
          - Past trail (last ~1s of positions) in gray
          - Prediction dots (linear, up to MAX_PREDICT_SEC) in cyan
            only when the player is actually moving

        speed is in torso-lengths/sec (PersonTrack.speed).
        """
        pts = list(self._buf)
        if len(pts) < 2:
            return img

        # ── Past trail ─────────────────────────────────────────────────────
        # Show roughly the last 1 second of real positions.
        now = pts[-1][0]
        trail_pts = [(int(np.clip(x, 0, w)), int(np.clip(y, 0, h)))
                     for (t, x, y) in pts if now - t <= 1.0]
        for i in range(1, len(trail_pts)):
            # Fade older segments: fully visible at i == len-1, dim at i == 0
            frac  = i / max(len(trail_pts) - 1, 1)
            alpha = int(80 + 120 * frac)
            col   = tuple(int(c * alpha / 200) for c in trail_color)
            cv2.line(img, trail_pts[i - 1], trail_pts[i], col, 1, cv2.LINE_AA)

        # ── Prediction dots (only when actually moving) ─────────────────────
        if speed < SPEED_SHUFFLE:
            return img

        vx, vy = self._avg_velocity()
        speed_px_sec = float(np.hypot(vx, vy))
        if speed_px_sec < 1.0:
            return img

        # Origin of prediction = most recent recorded position
        _, ox, oy = pts[-1]

        for i in range(1, N_DOTS + 1):
            t_ahead = (i / N_DOTS) * MAX_PREDICT_SEC
            px = ox + vx * t_ahead
            py = oy + vy * t_ahead

            # Skip if outside frame
            if px < 0 or px > w or py < 0 or py > h:
                break

            # Radius shrinks with distance (8px at first dot → 4px at last)
            radius = max(2, int(8 - (i / N_DOTS) * 4))

            # Colour fades (full opacity near, transparent far)
            fade  = 1.0 - (i / N_DOTS) * 0.6
            col   = tuple(int(c * fade) for c in dot_color)

            cv2.circle(img, (int(px), int(py)), radius, col, -1, cv2.LINE_AA)

        return img

    def reset(self) -> None:
        self._buf.clear()