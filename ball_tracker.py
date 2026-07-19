# HSV colour mask for tennis balls and pickleballs. No ML, just OpenCV.
#
# HSV ranges (H 0-179, S 0-255, V 0-255): H 20-50 covers both a tennis
# ball's yellow and a pickleball's yellow-green.
#
# Two filters keep this from tracking random yellow blobs: a size gate
# (radius between MIN_RADIUS and MAX_RADIUS) and a circularity check
# (real balls are round, shirts/lines aren't).
#
# Runs on a half-size copy of the frame for speed, then scales results
# back up.

import cv2
import numpy as np
import time
from collections import deque

HSV_LOWER = np.array([20,  60,  80], dtype=np.uint8)
HSV_UPPER = np.array([50, 255, 255], dtype=np.uint8)

MIN_RADIUS            = 6
MAX_RADIUS            = 80
CIRCULARITY_THRESHOLD = 0.55
HISTORY_LEN           = 8


class BallTracker:
    def __init__(self):
        self._history: deque = deque(maxlen=HISTORY_LEN)
        self.detected  = False
        self.cx        = 0.0
        self.cy        = 0.0
        self.radius    = 0.0
        self.vx        = 0.0   # px/sec
        self.vy        = 0.0

    def update(self, frame: np.ndarray) -> bool:
        now  = time.time()
        h, w = frame.shape[:2]
        small = cv2.resize(frame, (w // 2, h // 2))
        hsv   = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
        mask  = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask   = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        best_score, best_circle = -1.0, None

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < np.pi * MIN_RADIUS ** 2:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cx *= 2.0; cy *= 2.0; radius *= 2.0
            if not (MIN_RADIUS <= radius <= MAX_RADIUS):
                continue
            circularity = (area * 4) / (np.pi * radius ** 2)
            if circularity < CIRCULARITY_THRESHOLD:
                continue
            size_score = 1.0 - abs(radius - 20) / 60.0
            score = circularity * max(size_score, 0.1)
            if score > best_score:
                best_score, best_circle = score, (cx, cy, radius)

        if best_circle is not None:
            self.cx, self.cy, self.radius = best_circle
            self.detected = True
            self._history.append((now, self.cx, self.cy))
            self._update_velocity()
        else:
            self.detected = False

        return self.detected

    def _update_velocity(self) -> None:
        pts = list(self._history)
        if len(pts) < 2:
            self.vx = self.vy = 0.0
            return
        t0, x0, y0 = pts[0]
        t1, x1, y1 = pts[-1]
        dt = max(t1 - t0, 1e-3)
        self.vx = (x1 - x0) / dt
        self.vy = (y1 - y0) / dt

    def near_wrist(self, wrist_x: float, wrist_y: float,
                   torso_px: float) -> bool:
        if not self.detected:
            return False
        dist = float(np.hypot(self.cx - wrist_x, self.cy - wrist_y))
        return dist < 1.2 * max(torso_px, 1.0)

    def draw(self, img: np.ndarray) -> np.ndarray:
        if not self.detected:
            return img
        cx, cy, r = int(self.cx), int(self.cy), int(self.radius)
        cv2.circle(img, (cx, cy), r, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.circle(img, (cx, cy), 3, (0, 230, 255), -1, cv2.LINE_AA)
        spd = float(np.hypot(self.vx, self.vy))
        if spd > 20:
            scale = min(spd * 0.08, 60.0)
            ex = int(cx + self.vx / spd * scale)
            ey = int(cy + self.vy / spd * scale)
            cv2.arrowedLine(img, (cx, cy), (ex, ey), (0, 200, 255), 2,
                            tipLength=0.4, line_type=cv2.LINE_AA)
        return img

    def reset(self) -> None:
        self._history.clear()
        self.detected = False
        self.vx = self.vy = 0.0