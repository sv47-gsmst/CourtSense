# analytics_module.py
#
# Note on the angle naming here: the old "torso" angle (shoulder-hip-knee)
# actually measures hip flexion, not how much the torso leans, so it's
# called hip_angle. torso_lean is the real lean angle — shoulder-hip line
# vs vertical. Speed-zone colours and thresholds use the same
# torso-lengths/sec units as everything else in the analytics pipeline,
# so "red = fast" means the same real speed no matter how far the camera
# is from the player.

import numpy as np
import cv2
import time
from collections import deque

import mediapipe as mp

from action_module import SPEED_READY, SPEED_SHUFFLE, SPEED_SPRINT


SPEED_COLORS = {
    "fast":   (0, 60, 255),    # red    — sprinting  (> SPEED_SPRINT)
    "medium": (0, 200, 255),   # yellow — jogging    (SPEED_SHUFFLE..SPEED_SPRINT)
    "slow":   (200, 120, 0),   # blue   — shuffling/ready (<= SPEED_SHUFFLE)
}


def _speed_zone(speed_torso_per_sec: float) -> str:
    if speed_torso_per_sec > SPEED_SPRINT:
        return "fast"
    if speed_torso_per_sec > SPEED_SHUFFLE:
        return "medium"
    return "slow"


# ── Body analytics — rolling speed/acceleration history ─────────────────
class BodyAnalytics:
    """
    speed_history / accel_history are stored in TORSO-LENGTHS/SEC and
    TORSO-LENGTHS/SEC^2 (matches PersonTrack.speed from tracker_module).
    time_idle / time_moving use real elapsed seconds (dt), not frame counts,
    so they're correct regardless of frame rate or dropped frames.
    """
    def __init__(self, history_len: int = 200):
        self.speed_history: deque = deque(maxlen=history_len)
        self.accel_history: deque = deque(maxlen=history_len)
        self.last_speed  = 0.0
        self.last_accel  = 0.0
        self.avg_speed   = 0.0
        self.time_idle   = 0.0
        self.time_moving = 0.0
        self._last_ts    = time.time()
        self._has_prev   = False

    def update(self, speed: float) -> None:
        now = time.time()
        dt  = max(now - self._last_ts, 1e-3)
        self._last_ts = now

        accel = (speed - self.last_speed) / dt if self._has_prev else 0.0

        self.speed_history.append(speed)
        self.accel_history.append(accel)
        self.last_speed = speed
        self.last_accel = accel
        self.avg_speed  = float(np.mean(self.speed_history))
        self._has_prev  = True

        if speed < SPEED_READY:
            self.time_idle += dt
        else:
            self.time_moving += dt

    def speed_zone(self, speed: float) -> str:
        return _speed_zone(speed)


# ── Body angles ─────────────────────────────────────────────────────────
def calculate_body_angles(landmarks, w: int, h: int) -> dict:
    """
    Returns (degrees, or None if landmarks aren't visible enough):
      left_leg / right_leg   — knee flexion   (180 = straight leg)
      left_arm / right_arm   — elbow flexion  (180 = straight arm)
      hip_angle              — hip flexion, shoulder-hip-knee
                                (180 = standing tall, smaller = bent
                                forward at the hip)
      torso_lean             — degrees the shoulder-hip line leans away
                                from vertical (0 = perfectly upright,
                                larger = leaning toward horizontal)
    """
    def px(lm):
        return np.array([lm.x * w, lm.y * h])

    def vis(i):
        return getattr(landmarks[i], 'visibility', 1.0)

    def angle_2d(a, b, c):
        va, vc = a - b, c - b
        n1, n2 = np.linalg.norm(va), np.linalg.norm(vc)
        if n1 < 1e-6 or n2 < 1e-6:
            return None
        cosang = np.dot(va, vc) / (n1 * n2)
        return float(np.degrees(np.arccos(np.clip(cosang, -1, 1))))

    lms = landmarks
    result: dict = {}

    result["left_leg"] = (
        angle_2d(px(lms[23]), px(lms[25]), px(lms[27]))
        if vis(23) > 0.3 and vis(25) > 0.3 and vis(27) > 0.3 else None
    )
    result["right_leg"] = (
        angle_2d(px(lms[24]), px(lms[26]), px(lms[28]))
        if vis(24) > 0.3 and vis(26) > 0.3 and vis(28) > 0.3 else None
    )
    result["left_arm"] = (
        angle_2d(px(lms[11]), px(lms[13]), px(lms[15]))
        if vis(11) > 0.3 and vis(13) > 0.3 and vis(15) > 0.3 else None
    )
    result["right_arm"] = (
        angle_2d(px(lms[12]), px(lms[14]), px(lms[16]))
        if vis(12) > 0.3 and vis(14) > 0.3 and vis(16) > 0.3 else None
    )

    # Hip flexion — prefer left side, fall back to right if left isn't visible
    if vis(11) > 0.3 and vis(23) > 0.3 and vis(25) > 0.3:
        result["hip_angle"] = angle_2d(px(lms[11]), px(lms[23]), px(lms[25]))
    elif vis(12) > 0.3 and vis(24) > 0.3 and vis(26) > 0.3:
        result["hip_angle"] = angle_2d(px(lms[12]), px(lms[24]), px(lms[26]))
    else:
        result["hip_angle"] = None

    # Torso lean from vertical — uses both-side midpoints for stability
    if vis(11) > 0.3 and vis(12) > 0.3 and vis(23) > 0.3 and vis(24) > 0.3:
        sh = (px(lms[11]) + px(lms[12])) / 2
        hp = (px(lms[23]) + px(lms[24])) / 2
        dx = abs(sh[0] - hp[0])
        dy = abs(sh[1] - hp[1])
        result["torso_lean"] = float(np.degrees(np.arctan2(dx, dy + 1e-6)))
    else:
        result["torso_lean"] = None

    return result


# ── Speed-zone skeleton ─────────────────────────────────────────────────
def draw_speed_zone_skeleton(img, landmarks, speed: float, w: int, h: int,
                             threshold: float = 0.3):
    """speed is torso-lengths/sec — zone boundaries match classify_movement."""
    mp_pose = mp.solutions.pose
    zone  = _speed_zone(speed)
    color = SPEED_COLORS[zone]

    hand_indices = {17, 18, 19, 20, 21, 22}
    for conn in mp_pose.POSE_CONNECTIONS:
        if conn[0] in hand_indices or conn[1] in hand_indices:
            continue
        p1, p2 = landmarks[conn[0]], landmarks[conn[1]]
        v1 = getattr(p1, 'visibility', 1.0)
        v2 = getattr(p2, 'visibility', 1.0)
        if v1 > threshold and v2 > threshold:
            if abs(p1.x - p2.x) > 0.88:
                continue
            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)
            cv2.line(img, (x1, y1), (x2, y2), color, 3, cv2.LINE_AA)
    return img


# ── Inter-player distance ────────────────────────────────────────────────
def distance_between_players(tracks):
    """
    Returns (id1, id2, pixel_distance) for every pair of tracked people.
    Pixel distance alone is meaningless across cameras/zoom levels — the
    caller (main.py) converts this to "body-lengths apart" using each
    track's own smoothed torso_px, which is the only per-frame scale
    reference available from a single uncalibrated camera.
    """
    out = []
    for i in range(len(tracks)):
        for j in range(i + 1, len(tracks)):
            t1, t2 = tracks[i], tracks[j]
            d = float(np.hypot(t1.center[0] - t2.center[0],
                               t1.center[1] - t2.center[1]))
            out.append((t1.id, t2.id, d))
    return out