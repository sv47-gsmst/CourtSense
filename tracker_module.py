# Speed here is scale-invariant — measured in torso-lengths per second
# instead of raw pixels per frame, since raw pixel speed changes
# depending on how far the player is from the camera and what the
# camera's frame rate is. Torso length (shoulder-to-hip distance) acts
# as a personal ruler that's roughly constant per person. An adult
# torso is about 0.45m, so 1.0 torso-length/sec is roughly a brisk
# walk and 4.0 is roughly a sprint.

import time
import numpy as np
from collections import deque


def torso_length_px(landmarks, w: int, h: int) -> float:
    """Shoulder-midpoint to hip-midpoint distance, in pixels.
    Returns 0.0 if shoulders/hips aren't confidently visible this frame —
    callers should fall back to a track's smoothed value in that case."""
    def vis(i):
        return getattr(landmarks[i], 'visibility', 1.0)

    if vis(11) > 0.3 and vis(12) > 0.3 and vis(23) > 0.3 and vis(24) > 0.3:
        sx = (landmarks[11].x + landmarks[12].x) / 2 * w
        sy = (landmarks[11].y + landmarks[12].y) / 2 * h
        hx = (landmarks[23].x + landmarks[24].x) / 2 * w
        hy = (landmarks[23].y + landmarks[24].y) / 2 * h
        return float(np.hypot(sx - hx, sy - hy))
    return 0.0


class PersonTrack:
    _next_id = 1

    def __init__(self, center, landmarks, torso_px, ts):
        self.id = PersonTrack._next_id
        PersonTrack._next_id += 1

        self.center    = center
        self.landmarks = landmarks
        self.history   = deque(maxlen=90)
        self.history.append(center)

        self.last_seen        = ts
        self.last_update_time = ts

        # velocity in px/sec (real time-based, not px/frame)
        self.velocity = (0.0, 0.0)

        # Smoothed torso length in pixels — this is the per-person "ruler".
        # Clamped to >= 1.0 so we never divide by zero.
        self.torso_px = max(torso_px, 1.0)

        # speed in TORSO-LENGTHS PER SECOND
        self.speed = 0.0
        self.label = "idle"

    def update(self, center, landmarks, torso_px, ts):
        dt = max(ts - self.last_update_time, 1e-3)

        if self.history:
            prev = self.history[-1]
            dx, dy = center[0] - prev[0], center[1] - prev[1]
            self.velocity = (dx / dt, dy / dt)            # px/sec
            raw_speed_px_per_sec = float(np.hypot(dx, dy)) / dt
        else:
            raw_speed_px_per_sec = 0.0

        # Smooth the torso-length estimate (EMA) so normalisation doesn't
        # jump around just because pose estimation jitters slightly
        if torso_px > 1.0:
            self.torso_px = 0.85 * self.torso_px + 0.15 * torso_px

        self.speed = raw_speed_px_per_sec / self.torso_px

        self.center    = center
        self.landmarks = landmarks
        self.history.append(center)
        self.last_seen        = ts
        self.last_update_time = ts

        # 0.4 torso-lengths/sec ≈ 0.18 m/s — a realistic "barely shifting
        # weight" threshold for an adult
        self.label = "moving" if self.speed > 0.4 else "idle"


class MultiPersonTracker:
    def __init__(self, max_distance_torso: float = 2.5, max_lost_frames: int = 30):
        self.tracks: list[PersonTrack] = []
        # Matching radius is expressed in TORSO LENGTHS of the track being
        # matched, not raw pixels — so it scales correctly whether the
        # person is near or far from the camera
        self.max_distance_torso = max_distance_torso
        self.max_lost_frames    = max_lost_frames
        self._lost_counts: dict[int, int] = {}

    def reset(self) -> None:
        # PersonTrack._next_id is a class-level counter shared across all
        # tracks ever created — without resetting it, "primary" (lowest
        # ID) can stay locked onto whoever arrived first across the whole
        # program's run, long after that person and their track are gone.
        self.tracks = []
        self._lost_counts = {}
        PersonTrack._next_id = 1

    def _center(self, landmarks, w, h):
        pts = [landmarks[i] for i in (11, 12, 23, 24)
               if getattr(landmarks[i], 'visibility', 1.0) > 0.3]
        if not pts:
            pts = landmarks
        cx = sum(p.x for p in pts) / len(pts) * w
        cy = sum(p.y for p in pts) / len(pts) * h
        return (cx, cy)

    def update(self, all_pose_landmarks, w, h):
        ts = time.time()
        detections = []
        for lms in all_pose_landmarks:
            c    = self._center(lms, w, h)
            t_px = torso_length_px(lms, w, h)
            detections.append((c, lms, t_px))

        matched_tracks, matched_dets = set(), set()
        assignments = []

        for ti, track in enumerate(self.tracks):
            best_dist, best_di = float('inf'), -1
            for di, (c, _, _) in enumerate(detections):
                if di in matched_dets:
                    continue
                d_px    = float(np.hypot(c[0] - track.center[0], c[1] - track.center[1]))
                d_torso = d_px / max(track.torso_px, 1.0)
                if d_torso < best_dist:
                    best_dist, best_di = d_torso, di
            if best_di >= 0 and best_dist < self.max_distance_torso:
                assignments.append((ti, best_di))
                matched_tracks.add(ti)
                matched_dets.add(best_di)

        for ti, di in assignments:
            c, lms, t_px = detections[di]
            self.tracks[ti].update(c, lms, t_px, ts)
            self._lost_counts[self.tracks[ti].id] = 0

        for di, (c, lms, t_px) in enumerate(detections):
            if di not in matched_dets:
                nt = PersonTrack(c, lms, t_px, ts)
                self.tracks.append(nt)
                self._lost_counts[nt.id] = 0

        for ti, track in enumerate(self.tracks):
            if ti not in matched_tracks:
                self._lost_counts[track.id] = self._lost_counts.get(track.id, 0) + 1

        self.tracks = [t for t in self.tracks
                       if self._lost_counts.get(t.id, 0) < self.max_lost_frames]
        return self.tracks