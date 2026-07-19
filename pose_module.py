# pose_module.py
#
# Multi-person support: num_poses is set to 3 (MediaPipe defaults to 1),
# and find_pose() keeps every detected pose instead of discarding all but
# the tracked one, so tracker_module.py in analytics mode can see everyone
# in frame. Mode A still only ever reads pose_landmarks[0] (the smoothed,
# best-matched person), so exercise tracking is unaffected.

import os
import time
import json
import threading
from collections import deque
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Exercise modes
MODE_NONE   = 0
MODE_SQUAT  = 1
MODE_PUSHUP = 2
MODE_CURL   = 3

SMOOTHING_PRESETS = {
    "SMOOTH":     {"alpha": 0.08},
    "BALANCED":   {"alpha": 0.25},
    "RESPONSIVE": {"alpha": 0.65},
}


class PoseDetector:
    def __init__(self, config_path="config.json"):
        self.base_dir = os.path.dirname(os.path.abspath(__file__))
        base_pose_model = "pose_landmarker_full.task"
        base_hand_model = "hand_landmarker.task"

        self.config = {
            "model_path": os.path.normpath(
                os.path.join(self.base_dir, base_pose_model)),
            "hand_model_path": os.path.normpath(
                os.path.join(self.base_dir, base_hand_model)),
            "min_pose_detection_confidence": 0.5,
            "min_pose_presence_confidence":  0.5,
            "min_tracking_confidence":       0.5,
        }

        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    user_config = json.load(f)
                for key in ("model_path", "hand_model_path"):
                    if key in user_config:
                        fname = os.path.basename(user_config[key])
                        user_config[key] = os.path.normpath(
                            os.path.join(self.base_dir, fname))
                self.config.update(user_config)
            except json.JSONDecodeError:
                pass

        self._ts_lock        = threading.Lock()
        self._last_ts_ms     = 0
        self._previous_landmarks = None
        self.current_preset  = "BALANCED"

        self.visibility_threshold = 0.35
        self.show_roi             = True
        self.session_data         = []

        self.current_mode = MODE_NONE
        self.reps   = {MODE_SQUAT: 0, MODE_PUSHUP: 0, MODE_CURL: 0}
        self.states = {MODE_SQUAT: "UP", MODE_PUSHUP: "UP", MODE_CURL: "DOWN"}
        self.latest_angles  = {}
        self.form_feedback  = "Ready"
        self.state_timers   = {MODE_SQUAT: time.time(), MODE_PUSHUP: time.time(), MODE_CURL: time.time()}
        self.min_rep_duration = 0.12   # floor against sensor-glitch double-counts only

        # ── Rep-counting robustness (debounce + side-lock + quality) ───────
        # CONFIRM_FRAMES: consecutive frames a joint angle must stay past a
        # threshold before a state transition commits — filters the frame-
        # to-frame jitter that was previously causing phantom rep counts.
        self.CONFIRM_FRAMES = 2
        self._up_frames    = {MODE_SQUAT: 0, MODE_PUSHUP: 0, MODE_CURL: 0}
        self._down_frames  = {MODE_SQUAT: 0, MODE_PUSHUP: 0, MODE_CURL: 0}
        # Locked side (0=left,1=right) chosen once per rep so mid-rep
        # left/right visibility flips can't jump the tracked angle around
        self._locked_side  = {MODE_SQUAT: None, MODE_PUSHUP: None, MODE_CURL: None}
        # Extreme angle actually reached during the active phase of the rep
        # — used to score depth/ROM quality, and to reject reps that never
        # reached a real range of motion
        self._rep_extreme  = {MODE_SQUAT: 180.0, MODE_PUSHUP: 180.0, MODE_CURL: 180.0}
        # Light smoothing applied only to the rep-counting angle (separate
        # from the display smoothing), reduces single-frame noise further
        self._rep_angle_buf = {
            MODE_SQUAT: deque(maxlen=4),
            MODE_PUSHUP: deque(maxlen=4),
            MODE_CURL: deque(maxlen=4),
        }
        # Most recent rep's quality score (0-100) and rolling history for
        # a session average, read by hud.py to draw the quality bar
        self.last_rep_quality = {MODE_SQUAT: None, MODE_PUSHUP: None, MODE_CURL: None}
        self.rep_quality_history = {
            MODE_SQUAT: deque(maxlen=20),
            MODE_PUSHUP: deque(maxlen=20),
            MODE_CURL: deque(maxlen=20),
        }

        # ── Load pose model ────────────────────────────────────────────────
        try:
            with open(self.config["model_path"], "rb") as f:
                pose_model_buffer = f.read()
        except Exception as e:
            raise FileNotFoundError(f"Pose model error: {e}")

        # ── Load hand model ────────────────────────────────────────────────
        try:
            with open(self.config["hand_model_path"], "rb") as f:
                hand_model_buffer = f.read()
        except Exception as e:
            raise FileNotFoundError(f"Hand model error: {e}")

        self.mp_pose  = mp.solutions.pose
        self.mp_hands = mp.solutions.hands

        # num_poses raised from MediaPipe's default of 1 so tracker_module
        # can follow multiple people at once
        pose_options = vision.PoseLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_buffer=pose_model_buffer),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=3,                           # ← was 1
            min_pose_detection_confidence=self.config["min_pose_detection_confidence"],
            min_pose_presence_confidence=self.config["min_pose_presence_confidence"],
            min_tracking_confidence=self.config["min_tracking_confidence"],
        )
        self.detector = vision.PoseLandmarker.create_from_options(pose_options)

        hand_options = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_buffer=hand_model_buffer),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=self.config["min_pose_detection_confidence"],
            min_hand_presence_confidence=self.config["min_pose_presence_confidence"],
            min_tracking_confidence=self.config["min_tracking_confidence"],
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)

        # ── Mode selection ──────────────────────────────────────────────────
    def set_mode(self, mode_key):
        mapping = {'0': MODE_NONE, '1': MODE_SQUAT,
                   '2': MODE_PUSHUP, '3': MODE_CURL}
        if mode_key in mapping:
            self.current_mode = mapping[mode_key]
            self.reps   = {MODE_SQUAT: 0, MODE_PUSHUP: 0, MODE_CURL: 0}
            self.states = {MODE_SQUAT: "UP", MODE_PUSHUP: "UP", MODE_CURL: "DOWN"}
            self.state_timers   = {MODE_SQUAT: time.time(), MODE_PUSHUP: time.time(), MODE_CURL: time.time()}
            self.latest_angles  = {}
            self.form_feedback  = "Ready"
            self._up_frames     = {MODE_SQUAT: 0, MODE_PUSHUP: 0, MODE_CURL: 0}
            self._down_frames   = {MODE_SQUAT: 0, MODE_PUSHUP: 0, MODE_CURL: 0}
            self._locked_side   = {MODE_SQUAT: None, MODE_PUSHUP: None, MODE_CURL: None}
            self._rep_extreme   = {MODE_SQUAT: 180.0, MODE_PUSHUP: 180.0, MODE_CURL: 180.0}
            for buf in self._rep_angle_buf.values():
                buf.clear()
            self.last_rep_quality = {MODE_SQUAT: None, MODE_PUSHUP: None, MODE_CURL: None}
            for hist in self.rep_quality_history.values():
                hist.clear()

        # ── Angle calculation (world-space, 3D) ─────────────────────────────
    def _calculate_angle_3d(self, a, b, c):
        va = np.array([a.x - b.x, a.y - b.y, a.z - b.z])
        vc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        n1, n2 = np.linalg.norm(va), np.linalg.norm(vc)
        if n1 == 0 or n2 == 0:
            return None
        cos = np.dot(va, vc) / (n1 * n2)
        return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))

    def _get_angle_3d_if_visible(self, a, b, c, raw_a, raw_b, raw_c):
        def vis(lm):
            return lm.visibility if hasattr(lm, 'visibility') else 1.0
        if (vis(raw_a) < self.visibility_threshold
                or vis(raw_b) < self.visibility_threshold
                or vis(raw_c) < self.visibility_threshold):
            return None
        return self._calculate_angle_3d(a, b, c)

    def _get_best_side(self, landmarks, joints):
        l_vis = sum(landmarks[j[0]].visibility
                    for j in joints
                    if hasattr(landmarks[j[0]], 'visibility'))
        r_vis = sum(landmarks[j[1]].visibility
                    for j in joints
                    if hasattr(landmarks[j[1]], 'visibility'))
        return 0 if l_vis >= r_vis else 1

    def _update_all_angles(self, screen_lms, world_lms):
        sl, wl = screen_lms, world_lms
        self.latest_angles = {
            "left_knee":   self._get_angle_3d_if_visible(
                wl[23], wl[25], wl[27], sl[23], sl[25], sl[27]),
            "right_knee":  self._get_angle_3d_if_visible(
                wl[24], wl[26], wl[28], sl[24], sl[26], sl[28]),
            "left_elbow":  self._get_angle_3d_if_visible(
                wl[11], wl[13], wl[15], sl[11], sl[13], sl[15]),
            "right_elbow": self._get_angle_3d_if_visible(
                wl[12], wl[14], wl[16], sl[12], sl[14], sl[16]),
            "left_hip":    self._get_angle_3d_if_visible(
                wl[11], wl[23], wl[25], sl[11], sl[23], sl[25]),
            "right_hip":   self._get_angle_3d_if_visible(
                wl[12], wl[24], wl[26], sl[12], sl[24], sl[26]),
            "left_shoulder":  self._get_angle_3d_if_visible(
                wl[13], wl[11], wl[23], sl[13], sl[11], sl[23]),
            "right_shoulder": self._get_angle_3d_if_visible(
                wl[14], wl[12], wl[24], sl[14], sl[12], sl[24]),
        }

        # ── Debounce / side-recovery helpers ────────────────────────────────
    def _update_confirm_counters(self, angle: float, up_thresh: float,
                                  down_thresh: float, mode) -> None:
        """
        Accumulate/decay hysteresis counter instead of a hard reset-to-zero
        debounce. Facing the camera head-on foreshortens a straight leg or
        arm so the angle can hover right at a threshold and flicker across
        it frame to frame — a reset-to-zero debounce means one bad frame
        wipes out all progress, which was leaving squats stuck mid-rep.

        Each frame past the threshold adds to the counter (capped); each
        frame that isn't just subtracts one instead of zeroing it out, so
        a single noisy frame costs a little progress, not all of it.
        """
        cap = self.CONFIRM_FRAMES + 2
        up, down = self._up_frames[mode], self._down_frames[mode]
        if angle > up_thresh:
            up   = min(up + 1, cap)
            down = max(down - 1, 0)
        elif angle < down_thresh:
            down = min(down + 1, cap)
            up   = max(up - 1, 0)
        else:
            up   = max(up - 1, 0)
            down = max(down - 1, 0)
        self._up_frames[mode]   = up
        self._down_frames[mode] = down

    def _resolve_side(self, lms, joints, mode) -> int:
        """
        Returns the side (0=left, 1=right) to track this frame. Prefers
        staying on the currently locked side (keeps a rep's angle series
        consistent) but falls back to re-selecting the best side if the
        locked side's joints have degraded below a usable visibility —
        this recovers gracefully instead of getting stuck tracking a limb
        MediaPipe can no longer see well (common when facing the camera
        head-on and one side's landmarks briefly lose confidence).
        """
        locked = self._locked_side[mode]
        if locked is not None:
            vis_ok = all(getattr(lms[j[locked]], 'visibility', 1.0) > 0.35
                        for j in joints)
            if vis_ok:
                return locked
        return self._get_best_side(lms, joints)

        # ── Rep quality scoring ─────────────────────────────────────────────
    def _log_rep(self, mode, quality: int, duration: float) -> None:
        # session_data feeds export_session() — one entry per completed
        # rep rather than per frame, so the exported file stays a
        # reasonable size and actually reflects what export_session
        # promises when the user presses [e].
        names = {MODE_SQUAT: "squat", MODE_PUSHUP: "pushup", MODE_CURL: "curl"}
        self.session_data.append({
            "timestamp":    time.time(),
            "exercise":     names.get(mode, "unknown"),
            "rep_number":   self.reps.get(mode, 0),
            "quality":      quality,
            "duration_sec": round(duration, 3),
        })

    def _score_rep(self, extreme_angle: float, duration: float,
                   excellent_angle: float, ok_angle: float,
                   ideal_min_sec: float, ideal_max_sec: float) -> int:
        """0-100 rep score. Depth counts for most of it; tempo is a light
        nudge, not a gate — a fast rep and a slow rep both count, they
        just score a bit lower on tempo if they're outside a normal range."""
        depth_range = max(abs(ok_angle - excellent_angle), 1.0)
        depth_frac  = 1.0 - min(abs(extreme_angle - excellent_angle) / depth_range, 1.0)
        depth_score = max(0.0, min(1.0, depth_frac)) * 100

        if ideal_min_sec <= duration <= ideal_max_sec:
            tempo_score = 100.0
        elif duration < ideal_min_sec:
            tempo_score = max(55.0, 100 - (ideal_min_sec - duration) * 120)
        else:
            tempo_score = max(30.0, 100 - (duration - ideal_max_sec) * 15)

        return int(round(0.70 * depth_score + 0.30 * tempo_score))

        # ── Exercise logic — debounced, side-locked, ROM-validated ─────────
    def _squat_logic(self, lms):
        UP_THRESH, DOWN_THRESH = 155.0, 108.0   # slightly relaxed for frontal-view margin
        EXCELLENT_DEPTH        = 95.0
        now = time.time()
        joints = [(23, 24), (25, 26), (27, 28)]

        side = self._resolve_side(lms, joints, MODE_SQUAT)
        knee_raw = (self.latest_angles["left_knee"] if side == 0
                   else self.latest_angles["right_knee"])
        if knee_raw is None:
            self.form_feedback = "Knee obscured"
            return

        buf = self._rep_angle_buf[MODE_SQUAT]
        buf.append(knee_raw)
        knee = sum(buf) / len(buf)

        if self.states[MODE_SQUAT] == "DOWN":
            self._rep_extreme[MODE_SQUAT] = min(self._rep_extreme[MODE_SQUAT], knee)

        self._update_confirm_counters(knee, UP_THRESH, DOWN_THRESH, MODE_SQUAT)
        CONFIRM = self.CONFIRM_FRAMES

        if (self._down_frames[MODE_SQUAT] >= CONFIRM
                and self.states[MODE_SQUAT] == "UP"):
            self.states[MODE_SQUAT]       = "DOWN"
            self.state_timers[MODE_SQUAT] = now
            self._locked_side[MODE_SQUAT] = side
            self._rep_extreme[MODE_SQUAT] = knee
            self.form_feedback = "Good depth."

        elif (self._up_frames[MODE_SQUAT] >= CONFIRM
                and self.states[MODE_SQUAT] == "DOWN"):
            elapsed = now - self.state_timers[MODE_SQUAT]
            extreme = self._rep_extreme[MODE_SQUAT]
            if elapsed >= self.min_rep_duration:
                self.reps[MODE_SQUAT] += 1
                q = self._score_rep(extreme, elapsed, EXCELLENT_DEPTH,
                                    DOWN_THRESH, 0.25, 2.5)
                self.last_rep_quality[MODE_SQUAT] = q
                self.rep_quality_history[MODE_SQUAT].append(q)
                self._log_rep(MODE_SQUAT, q, elapsed)
                self.form_feedback = ("Good Rep!" if q >= 70
                                      else "Shallow — go lower")
            else:
                self.form_feedback = "Reset — try again"
            self.states[MODE_SQUAT]       = "UP"
            self._locked_side[MODE_SQUAT] = None

    def _pushup_logic(self, lms, w, h):
        UP_THRESH, DOWN_THRESH = 150.0, 103.0   # slightly relaxed for margin
        EXCELLENT_DEPTH        = 85.0
        now = time.time()
        joints = [(11, 12), (13, 14), (15, 16), (23, 24)]

        side = self._resolve_side(lms, joints, MODE_PUSHUP)

        sh = lms[11] if side == 0 else lms[12]
        hp = lms[23] if side == 0 else lms[24]
        dx = (sh.x - hp.x) * w
        dy = (sh.y - hp.y) * h
        if dx == 0:
            dx = 0.001
        torso_incl = np.degrees(np.abs(np.arctan2(dy, dx)))
        if torso_incl > 90:
            torso_incl = 180 - torso_incl
        if torso_incl > 50:
            self.form_feedback = "Please get into horizontal plank posture"
            return

        shoulder_span = np.abs(lms[11].x - lms[12].x) * w
        hip_span      = np.abs(lms[23].x - lms[24].x) * w
        is_front      = shoulder_span > hip_span * 1.3

        if is_front:
            valid = [v for v in [self.latest_angles["left_elbow"],
                                  self.latest_angles["right_elbow"]]
                     if v is not None]
            if not valid:
                self.form_feedback = "Arms obscured"
                return
            elbow_raw = float(np.mean(valid))
        else:
            elbow_raw = (self.latest_angles["left_elbow"] if side == 0
                        else self.latest_angles["right_elbow"])

        hip = (self.latest_angles["left_hip"] if side == 0
              else self.latest_angles["right_hip"])
        if elbow_raw is None or hip is None:
            self.form_feedback = "Body parts obscured"
            return

        buf = self._rep_angle_buf[MODE_PUSHUP]
        buf.append(elbow_raw)
        elbow = sum(buf) / len(buf)

        if self.states[MODE_PUSHUP] == "DOWN":
            self._rep_extreme[MODE_PUSHUP] = min(self._rep_extreme[MODE_PUSHUP], elbow)

        self._update_confirm_counters(elbow, UP_THRESH, DOWN_THRESH, MODE_PUSHUP)
        CONFIRM = self.CONFIRM_FRAMES

        if (self._down_frames[MODE_PUSHUP] >= CONFIRM
                and self.states[MODE_PUSHUP] == "UP"):
            self.states[MODE_PUSHUP]       = "DOWN"
            self.state_timers[MODE_PUSHUP] = now
            self._locked_side[MODE_PUSHUP] = side
            self._rep_extreme[MODE_PUSHUP] = elbow
            self.form_feedback = "Good depth."

        elif (self._up_frames[MODE_PUSHUP] >= CONFIRM
                and self.states[MODE_PUSHUP] == "DOWN"):
            elapsed = now - self.state_timers[MODE_PUSHUP]
            extreme = self._rep_extreme[MODE_PUSHUP]
            if elapsed >= self.min_rep_duration:
                self.reps[MODE_PUSHUP] += 1
                q = self._score_rep(extreme, elapsed, EXCELLENT_DEPTH,
                                    DOWN_THRESH, 0.25, 3.0)
                if hip < 135:
                    q = max(0, q - 20)   # hip sag penalty
                    self.form_feedback = "Rep counted, but fix hip sag!"
                else:
                    self.form_feedback = "Good Rep!" if q >= 70 else "Shallow — go lower"
                self.last_rep_quality[MODE_PUSHUP] = q
                self.rep_quality_history[MODE_PUSHUP].append(q)
                self._log_rep(MODE_PUSHUP, q, elapsed)
            else:
                self.form_feedback = "Reset — try again"
            self.states[MODE_PUSHUP]       = "UP"
            self._locked_side[MODE_PUSHUP] = None

    def _curl_logic(self, lms):
        UP_THRESH, DOWN_THRESH = 140.0, 60.0   # UP relaxed from 155 — most
        # real curl form doesn't fully lock the elbow out straight between
        # reps, and requiring a strict 155° was causing the state machine
        # to get permanently stuck at "contracted" after the first rep
        EXCELLENT_CONTRACT = 40.0
        now = time.time()
        joints = [(11, 12), (13, 14), (15, 16)]

        side = self._resolve_side(lms, joints, MODE_CURL)
        elbow_raw = (self.latest_angles["left_elbow"] if side == 0
                    else self.latest_angles["right_elbow"])
        if elbow_raw is None:
            self.form_feedback = "Arm obscured"
            return

        buf = self._rep_angle_buf[MODE_CURL]
        buf.append(elbow_raw)
        elbow = sum(buf) / len(buf)

        # State naming matches the original: "DOWN" = arm extended (resting
        # / waiting to curl), "UP" = arm contracted (rep just completed).
        if self.states[MODE_CURL] == "DOWN":
            self._rep_extreme[MODE_CURL] = min(self._rep_extreme[MODE_CURL], elbow)

        self._update_confirm_counters(elbow, UP_THRESH, DOWN_THRESH, MODE_CURL)
        CONFIRM = self.CONFIRM_FRAMES

        # Confirmed contraction — this is where the rep is COUNTED
        if (self._down_frames[MODE_CURL] >= CONFIRM
                and self.states[MODE_CURL] == "DOWN"):
            elapsed = now - self.state_timers[MODE_CURL]
            extreme = self._rep_extreme[MODE_CURL]
            if elapsed >= self.min_rep_duration:
                self.reps[MODE_CURL] += 1
                q = self._score_rep(extreme, elapsed, EXCELLENT_CONTRACT,
                                    DOWN_THRESH, 0.25, 2.5)
                self.last_rep_quality[MODE_CURL] = q
                self.rep_quality_history[MODE_CURL].append(q)
                self._log_rep(MODE_CURL, q, elapsed)
                self.form_feedback = "Good Rep!" if q >= 70 else "Partial ROM"
            else:
                self.form_feedback = "Control compression!"
            self.states[MODE_CURL]       = "UP"
            self._locked_side[MODE_CURL] = side

        # Confirmed re-extension — resets for the next rep
        elif (self._up_frames[MODE_CURL] >= CONFIRM
                and self.states[MODE_CURL] == "UP"):
            self.states[MODE_CURL]        = "DOWN"
            self.state_timers[MODE_CURL]  = now
            self._rep_extreme[MODE_CURL]  = elbow
            self._locked_side[MODE_CURL]  = None
            self.form_feedback = "Fully extended."

        # ── Timestamp generator ─────────────────────────────────────────────
    def _generate_ts(self):
        with self._ts_lock:
            ts = int(time.time() * 1000)
            if ts <= self._last_ts_ms:
                ts = self._last_ts_ms + 1
            self._last_ts_ms = ts
            return ts

        # ── Pose utility helpers ────────────────────────────────────────────
    def _get_pose_center(self, landmarks):
        valid = [landmarks[i] for i in (11, 12, 23, 24)
                 if hasattr(landmarks[i], 'visibility')
                 and landmarks[i].visibility > self.visibility_threshold]
        if not valid:
            return 0.5, 0.5
        return (sum(p.x for p in valid) / len(valid),
                sum(p.y for p in valid) / len(valid))

    def _get_pose_distance(self, pose, prev_pose):
        dist_sum, count = 0.0, 0
        for i in range(min(len(pose), len(prev_pose))):
            p1, p2 = pose[i], prev_pose[i]
            v1 = p1.visibility if hasattr(p1, 'visibility') else 1.0
            v2 = p2.visibility if hasattr(p2, 'visibility') else 1.0
            if (v1 > self.visibility_threshold
                    and v2 > self.visibility_threshold):
                dist_sum += ((p1.x-p2.x)**2 + (p1.y-p2.y)**2
                             + (p1.z-p2.z)**2) ** 0.5
                count += 1
        if count == 0:
            cx1, cy1 = self._get_pose_center(pose)
            cx2, cy2 = self._get_pose_center(prev_pose)
            return ((cx1-cx2)**2 + (cy1-cy2)**2) ** 0.5
        return dist_sum / count

    def apply_smoothing(self, current_landmarks):
        if (not self._previous_landmarks
                or len(self._previous_landmarks) != len(current_landmarks)):
            self._previous_landmarks = current_landmarks
            return current_landmarks

        alpha = SMOOTHING_PRESETS[self.current_preset]["alpha"]

        class SmoothedLM:
            def __init__(self, x, y, z, v, p):
                self.x, self.y, self.z = x, y, z
                self.visibility, self.presence = v, p

        smoothed = []
        for curr, prev in zip(current_landmarks, self._previous_landmarks):
            dyn_alpha = alpha * max(0.0, min(1.0, curr.visibility))
            smoothed.append(SmoothedLM(
                dyn_alpha * curr.x + (1 - dyn_alpha) * prev.x,
                dyn_alpha * curr.y + (1 - dyn_alpha) * prev.y,
                dyn_alpha * curr.z + (1 - dyn_alpha) * prev.z,
                curr.visibility,
                curr.presence,
            ))

        self._previous_landmarks = smoothed
        return smoothed

    def get_roi(self, landmarks, w, h):
        if not landmarks:
            return None
        valid = [lm for lm in landmarks
                 if (lm.visibility if hasattr(lm, 'visibility') else 1.0)
                 > self.visibility_threshold]
        if not valid:
            return None
        xs = [lm.x * w for lm in valid]
        ys = [lm.y * h for lm in valid]
        pad = 40
        return (max(0, int(min(xs)) - pad), max(0, int(min(ys)) - pad),
                min(w, int(max(xs)) + pad), min(h, int(max(ys)) + pad))

        # ── Main inference ──────────────────────────────────────────────────
    def find_pose(self, frame):
        h, w = frame.shape[:2]
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        ts    = self._generate_ts()

        with self._ts_lock:
            result      = self.detector.detect_for_video(mp_img, ts)
            hand_result = self.hand_detector.detect_for_video(mp_img, ts)

        roi        = None
        best_idx   = 0
        best_pose  = None
        best_world = None

        if result and result.pose_landmarks and len(result.pose_landmarks) > 0:
            # Match against the previous frame's landmarks to find which
            # detected pose is the one we've been tracking, then use that
            # same index for the world-landmark lookup below — otherwise
            # angle calculations end up using a different person's
            # coordinates than the one on screen.
            if not self._previous_landmarks or len(result.pose_landmarks) == 1:
                best_idx = 0
            else:
                best_idx = min(
                    range(len(result.pose_landmarks)),
                    key=lambda i: self._get_pose_distance(
                        result.pose_landmarks[i], self._previous_landmarks)
                )

            best_pose = result.pose_landmarks[best_idx]

            if (result.pose_world_landmarks
                    and best_idx < len(result.pose_world_landmarks)):
                best_world = result.pose_world_landmarks[best_idx]
            elif result.pose_world_landmarks:
                best_world = result.pose_world_landmarks[0]
        else:
            self._previous_landmarks = None

        exercise_data = {
            "mode":          self.current_mode,
            "rep_count":     self.reps.get(self.current_mode, 0),
            "state":         self.states.get(self.current_mode, "IDLE"),
            "angles":        {},
            "feedback":      "Searching for body...",
            "valid_tracking": False,
        }

        if best_pose and best_world:
            smoothed = self.apply_smoothing(best_pose)

            # Hand-wrist fusion (keeps wrist positions accurate for curl/pushup)
            if (hand_result
                    and hasattr(hand_result, 'hand_landmarks')
                    and hand_result.hand_landmarks):
                left_wrist  = smoothed[15]
                right_wrist = smoothed[16]
                pairings    = []
                for h_idx, hand_lms in enumerate(hand_result.hand_landmarks[:2]):
                    root = hand_lms[0]
                    dl   = ((root.x-left_wrist.x)**2  + (root.y-left_wrist.y)**2)  **0.5
                    dr   = ((root.x-right_wrist.x)**2 + (root.y-right_wrist.y)**2) **0.5
                    pairings.append((dl, h_idx, 15))
                    pairings.append((dr, h_idx, 16))
                pairings.sort(key=lambda x: x[0])
                assigned_hands  = set()
                assigned_wrists = set()
                for dist, h_idx, wrist_idx in pairings:
                    if (h_idx not in assigned_hands
                            and wrist_idx not in assigned_wrists
                            and dist < 0.50):
                        smoothed[wrist_idx].x = \
                            hand_result.hand_landmarks[h_idx][0].x
                        smoothed[wrist_idx].y = \
                            hand_result.hand_landmarks[h_idx][0].y
                        assigned_hands.add(h_idx)
                        assigned_wrists.add(wrist_idx)

            # Keep every detected pose, smoothed best-match at index 0 —
            # tracker_module.py needs all of them for multi-person analytics
            all_poses = [smoothed]
            for i, raw_pose in enumerate(result.pose_landmarks):
                if i != best_idx:
                    all_poses.append(raw_pose)
            result.pose_landmarks = all_poses

            roi = self.get_roi(smoothed, w, h)
            self._update_all_angles(smoothed, best_world)

            # Mode A exercise rep-counting (uses smoothed / best_world only)
            if self.current_mode == MODE_SQUAT:
                self._squat_logic(smoothed)
            elif self.current_mode == MODE_PUSHUP:
                self._pushup_logic(smoothed, w, h)
            elif self.current_mode == MODE_CURL:
                self._curl_logic(smoothed)
            elif self.current_mode == MODE_NONE:
                self.form_feedback = "Ready"

            filtered_angles = {k: v for k, v in self.latest_angles.items()
                               if v is not None}
            hist = self.rep_quality_history.get(self.current_mode)
            avg_quality = (round(sum(hist) / len(hist)) if hist else None)
            exercise_data.update({
                "rep_count":      self.reps.get(self.current_mode, 0),
                "state":          self.states.get(self.current_mode, "IDLE"),
                "angles":         filtered_angles,
                "feedback":       self.form_feedback,
                "valid_tracking": True,
                "last_quality":   self.last_rep_quality.get(self.current_mode),
                "avg_quality":    avg_quality,
            })

        return {
            "result":         result,
            "hand_result":    hand_result,
            "roi":            roi,
            "exercise_data":  exercise_data,
            "frame_w":        w,
            "frame_h":        h,
            "in_frame_shape": frame.shape,
        }

        # ── Drawing — draws all detected people's skeletons ────────────────
    def draw_landmarks(self, frame, dict_data):
        out = frame.copy()
        if not dict_data:
            return out

        result      = dict_data.get("result")
        hand_result = dict_data.get("hand_result")
        roi         = dict_data.get("roi")
        h = dict_data.get("frame_h", frame.shape[0])
        w = dict_data.get("frame_w", frame.shape[1])

        if roi and self.show_roi:
            rx1, ry1, rx2, ry2 = roi
            cv2.rectangle(out, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)

        if result and hasattr(result, 'pose_landmarks') and result.pose_landmarks:
            hand_indices = {17, 18, 19, 20, 21, 22}

            for pose_idx, body in enumerate(result.pose_landmarks):
                # Primary (smoothed, index 0) gets full-bright colour;
                # additional people are slightly dimmer so they're visually
                # distinct but still readable.
                is_primary   = (pose_idx == 0)
                line_color   = (255, 144, 30) if is_primary else (180, 100, 20)
                point_color  = (0, 0, 255)    if is_primary else (0, 100, 200)

                for conn in self.mp_pose.POSE_CONNECTIONS:
                    if conn[0] in hand_indices or conn[1] in hand_indices:
                        continue
                    p1, p2 = body[conn[0]], body[conn[1]]
                    v1 = p1.visibility if hasattr(p1, 'visibility') else 1.0
                    v2 = p2.visibility if hasattr(p2, 'visibility') else 1.0
                    if v1 > self.visibility_threshold and v2 > self.visibility_threshold:
                        if abs(p1.x - p2.x) > 0.88:
                            continue
                        cv2.line(out,
                                 (int(p1.x*w), int(p1.y*h)),
                                 (int(p2.x*w), int(p2.y*h)),
                                 line_color, 2)

                for idx, lm in enumerate(body):
                    if idx in hand_indices:
                        continue
                    lm_v = lm.visibility if hasattr(lm, 'visibility') else 1.0
                    if lm_v > self.visibility_threshold:
                        cv2.circle(out,
                                   (int(lm.x*w), int(lm.y*h)),
                                   4, point_color, -1)

        if (hand_result
                and hasattr(hand_result, 'hand_landmarks')
                and hand_result.hand_landmarks):
            for hand_lms in hand_result.hand_landmarks[:2]:
                for conn in self.mp_hands.HAND_CONNECTIONS:
                    p1, p2 = hand_lms[conn[0]], hand_lms[conn[1]]
                    cv2.line(out,
                             (int(p1.x*w), int(p1.y*h)),
                             (int(p2.x*w), int(p2.y*h)),
                             (0, 255, 100), 2)
                for lm in hand_lms:
                    cv2.circle(out,
                               (int(lm.x*w), int(lm.y*h)),
                               3, (200, 50, 200), -1)

        return out

        # ── Session export ──────────────────────────────────────────────────
    def export_session(self):
        import json, time as _time
        filename = f"session_{int(_time.time())}.json"
        with open(filename, "w") as f:
            json.dump(self.session_data, f, indent=2)
        return filename