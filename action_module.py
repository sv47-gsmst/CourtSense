# action_module.py
#
# All existing classes (JumpDetector, SplitStepDetector, LungeDetector,
# StrokeDetector, classify_movement) are unchanged.
#
# NEW classes added for meaningful feature redesign:
#   RecoveryTracker  — measures time from stroke back to ready position
#   StaminaTracker   — rolling intensity trend (fading / holding / building)
#   RallyCounter     — counts consecutive strokes as a rally
#   check_ready_stance() — interprets angles into coaching cues
# ----------------------------------------------------------------------


import time
import numpy as np
from collections import deque
from typing import Optional




# =========================================================================
# SPEED TIERS  (torso-lengths / second — scale-invariant)
# =========================================================================
# Adult torso ≈ 0.45m, so roughly:
#   SPEED_READY   0.4  ≈ 0.18 m/s  (standing, weight shifts)
#   SPEED_SHUFFLE 1.5  ≈ 0.68 m/s  (small positioning steps)
#   SPEED_SPRINT  4.0  ≈ 1.80 m/s  (explosive movement)
SPEED_READY   = 0.4
SPEED_SHUFFLE = 1.5
SPEED_SPRINT  = 4.0




def classify_movement(speed_torso_per_sec: float) -> str:
    if speed_torso_per_sec > SPEED_SPRINT:
        return "sprinting"
    if speed_torso_per_sec > SPEED_SHUFFLE:
        return "jogging"
    if speed_torso_per_sec > SPEED_READY:
        return "shuffling"
    return "ready"




# =========================================================================
# JUMP DETECTOR
# =========================================================================
class JumpDetector:
    """
    Detects vertical hops via torso-normalised hip height.
    Baseline only updates while grounded — prevents a slow squat from
    registering as a jump. Rise must hold for MIN_RISE_FRAMES before
    counting as airborne (filters single-frame jitter).
    """
    RISE_THRESHOLD_TORSO = 0.18
    MIN_RISE_FRAMES      = 2
    LAND_THRESHOLD_TORSO = 0.08
    COOLDOWN_SEC         = 0.35
    BASELINE_WINDOW      = 10


    def __init__(self):
        self._baseline_px     = None
        self._grounded_hip_px = deque(maxlen=self.BASELINE_WINDOW)
        self._rise_frames     = 0
        self._in_air          = False
        self._last_jump_time  = 0.0
        self.jump_count       = 0
        self.phase            = "grounded"


    def update(self, landmarks, torso_px: float, frame_h: int) -> None:
        def vis(i):
            return getattr(landmarks[i], 'visibility', 1.0)


        hip_idx = [i for i in (23, 24) if vis(i) > 0.3]
        if not hip_idx or torso_px <= 1.0:
            return


        hip_y_px   = (sum(landmarks[i].y for i in hip_idx) / len(hip_idx)) * frame_h
        now        = time.time()


        if self._baseline_px is None:
            self._baseline_px = hip_y_px
            self._grounded_hip_px.append(hip_y_px)
            return


        rise_torso = (self._baseline_px - hip_y_px) / torso_px


        if not self._in_air:
            self._grounded_hip_px.append(hip_y_px)
            self._baseline_px = sum(self._grounded_hip_px) / len(self._grounded_hip_px)
            if rise_torso > self.RISE_THRESHOLD_TORSO:
                self._rise_frames += 1
                if self._rise_frames >= self.MIN_RISE_FRAMES:
                    self._in_air = True
                    self.phase   = "rising"
            else:
                self._rise_frames = 0
                self.phase = "grounded"
        else:
            if rise_torso > self.RISE_THRESHOLD_TORSO:
                self.phase = "peak" if rise_torso > self.RISE_THRESHOLD_TORSO * 1.3 else "rising"
            elif rise_torso < self.LAND_THRESHOLD_TORSO:
                self._in_air = False
                self._rise_frames = 0
                self.phase = "landing"
                if now - self._last_jump_time > self.COOLDOWN_SEC:
                    self.jump_count += 1
                    self._last_jump_time = now
                self._grounded_hip_px.clear()
                self._grounded_hip_px.append(hip_y_px)
                self._baseline_px = hip_y_px
            else:
                self.phase = "rising"




# =========================================================================
# SPLIT-STEP DETECTOR
# =========================================================================
class SplitStepDetector:
    """
    Detects the bounce-on-balls-of-feet that tennis/pickleball players use
    to load before reacting. Requires a local hip-y dip (rise-then-fall)
    in a short window with low horizontal travel and a cooldown.
    """
    WINDOW_SEC      = 0.35
    MIN_AMPLITUDE   = 0.10
    MAX_HORIZ_DRIFT = 0.6
    COOLDOWN_SEC    = 0.4


    def __init__(self):
        self._buf: deque = deque()
        self._last_count_time = 0.0
        self.split_step_count = 0


    def update(self, landmarks, torso_px: float, frame_w: int, frame_h: int) -> bool:
        def vis(i):
            return getattr(landmarks[i], 'visibility', 1.0)


        hip_idx = [i for i in (23, 24) if vis(i) > 0.3]
        if not hip_idx or torso_px <= 1.0:
            return False


        hip_x = (sum(landmarks[i].x for i in hip_idx) / len(hip_idx)) * frame_w
        hip_y = (sum(landmarks[i].y for i in hip_idx) / len(hip_idx)) * frame_h
        now   = time.time()


        self._buf.append((now, hip_y, hip_x))
        while self._buf and now - self._buf[0][0] > self.WINDOW_SEC:
            self._buf.popleft()


        if len(self._buf) < 4:
            return False


        ys = [b[1] for b in self._buf]
        xs = [b[2] for b in self._buf]


        vertical_range   = (max(ys) - min(ys)) / torso_px
        horizontal_range = (max(xs) - min(xs)) / torso_px
        min_idx          = ys.index(min(ys))
        is_local_dip     = 0 < min_idx < len(ys) - 1


        if (vertical_range > self.MIN_AMPLITUDE
                and horizontal_range < self.MAX_HORIZ_DRIFT
                and is_local_dip
                and now - self._last_count_time > self.COOLDOWN_SEC):
            self.split_step_count += 1
            self._last_count_time = now
            self._buf.clear()
            return True


        return False




# =========================================================================
# LUNGE DETECTOR
# =========================================================================
class LungeDetector:
    """
    One knee bends deeply (DEPTH_THRESHOLD) while the other stays
    relatively straight (ASYMMETRY_THRESHOLD difference), sustained for
    MIN_FRAMES, with a cooldown. Prevents the athletic ready stance from
    triggering (both knees similarly bent).
    """
    DEPTH_THRESHOLD     = 110.0
    ASYMMETRY_THRESHOLD = 25.0
    MIN_FRAMES          = 3
    COOLDOWN_SEC        = 0.6


    def __init__(self):
        self._frame_count     = 0
        self._last_count_time = 0.0
        self.lunge_count      = 0
        self.active           = False
        # "left", "right", or "--". The knee with the deeper bend determines
        # direction — the lead knee bends into the lunge in most cases.
        self.last_lunge_dir   = "--"


    def update(self, left_knee, right_knee) -> bool:
        if left_knee is None or right_knee is None:
            self._frame_count = 0
            self.active = False
            return False


        deepest = min(left_knee, right_knee)
        diff    = abs(left_knee - right_knee)


        if deepest < self.DEPTH_THRESHOLD and diff > self.ASYMMETRY_THRESHOLD:
            self._frame_count += 1
        else:
            self._frame_count = 0
            self.active = False


        now = time.time()
        if (self._frame_count >= self.MIN_FRAMES
                and not self.active
                and now - self._last_count_time > self.COOLDOWN_SEC):
            self.active = True
            self.lunge_count += 1
            self._last_count_time = now
            # Lead knee (deeper bend) indicates direction
            self.last_lunge_dir = "left" if left_knee < right_knee else "right"
            return True


        return False




# =========================================================================
# STROKE DETECTOR
# =========================================================================
class StrokeDetector:
    """
    Detects a racket/paddle swing via wrist speed in torso-lengths/sec.
    Wrist speed during a real stroke is far higher than repositioning.
    Also records wrist speed at detection for quality feedback.
    """
    SPEED_THRESHOLD = 4.0
    COOLDOWN_SEC    = 0.45


    def __init__(self):
        self._prev_pos: dict = {}
        self._last_count_time = 0.0
        self.stroke_count = 0
        self.last_wrist_speed = 0.0
        # "forehand", "backhand", or "--" — the triggering wrist on its own
        # side of the body counts as forehand, crossed past the shoulder
        # midline counts as backhand. Approximate but useful.
        self.last_stroke_side = "--"
        self._triggering_idx  = -1    # landmark index that fired (15 or 16)


    def update(self, landmarks, torso_px: float, frame_w: int, frame_h: int) -> bool:
        def vis(i):
            return getattr(landmarks[i], 'visibility', 1.0)


        if torso_px <= 1.0:
            return False


        now = time.time()
        max_speed = 0.0
        triggering_idx = -1


        for idx in (15, 16):
            if vis(idx) < 0.4:
                continue
            x = landmarks[idx].x * frame_w
            y = landmarks[idx].y * frame_h
            if idx in self._prev_pos:
                px, py, pt = self._prev_pos[idx]
                dt = max(now - pt, 1e-3)
                dist  = float(np.hypot(x - px, y - py))
                speed = (dist / dt) / torso_px
                if speed > max_speed:
                    max_speed = speed
                    triggering_idx = idx
            self._prev_pos[idx] = (x, y, now)


        if (max_speed > self.SPEED_THRESHOLD
                and now - self._last_count_time > self.COOLDOWN_SEC):
            self.stroke_count += 1
            self.last_wrist_speed = max_speed
            self._last_count_time = now
            # Determine forehand vs backhand from wrist position relative to
            # the shoulder midline at the moment of stroke.
            # Right wrist (16) on the right of centre = forehand, crossed left = backhand.
            # Left wrist (15) on the left of centre = forehand, crossed right = backhand.
            if triggering_idx >= 0:
                self._triggering_idx = triggering_idx
                wx = landmarks[triggering_idx].x
                vs11 = getattr(landmarks[11], 'visibility', 0)
                vs12 = getattr(landmarks[12], 'visibility', 0)
                if vs11 > 0.3 and vs12 > 0.3:
                    centre_x = (landmarks[11].x + landmarks[12].x) / 2
                else:
                    centre_x = 0.5
                if triggering_idx == 16:   # right wrist
                    self.last_stroke_side = "forehand" if wx > centre_x else "backhand"
                else:                      # left wrist
                    self.last_stroke_side = "forehand" if wx < centre_x else "backhand"
            return True


        return False




# =========================================================================
# TORSO CALIBRATOR
# =========================================================================
class TorsoCalibrator:
    """
    Collects torso-length samples during the first SETTLE_SEC seconds of
    an analytics session and locks a stable per-player baseline value.


    tracker_module's PersonTrack smooths torso_px with an EMA, so the
    very first frames (before the player settles into a consistent
    camera distance) can pull the baseline high or low — and every
    speed reading is divided by this value, so a bad baseline throws
    off the whole session.


    We collect the median torso reading over SETTLE_SEC seconds (median,
    not mean, so a few bad frames from partial occlusion don't skew it),
    then lock it in. The pipeline shows a "calibrating..." overlay until
    `calibrated` is True instead of showing junk numbers.
    """
    SETTLE_SEC   = 5.0    # seconds of data to collect before locking
    MIN_SAMPLES  = 30     # require at least this many frames regardless of time
    CLAMP_MIN    = 30.0   # discard readings below this (partial detection)
    CLAMP_MAX    = 400.0  # discard readings above this (implausibly close)


    def __init__(self):
        self._samples: list   = []
        self._start:   float  = 0.0
        self._started: bool   = False
        self.calibrated: bool = False
        self.baseline_px: float = 0.0   # locked median torso length in pixels
        self.progress:    float = 0.0   # 0.0 → 1.0 progress bar for the overlay


    def feed(self, torso_px: float, timestamp: float) -> None:
        """Call once per analytics frame with the primary track's torso_px."""
        if self.calibrated:
            return


        if not self._started:
            self._start   = timestamp
            self._started = True


        elapsed = timestamp - self._start
        self.progress = min(elapsed / self.SETTLE_SEC, 1.0)


        if self.CLAMP_MIN < torso_px < self.CLAMP_MAX:
            self._samples.append(torso_px)


        if (elapsed >= self.SETTLE_SEC
                and len(self._samples) >= self.MIN_SAMPLES):
            self.baseline_px = float(sorted(self._samples)[len(self._samples) // 2])
            self.calibrated  = True


    def reset(self) -> None:
        self._samples  = []
        self._started  = False
        self.calibrated  = False
        self.baseline_px = 0.0
        self.progress    = 0.0


    def draw_overlay(self, img, fw: int, fh: int) -> None:
        """
        Draws a centred calibration notice with a progress bar.
        Only called when calibrated is False.
        """
        import cv2 as _cv2
        pw, ph = 360, 90
        ox     = (fw - pw) // 2
        oy     = (fh - ph) // 2


        overlay = img.copy()
        _cv2.rectangle(overlay, (ox, oy), (ox+pw, oy+ph), (15, 20, 25), -1)
        _cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
        _cv2.rectangle(img, (ox, oy), (ox+pw, oy+ph), (60, 80, 80), 1)


        _cv2.putText(img, "Calibrating — stand still",
                     (ox+18, oy+28), _cv2.FONT_HERSHEY_DUPLEX,
                     0.55, (0, 220, 255), 1, _cv2.LINE_AA)
        _cv2.putText(img, "Analytics starts in a moment…",
                     (ox+18, oy+48), _cv2.FONT_HERSHEY_DUPLEX,
                     0.42, (160, 160, 160), 1, _cv2.LINE_AA)


        # Progress bar
        bar_x, bar_y = ox+18, oy+64
        bar_w, bar_h = pw-36, 12
        _cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h),
                        (50, 50, 50), -1)
        filled = int(bar_w * self.progress)
        if filled > 0:
            _cv2.rectangle(img, (bar_x, bar_y),
                            (bar_x+filled, bar_y+bar_h),
                            (0, 200, 140), -1)
        pct = int(self.progress * 100)
        _cv2.putText(img, f"{pct}%",
                     (bar_x + bar_w + 6, bar_y + 10),
                     _cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                     (150, 150, 150), 1, _cv2.LINE_AA)




# =========================================================================
# RECOVERY TRACKER
# =========================================================================
class RecoveryTracker:
    """
    After each stroke, measures how long until the player returns to a
    ready (near-stationary) position.


    Coaching context: a good recovery in tennis/pickleball is typically
    under 1.0s. Slow recovery (> 1.5s) means the player is not resetting
    quickly enough between shots.


    Implementation: starts a timer on notify_stroke(), stops it when
    speed has been below SPEED_READY for SETTLE_FRAMES consecutive frames.
    SETTLE_FRAMES prevents a single slow frame mid-movement from falsely
    ending the recovery clock early.
    """
    SETTLE_FRAMES  = 6     # ~0.2s at 30fps — must be stationary this long
    FAST_THRESHOLD = 0.8   # seconds — "fast" recovery
    SLOW_THRESHOLD = 1.5   # seconds — "slow" recovery
    MAX_VALID_SEC  = 12.0  # discard if player never settled (e.g. long rally)


    def __init__(self):
        self._recovering      = False
        self._stroke_time     = 0.0
        self._settle_count    = 0
        self.last_recovery_sec: Optional[float] = None
        self.avg_recovery_sec:  Optional[float] = None
        self._history: deque  = deque(maxlen=20)


    def notify_stroke(self, timestamp: float) -> None:
        """Call this immediately when StrokeDetector fires."""
        self._recovering   = True
        self._stroke_time  = timestamp
        self._settle_count = 0


    def update(self, speed: float, timestamp: float) -> bool:
        """
        Call every analytics frame.
        Returns True the moment a recovery completes.
        """
        if not self._recovering:
            return False


        # Safety: abandon if the player never settles for too long
        if timestamp - self._stroke_time > self.MAX_VALID_SEC:
            self._recovering = False
            return False


        if speed < SPEED_READY:
            self._settle_count += 1
        else:
            self._settle_count = 0


        if self._settle_count >= self.SETTLE_FRAMES:
            elapsed = timestamp - self._stroke_time
            if 0.15 < elapsed < self.MAX_VALID_SEC:
                self.last_recovery_sec = round(elapsed, 2)
                self._history.append(elapsed)
                self.avg_recovery_sec = round(
                    sum(self._history) / len(self._history), 2)
            self._recovering   = False
            self._settle_count = 0
            return True


        return False


    @property
    def quality(self) -> str:
        """Plain-English quality of the last measured recovery."""
        if self.last_recovery_sec is None:
            return "--"
        if self.last_recovery_sec < self.FAST_THRESHOLD:
            return "fast"
        if self.last_recovery_sec < self.SLOW_THRESHOLD:
            return "good"
        return "slow"


    @property
    def quality_color(self) -> tuple:
        """BGR colour for quality label."""
        q = self.quality
        if q == "fast":   return (50, 255, 100)   # green
        if q == "good":   return (0, 200, 255)     # yellow
        if q == "slow":   return (0, 80, 255)      # red
        return (150, 150, 150)




# =========================================================================
# STAMINA TRACKER
# =========================================================================
class StaminaTracker:
    """
    Compares current 30-second rolling average speed against the player's
    early-session baseline to detect intensity trends.


    FADING   — recent avg < 75% of early avg  (tiring)
    HOLDING  — recent avg within 75-115% of early avg
    BUILDING — recent avg > 115% of early avg  (warming up / pushing harder)


    Also maintains a sparkline buffer (20 × 5-second bucket averages) for
    drawing a mini bar chart in the HUD.
    """
    WINDOW_SEC       = 30.0   # rolling window for "current" average
    BUCKET_SEC       = 5.0    # time bucket for each sparkline bar
    FADING_RATIO     = 0.75
    BUILDING_RATIO   = 1.15
    BASELINE_MIN_SEC = 30.0   # minimum session time before trend is meaningful


    def __init__(self):
        self._buf: deque              = deque()   # (timestamp, speed)
        self.sparkline: deque         = deque(maxlen=20)
        self._bucket_speeds: list     = []
        self._bucket_start:  float    = 0.0
        self._session_start: float    = 0.0
        self._early_avg: Optional[float] = None
        self.current_avg: float       = 0.0
        self.trend: str               = "building"
        self._started                 = False


    def update(self, speed: float, timestamp: float) -> None:
        if not self._started:
            self._started        = True
            self._session_start  = timestamp
            self._bucket_start   = timestamp


        self._buf.append((timestamp, speed))


        # Purge entries older than WINDOW_SEC
        while self._buf and timestamp - self._buf[0][0] > self.WINDOW_SEC:
            self._buf.popleft()


        # Rolling average
        if self._buf:
            self.current_avg = sum(s for _, s in self._buf) / len(self._buf)


        # Sparkline bucket accumulation
        self._bucket_speeds.append(speed)
        if timestamp - self._bucket_start >= self.BUCKET_SEC:
            if self._bucket_speeds:
                self.sparkline.append(
                    sum(self._bucket_speeds) / len(self._bucket_speeds))
            self._bucket_speeds = []
            self._bucket_start  = timestamp


        # Lock in baseline after BASELINE_MIN_SEC
        session_elapsed = timestamp - self._session_start
        if self._early_avg is None and session_elapsed >= self.BASELINE_MIN_SEC:
            self._early_avg = self.current_avg


        # Compute trend
        if self._early_avg and self._early_avg > 0.05:
            ratio = self.current_avg / self._early_avg
            if ratio < self.FADING_RATIO:
                self.trend = "fading"
            elif ratio > self.BUILDING_RATIO:
                self.trend = "building"
            else:
                self.trend = "holding"
        else:
            self.trend = "building"   # default before baseline is established


    @property
    def trend_color(self) -> tuple:
        """BGR colour for trend label."""
        if self.trend == "fading":   return (0, 80, 255)    # red
        if self.trend == "holding":  return (0, 200, 255)   # yellow
        if self.trend == "building": return (50, 255, 100)  # green
        return (150, 150, 150)




# =========================================================================
# RALLY COUNTER
# =========================================================================
class RallyCounter:
    """
    Counts consecutive strokes as a rally. A rally ends when more than
    RALLY_BREAK_SEC passes without a stroke being detected.


    Designed for both solo practice (counts all strokes) and two-player
    use (strokes from either player count toward the same rally).


    'rally_just_ended' stays True for FLASH_SEC after a rally ends so
    the HUD can briefly display "RALLY ENDED — N strokes" before clearing.
    """
    RALLY_BREAK_SEC = 3.0    # gap without stroke = end of rally
    FLASH_SEC       = 2.0    # how long "rally ended" message stays on screen


    def __init__(self):
        self.current_rally:  int   = 0
        self.longest_rally:  int   = 0
        self.total_rallies:  int   = 0
        self._last_stroke_time: Optional[float] = None
        self.rally_just_ended: bool = False
        self._ended_time: Optional[float] = None
        self._in_rally: bool = False


    def notify_stroke(self, timestamp: float) -> None:
        """Call immediately when StrokeDetector fires."""
        if (self._last_stroke_time is not None
                and timestamp - self._last_stroke_time < self.RALLY_BREAK_SEC):
            # Continuation of current rally
            self.current_rally    += 1
            self.rally_just_ended  = False
        else:
            # New rally starting
            if self._in_rally:
                self.total_rallies += 1
            self.current_rally    = 1
            self._in_rally        = True
            self.rally_just_ended = False


        if self.current_rally > self.longest_rally:
            self.longest_rally = self.current_rally


        self._last_stroke_time = timestamp


    def update(self, timestamp: float) -> None:
        """Call every analytics frame to detect rally end."""
        if (self._in_rally
                and self._last_stroke_time is not None
                and timestamp - self._last_stroke_time > self.RALLY_BREAK_SEC
                and not self.rally_just_ended):
            self.rally_just_ended = True
            self._ended_time      = timestamp
            self._in_rally        = False
            self.total_rallies   += 1


        # Clear flash after FLASH_SEC
        if (self.rally_just_ended
                and self._ended_time is not None
                and timestamp - self._ended_time > self.FLASH_SEC):
            self.rally_just_ended = False




# =========================================================================
# READY STANCE CHECKER
# =========================================================================
def check_ready_stance(angles: dict, speed: float) -> Optional[dict]:
    """
    When the player is stationary (speed < SPEED_READY), interprets joint
    angles into plain-English coaching cues.


    Returns None when the player is moving (don't assess stance mid-movement).
    Returns a dict: { "knees": (quality, cue), "lean": (quality, cue) }
    where quality is "good" | "ok" | "poor" and cue is a short string.


    Good ready stance targets for tennis/pickleball:
      Knees: 130-158° (softly bent, not locked straight, not squatting)
      Torso lean: 5-22° forward from vertical (athletic forward weight)
    """
    if speed >= SPEED_READY:
        return None


    result = {}


    # Knee bend — prefer average of both sides, fall back to one
    l_knee = angles.get("left_leg")
    r_knee = angles.get("right_leg")
    if l_knee is not None and r_knee is not None:
        knee = (l_knee + r_knee) / 2
    elif l_knee is not None:
        knee = l_knee
    elif r_knee is not None:
        knee = r_knee
    else:
        knee = None


    if knee is not None:
        if knee > 168:
            result["knees"] = ("poor", "bend knees — legs too straight")
        elif knee > 158:
            result["knees"] = ("ok",   "soften knees a touch")
        elif 130 <= knee <= 158:
            result["knees"] = ("good",  "knees good \u2713")
        else:
            result["knees"] = ("ok",   "knees slightly over-bent")


    # Torso forward lean from vertical (degrees)
    lean = angles.get("torso_lean")
    if lean is not None:
        if lean < 3:
            result["lean"] = ("ok",   "lean slightly forward")
        elif 3 <= lean <= 22:
            result["lean"] = ("good",  "lean good \u2713")
        else:
            result["lean"] = ("ok",   "stand a little taller")


    return result if result else None