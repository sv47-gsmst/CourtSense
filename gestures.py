import cv2, time, numpy as np, mediapipe as mp
from typing import Optional, Dict, Any

GESTURE_COOLDOWN  = 0.8
POSE_HOLD_DURATION = 0.5

class GestureProcessor:
    def __init__(self):
        _mp = mp.solutions.hands
        self.engine = _mp.Hands(static_image_mode=False, max_num_hands=2,
                                min_detection_confidence=0.4,
                                min_tracking_confidence=0.75)
        self._last_time   = 0.0
        self._active_mode: Optional[str] = None
        self._mode_start  = 0.0

    def process(self, frame: np.ndarray, warmup_active: bool,
                analytics_mode: bool, detector) -> dict:
        """
        Returns dict with keys:
          toggle_warmup: bool   - palms-together gesture fired
          set_mode: str|None    - '1'/'2'/'3' exercise mode to set
          blocked_msg: str|None - message to show when gesture blocked
        """
        result = {'toggle_warmup': False, 'set_mode': None, 'blocked_msg': None}
        now = time.time()
        if now - self._last_time < GESTURE_COOLDOWN:
            return result

        small = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_LINEAR)
        rgb   = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res   = self.engine.process(rgb)

        if not res.multi_hand_landmarks or not res.multi_handedness:
            self._active_mode = None
            self._mode_start  = 0.0
            return result

        hand_map: Dict[str, Any] = {}
        for lm, hd in zip(res.multi_hand_landmarks, res.multi_handedness):
            raw   = hd.classification[0].label
            label = "Right" if raw == "Left" else "Left"
            hand_map[label] = lm.landmark

        if "Left" in hand_map and "Right" in hand_map:
            lh, rh = hand_map["Left"], hand_map["Right"]
            if (lh[0].x > rh[0].x and abs(lh[0].x - rh[0].x) < 0.15
                    and abs(lh[0].y - rh[0].y) < 0.15 and lh[0].y < 0.85):
                if analytics_mode:
                    result['blocked_msg'] = "Turn off Analytics (Tab) before using Warmup"
                else:
                    result['toggle_warmup'] = True
                    if detector:
                        detector._previous_landmarks = None
                self._last_time = now
                return result

        lms = res.multi_hand_landmarks[0].landmark
        ups = [lms[8].y < lms[6].y, lms[12].y < lms[10].y,
               lms[16].y < lms[14].y, lms[20].y < lms[18].y]
        dm  = {1: '1', 2: '2', 3: '3'}.get(sum(ups))

        if dm:
            if self._active_mode == dm:
                if now - self._mode_start >= POSE_HOLD_DURATION:
                    result['set_mode'] = dm
                    self._active_mode = None
                    self._mode_start  = 0.0
                    self._last_time   = now
            else:
                self._active_mode = dm
                self._mode_start  = now
        else:
            self._active_mode = None
            self._mode_start  = 0.0
        return result