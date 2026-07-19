from __future__ import annotations
import time
from dataclasses import dataclass, field
from typing import Optional, Any, Dict

@dataclass
class SystemState:
    ai_loaded: bool = False
    ui_visible: bool = False
    inference_active: bool = False
    warmup_active: bool = False
    paused: bool = False
    detector: Any = None
    last_detector_rep_count: int = 0
    last_detector_mode: Any = None

    persistent_reps: Dict = field(default_factory=dict)

    show_help: bool = False
    help_size: int = 1
    help_scroll: int = 0
    compact_hud: bool = False

    model_idx: int = 0
    smoothing_idx: int = 1

    presence_grace_max: int = 5

    toast_text: str = ""
    toast_expiry: float = 0.0

    global_frame_counter: int = 0
    
    gesture_proc: Any = None

    analytics_mode: bool = False
    analytics_used_this_run: bool = False

    show_heatmap: bool = False
    show_recovery: bool = False
    show_stamina: bool = False
    show_stance: bool = False
    show_stats: bool = False
    show_dashboard: bool = False
    show_footwork: bool = False
    show_stroke_fb: bool = False
    show_rally: bool = False
    show_ball: bool = False

    mp_tracker: Any = None
    body_analytics: Any = None
    heatmap: Any = None
    jump_detector: Any = None
    split_detector: Any = None
    lunge_detector: Any = None
    stroke_detector: Any = None
    recovery: Any = None
    stamina: Any = None
    rally: Any = None
    torso_cal: Any = None
    ball_tracker: Any = None
    audio_coach: Any = None
    dashboard: Any = None
    exporter: Any = None

    stroke_fb_text: str = ""
    stroke_fb_until: float = 0.0

    zone_counts: Dict = field(default_factory=lambda: {"baseline":0, "mid": 0, "net": 0})

    analytics_session_start: float = 0.0
    history_lines: list = field(default_factory=list)
    history_show_until: float = 0.0

    inference_time_ema: float = 0.0
    recording: bool = False
    video_writer: Any = None
    recording_filename: str = ""
    court_cal: Any = None
    calibrating: bool = False
    pose_quality: Any = None
    racket_detector: Any = None
    show_racket: bool = False

    def __pose_init__(self):
        if not self.persistent_reps:
            self.persistent_reps = {
                0: 0, 1: 0, 2: 0, 3: 0,
                '0': 0, '1': 0, '2': 0, '3': 0, 
            }
    
    TOAST_DURATIONS = {"info": 2.0, "success": 3.5, "warning": 4.0}

    def show_toast(self, msg: str, category: str = "info") -> None:
        duration = self.TOAST_DURATIONS.get(category, 2.0)
        self.toast_text = msg
        self.toast_expiry = time.time() + duration

    def hud_dict(self) -> dict:
        """Returns the state dict that draw_hud() / draw_analytics_bar() expect."""
        return {
            'warmup_active': self.warmup_active,
            'analytics_mode': self.analytics_mode,
            'ui_visible': self.ui_visible,
            'ai_loaded': self.ai_loaded,
            'inference_active': self.inference_active,
            'compact_hud': self.compact_hud,
            'show_help': self.show_help,
            'help_size': self.help_size,
            'help_scroll': self.help_scroll,
            'toast_text': self.toast_text,
            'detector_preset': (self.detector.current_preset if self.detector else ''),
            'recording': self.recording,
            'show_heatmap': self.show_heatmap,
            'show_recovery': self.show_recovery,
            'show_stamina': self.show_stamina,
            'show_stance': self.show_stance,
            'show_stats': self.show_stats,
            'show_dashboard': self.show_dashboard,
            'show_footwork': self.show_footwork,
            'show_stroke_fb': self.show_stroke_fb,
            'show_rally': self.show_rally,
            'show_ball': self.show_ball,
            'show_racket': self.show_racket,
            'history_lines': self.history_lines,
            'history_show_until': self.history_show_until,
        }
    
    def reset_analytics(self) -> None:
        if self.pose_quality: self.pose_quality.reset()
        if self.mp_tracker: self.mp_tracker.reset()
        if self.body_analytics: self.body_analytics.reset()
        if self.heatmap: self.heatmap.reset()
        if self.ball_tracker: self.ball_tracker.reset()
        if self.jump_detector: self.jump_detector.__init__()
        if self.split_detector: self.split_detector.__init__()
        if self.lunge_detector: self.lunge_detector.__init__()
        if self.stroke_detector: self.stroke_detector.__init__()
        if self.recovery: self.recovery.__init__()
        if self.stamina: self.stamina.__init__()
        if self.rally: self.rally.__init__()
        if self.torso_cal: self.torso_cal.reset()
        if self.ball_tracker: self.ball_tracker.reset()
        if self.dashboard: self.dashboard.reset()
        if self.exporter: self.exporter.reset()
        self.zone_counts = {"baseline": 0, "mid": 0, "net": 0}
        self.stroke_fb_text = ""
        self.stroke_fb_until = 0.0
        