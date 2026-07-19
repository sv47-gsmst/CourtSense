# -------------------------------------------------------------------------
# FILE: main.py
#
# STARTUP
#   python main.py          (default camera)
#   python main.py --cam 1  (alternate camera)
#   python main.py --rtsp rtsp://...  (phone camera)
#
# MODE A  [i] boot AI → [Space] Warmup (rep counter)
# MODE B  [i] boot AI → [Tab]   Analytics (tennis/pickleball)
# Space and Tab are mutually exclusive.
# -------------------------------------------------------------------------

import cv2, time, sys, logging, argparse, numpy as np, threading, queue
from typing import Optional, Any

from state        import SystemState
from court_calibration import CourtCalibrator
from pose_module  import PoseDetector, MODE_NONE, MODE_SQUAT, MODE_PUSHUP, MODE_CURL
from gestures     import GestureProcessor
from hud          import draw_hud, draw_session_end_overlay, draw_paused_overlay
from pipeline_b   import run_analytics_pipeline

# ── Analytics module imports (graceful fallback) ───────────────────────────
try:
    from tracker_module   import MultiPersonTracker
    from analytics_module import BodyAnalytics, calculate_body_angles
    from heatmap_module   import HeatmapAccumulator
    from action_module    import (JumpDetector, SplitStepDetector, LungeDetector,
                                  StrokeDetector, RecoveryTracker, StaminaTracker,
                                  RallyCounter, TorsoCalibrator,
                                  check_ready_stance, classify_movement,
                                  SPEED_READY, SPEED_SHUFFLE, SPEED_SPRINT)
    from dashboard_module import LiveDashboard, export_html_report
    from export_module    import (SessionExporter, load_history, format_history_lines)
    _ANALYTICS_AVAILABLE = True
except ImportError as _ie:
    _ANALYTICS_AVAILABLE = False
    logging.warning(f"Analytics modules not found — Mode B disabled. ({_ie})")

try:
    from ball_tracker import BallTracker
    _BALL_AVAILABLE = True
except ImportError:
    _BALL_AVAILABLE = False

try:
    from audio_module import (AudioCoach, cue_stroke, cue_lunge,
                              cue_split_step, cue_jump, cue_stance, cue_rally_end)
    _AUDIO_AVAILABLE = True
except ImportError:
    _AUDIO_AVAILABLE = False

try:
    from racket_detector import RacketDetector
    _RACKET_MODULE_AVAILABLE = True
except ImportError:
    _RACKET_MODULE_AVAILABLE = False


# =========================================================================
# CONFIGURATION
# =========================================================================
WINDOW_NAME   = "Pose Tracking HUD"
MODEL_PRESETS = ["balanced", "low_latency", "high_accuracy"]
SMOOTH_KEYS   = ["RESPONSIVE", "BALANCED", "SMOOTH"]
SHIFT_KEY_MAP = {
    ord('!'): "S1", ord('@'): "S2", ord('#'): "S3",
    ord('$'): "S4", ord('%'): "S5", ord('^'): "S6",
    ord('&'): "S7", ord('*'): "S8", ord('('): "S9",
    ord(')'): "S0",
}

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)


# =========================================================================
# ANALYTICS INIT / RESET  (methods on SystemState, wired here for access
# to module-level imports that state.py doesn't import)
# =========================================================================
def init_analytics(s: SystemState) -> None:
    if not _ANALYTICS_AVAILABLE:
        return
    if s.mp_tracker is not None:
        return   # already initialised

    s.mp_tracker      = MultiPersonTracker()
    s.body_analytics  = BodyAnalytics()
    s.heatmap         = HeatmapAccumulator()
    s.jump_detector   = JumpDetector()
    s.split_detector  = SplitStepDetector()
    s.lunge_detector  = LungeDetector()
    s.stroke_detector = StrokeDetector()
    s.recovery        = RecoveryTracker()
    s.stamina         = StaminaTracker()
    s.rally           = RallyCounter()
    s.torso_cal       = TorsoCalibrator()
    if _BALL_AVAILABLE:
        s.ball_tracker = BallTracker()
    if _RACKET_MODULE_AVAILABLE:
        s.racket_detector = RacketDetector()
        if s.racket_detector.available:
            logger.info("Racket detector loaded (COCO SSD MobileNet v2).")
        else:
            logger.info("Racket detector module present but model weights "
                       "not found in models/ — racket detection inactive.")
    if _AUDIO_AVAILABLE:
        s.audio_coach = AudioCoach()
        s.audio_coach.start()
    s.dashboard      = LiveDashboard()
    s.exporter       = SessionExporter()
    s.analytics_session_start = time.time()
    s.zone_counts    = {"baseline": 0, "mid": 0, "net": 0}
    logger.info("Analytics objects initialised.")


def update_persistent_reps(s: SystemState, dict_data: Optional[dict]) -> None:
    if not dict_data or "exercise_data" not in dict_data:
        return
    ex     = dict_data["exercise_data"]
    c_mode = ex.get("mode", MODE_NONE)
    reps   = ex.get("rep_count", 0)
    if c_mode != s.last_detector_mode:
        s.last_detector_mode      = c_mode
        s.last_detector_rep_count = reps
    if reps > s.last_detector_rep_count:
        diff = reps - s.last_detector_rep_count
        s.persistent_reps[c_mode] = s.persistent_reps.get(c_mode, 0) + diff
        s.last_detector_rep_count = reps
    elif reps < s.last_detector_rep_count:
        s.last_detector_rep_count = reps
    ex["rep_count"] = s.persistent_reps.get(c_mode, 0)


def _window_mouse_handler(event, x, y, flags, s: SystemState) -> None:
    """Routes mouse wheel to help-panel scrolling, everything else to
    court calibration's click handler."""
    if event == cv2.EVENT_MOUSEWHEEL and s.show_help:
        delta = flags >> 16
        if delta > 0:
            s.help_scroll = max(0, s.help_scroll - 3)
        else:
            s.help_scroll = s.help_scroll + 3
        return
    if s.court_cal:
        s.court_cal.on_mouse(event, x, y, flags, None)


def start_recording(s: SystemState, fw: int, fh: int, fps: float = 20.0) -> None:
    ts = time.strftime("%Y%m%d_%H%M%S")
    s.recording_filename = f"recording_{ts}.avi"
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    write_fps = max(10.0, min(fps, 60.0))   # clamp to a sane range
    s.video_writer = cv2.VideoWriter(s.recording_filename, fourcc, write_fps, (fw, fh))
    s.recording    = True


def stop_recording(s: SystemState) -> None:
    if s.video_writer:
        s.video_writer.release()
        s.video_writer = None
    s.recording = False


def _toggle_flag(s: SystemState, attr: str, label: str) -> None:
    val = not getattr(s, attr)
    setattr(s, attr, val)
    s.show_toast(f"{label}: {'ON' if val else 'OFF'}")


def _build_session_summary(s: SystemState) -> dict:
    """Real event counts + active% + avg speed, pulled from the live
    detector objects — this is what actually gets persisted to disk
    now, instead of trying to infer counts later from a field that
    never contained them."""
    ba = s.body_analytics
    total_t = (ba.time_idle + ba.time_moving) if ba else 0
    active_pct = (100.0 * ba.time_moving / total_t) if total_t > 0 else 0.0
    return {
        "strokes":    s.stroke_detector.stroke_count    if s.stroke_detector else 0,
        "lunges":     s.lunge_detector.lunge_count      if s.lunge_detector  else 0,
        "splits":     s.split_detector.split_step_count if s.split_detector  else 0,
        "jumps":      s.jump_detector.jump_count        if s.jump_detector   else 0,
        "active_pct": active_pct,
        "avg_speed":  ba.avg_speed if ba else 0.0,
    }


# ── Threaded inference ──────────────────────────────────────────────────
# Two queues decouple camera capture+display from MediaPipe inference.
#
# _infer_q  (maxsize=1): main thread puts frames here for the worker.
#           maxsize=1 means if the worker is busy the previous frame is
#           dropped — we always process the LATEST frame, never a backlog.
#
# _result_q (maxsize=1): worker puts completed results here.
#           main thread polls non-blocking; on miss it reuses the previous
#           result so the HUD keeps drawing at full camera fps.
#
# Thread safety:
#   - The worker ONLY calls s.detector.find_pose() (which has its own
#     internal _ts_lock) and writes to _result_q.
#   - Everything else (update_persistent_reps, draw_landmarks, key handler,
#     state mutations) stays on the main thread.

_infer_q:  queue.Queue = queue.Queue(maxsize=1)
_result_q: queue.Queue = queue.Queue(maxsize=1)
_infer_thread: Optional[threading.Thread] = None


def _inference_worker(s: SystemState) -> None:
    """Background daemon: pull frame → find_pose → push result."""
    logger.info("Inference thread started.")
    while True:
        item = _infer_q.get()
        if item is None:        # shutdown sentinel
            logger.info("Inference thread stopping.")
            break
        frame, seq = item
        if not s.ai_loaded or not s.inference_active or s.detector is None:
            continue
        try:
            t0     = time.time()
            result = s.detector.find_pose(frame)
            s.inference_time_ema = (0.9 * s.inference_time_ema
                                    + 0.1 * (time.time() - t0))
        except Exception as e:
            logger.debug(f"Inference error: {e}")
            result = None
        if result is None:
            continue
        # Drop stale result if main thread hasn't consumed the previous one
        try:
            _result_q.put_nowait((result, frame, seq))
        except queue.Full:
            pass


def _start_inference_thread(s: SystemState) -> None:
    global _infer_thread
    # Clear any leftover items from a previous session
    while not _infer_q.empty():
        try: _infer_q.get_nowait()
        except queue.Empty: break
    while not _result_q.empty():
        try: _result_q.get_nowait()
        except queue.Empty: break
    _infer_thread = threading.Thread(target=_inference_worker, args=(s,),
                                     daemon=True, name="InferenceWorker")
    _infer_thread.start()


def _stop_inference_thread() -> None:
    try:
        _infer_q.put_nowait(None)   # sentinel
    except queue.Full:
        pass


# =========================================================================
# MAIN
# =========================================================================
def main():
    parser = argparse.ArgumentParser(description="Human Tracking — Pose + Analytics")
    parser.add_argument("--cam",  type=int, default=0)
    parser.add_argument("--rtsp", type=str, default="",
                        help="RTSP URL for phone camera (e.g. DroidCam)")
    args = parser.parse_args()

    s = SystemState()
    s.gesture_proc = GestureProcessor()

    # Camera
    source = args.rtsp if args.rtsp else args.cam
    logger.info(f"Opening camera: {source}")
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.critical(f"Cannot open camera: {source}")
        sys.exit(1)

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_NAME, 1280, 720)
    cv2.moveWindow(WINDOW_NAME, 100, 50)

    # Court calibration — loads any existing calibration from config.json
    s.court_cal = CourtCalibrator(config_path="config.json")
    cv2.setMouseCallback(WINDOW_NAME, _window_mouse_handler, s)

    # Warmup
    logger.info("Warming up camera...")
    for _ in range(60):
        ret, _ = cap.read()
        if ret:
            break
        time.sleep(0.03)
    for _ in range(5):
        cap.read()
    time.sleep(0.1)
    logger.info("Camera ready. [i] boot AI → [Space] Warmup | [Tab] Analytics")

    last_time = time.time()
    fps        = 0.0
    last_good_frame  = None
    current_dict     = None     # most recent inference result
    current_raw_frame = None    # the frame that produced current_dict
    frame_seq        = 0        # monotonic counter to match frames to results
    model_idx  = 0
    smooth_idx = 1

    try:
        while True:
            if s.paused:
                if last_good_frame is not None:
                    paused_frame = draw_paused_overlay(last_good_frame.copy())
                    cv2.imshow(WINDOW_NAME, paused_frame)
                key = cv2.waitKey(50) & 0xFF
                if key in (27, ord('q')): break
                if key == ord('P'):
                    s.paused = False
                    s.show_toast("Resumed")
                continue

            ret, frame = cap.read()
            if not ret or frame is None:
                if last_good_frame is not None:
                    cv2.imshow(WINDOW_NAME, last_good_frame)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord('q')): break
                time.sleep(0.005)
                continue

            frame = cv2.flip(frame, 1)
            last_good_frame = frame
            s.global_frame_counter += 1

            # Adaptive gesture processing
            ival = 8 if s.inference_time_ema > 0.10 else 4 if s.inference_time_ema > 0.05 else 2
            if s.global_frame_counter % ival == 0:
                gr = s.gesture_proc.process(frame, s.warmup_active,
                                            s.analytics_mode, s.detector)
                if gr['toggle_warmup']:
                    s.warmup_active = not s.warmup_active
                    s.ui_visible    = s.warmup_active
                    s.show_toast(f"Warmup {'ON' if s.warmup_active else 'OFF'}")
                if gr['set_mode'] and s.detector and s.inference_active:
                    s.detector.set_mode(gr['set_mode'])
                    s.detector.latest_angles = {}
                    names = {'1':'Squat','2':'Pushup','3':'Curl'}
                    s.show_toast(f"Mode: {names.get(gr['set_mode'], gr['set_mode'])}")
                if gr['blocked_msg']:
                    s.show_toast(gr['blocked_msg'])

            now = time.time()
            fps = 0.9*fps + 0.1/(now - last_time) if (now - last_time) > 0 else fps
            last_time = now

            frame_seq += 1

            # ── Dispatch frame to inference thread ─────────────────────────
            if s.ai_loaded and s.inference_active and s.detector:
                try:
                    _infer_q.put_nowait((frame, frame_seq))
                except queue.Full:
                    pass   # worker still busy — skip this frame (display continues)

            # ── Poll for latest inference result (non-blocking) ────────────
            try:
                _res, _raw, _seq = _result_q.get_nowait()
                current_dict      = _res
                current_raw_frame = _raw
                if current_dict:
                    update_persistent_reps(s, current_dict)
            except queue.Empty:
                pass   # reuse previous current_dict — HUD draws at full fps

            display_frame = frame.copy()
            if current_dict and s.detector and current_raw_frame is not None:
                display_frame = s.detector.draw_landmarks(current_raw_frame, current_dict)

            if s.analytics_mode:
                display_frame = run_analytics_pipeline(display_frame, current_dict, s, frame)

            # Court overlays
            # Calibration-in-progress overlay always shows (you need to see
            # it to click corners regardless of mode). The finished boundary
            # + bird's-eye view are analytics-only — this was previously
            # drawing on top of Warmup mode too, which is confusing since
            # it's not a Warmup feature.
            if s.court_cal:
                if s.calibrating:
                    display_frame = s.court_cal.draw_calibration_overlay(display_frame)
                elif s.court_cal.calibrated and s.analytics_mode:
                    display_frame = s.court_cal.draw_court_boundary(display_frame)
                    if current_dict:
                        tracks = s.mp_tracker.tracks if s.mp_tracker else []
                        positions = [t.center for t in tracks]
                        if positions:
                            ih, iw = display_frame.shape[:2]
                            display_frame = s.court_cal.draw_birdeye_panel(
                                display_frame, positions,
                                iw - 195, ih - 400)

            display_frame = draw_hud(display_frame, fps, current_dict, s.hud_dict())

            if s.recording and s.video_writer:
                # REC indicator already drawn by hud.py
                s.video_writer.write(display_frame)

            cv2.imshow(WINDOW_NAME, display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 255: continue

            # ── Quit ──────────────────────────────────────────────────────
            if key in (27, ord('q')): break

            # ── MODE B KEYS ───────────────────────────────────────────────
            if key == 9:   # Tab
                if s.warmup_active:
                    s.show_toast("Turn off Warmup (Space) first")
                else:
                    s.analytics_mode = not s.analytics_mode
                    if s.analytics_mode:
                        s.analytics_used_this_run = True
                        init_analytics(s)
                        s.history_lines      = format_history_lines(load_history(".")) if _ANALYTICS_AVAILABLE else []
                        s.history_show_until = time.time() + 6.0
                        s.show_toast("Analytics ON — Shift+1..0 for features")
                    else:
                        if s.show_dashboard and s.dashboard:
                            s.dashboard.close()
                        s.show_dashboard = False
                        s.show_toast("Analytics OFF")
                continue

            sk = SHIFT_KEY_MAP.get(key)

            if sk and s.analytics_mode:
                if   sk=="S1": _toggle_flag(s, 'show_heatmap',   "Heatmap")
                elif sk=="S2": _toggle_flag(s, 'show_recovery',  "Recovery Timer")
                elif sk=="S3": _toggle_flag(s, 'show_stamina',   "Intensity")
                elif sk=="S4": _toggle_flag(s, 'show_stance',    "Stance Checker")
                elif sk=="S5": _toggle_flag(s, 'show_stats',     "Session Stats")
                elif sk=="S6":
                    s.show_dashboard = not s.show_dashboard
                    if s.show_dashboard and s.dashboard: s.dashboard.open()
                    elif s.dashboard: s.dashboard.close()
                    s.show_toast(f"Dashboard: {'ON' if s.show_dashboard else 'OFF'}")
                elif sk=="S7": _toggle_flag(s, 'show_footwork',  "Footwork")
                elif sk=="S8": _toggle_flag(s, 'show_stroke_fb', "Stroke Feedback")
                elif sk=="S9": _toggle_flag(s, 'show_rally',     "Rally Counter")
                elif sk=="S0" and _ANALYTICS_AVAILABLE:
                    snap  = s.heatmap.get_overlay(frame.copy()) if s.heatmap else frame.copy()
                    sh    = list(s.body_analytics.speed_history) if s.body_analytics else []
                    ah    = list(s.body_analytics.accel_history) if s.body_analytics else []
                    ac    = {
                        "Strokes":     s.stroke_detector.stroke_count    if s.stroke_detector else 0,
                        "Lunges":      s.lunge_detector.lunge_count      if s.lunge_detector  else 0,
                        "Split-steps": s.split_detector.split_step_count if s.split_detector  else 0,
                        "Jumps":       s.jump_detector.jump_count        if s.jump_detector   else 0,
                    }
                    fname = export_html_report(sh, ah, snap, ac,
                                               s.body_analytics.time_idle   if s.body_analytics else 0,
                                               s.body_analytics.time_moving if s.body_analytics else 0,
                                               "session_report.html")
                    if s.exporter:
                        s.exporter.save_csv()
                        s.exporter.save_json(summary=_build_session_summary(s))
                    s.show_toast(f"Saved: {fname}", category="success")
                continue

            if key == ord('r') and s.analytics_mode:
                s.reset_analytics()
                s.show_toast("Analytics Reset")
                continue

            # ── Exercise mode keys ────────────────────────────────────────
            if key in (ord('0'),ord('1'),ord('2'),ord('3')):
                if s.warmup_active and s.detector and s.inference_active:
                    s.detector.set_mode(chr(key))
                    s.detector.latest_angles = {}
                    en = {'0':'None','1':'Squat','2':'Pushup','3':'Curl'}
                    s.show_toast(f"Mode: {en[chr(key)]}")
                continue

            # ── AI boot ───────────────────────────────────────────────────
            if key == ord('i'):
                if s.warmup_active:
                    s.warmup_active = False
                    s.ui_visible    = False
                    if s.detector: s.detector.latest_angles = {}
                if not s.ai_loaded:
                    logger.info("Booting PoseDetector...")
                    s.detector        = PoseDetector()
                    s.ai_loaded       = True
                    s.inference_active = True
                    _start_inference_thread(s)
                    s.show_toast("AI Online — [Space] Warmup | [Tab] Analytics")
                else:
                    s.inference_active = not s.inference_active
                    if not s.inference_active:
                        _stop_inference_thread()
                        s.warmup_active = False
                        s.ui_visible    = False
                        s.persistent_reps = {k:0 for k in s.persistent_reps}
                        s.last_detector_rep_count = 0
                        s.last_detector_mode      = None
                        if s.detector:
                            s.detector.set_mode('0')
                            s.detector.latest_angles = {}
                        s.show_toast("AI Offline. Session cleared.")
                    else:
                        _start_inference_thread(s)
                        s.show_toast("AI Rebooted — [Space] Warmup | [Tab] Analytics")
                continue

            # ── Help — always available, even before AI is booted ───────────
            if key == ord('h'):
                s.show_help = not s.show_help
                s.help_scroll = 0
                continue

            if key == ord('H'):
                s.help_size = (s.help_size + 1) % 3
                continue

            if s.show_help and key == ord('['):
                s.help_scroll = max(0, s.help_scroll - 3)
                continue

            if s.show_help and key == ord(']'):
                s.help_scroll = s.help_scroll + 3
                continue

            if not s.ai_loaded or not s.inference_active:
                continue

            # ── MODE A KEYS ───────────────────────────────────────────────
            if key in (ord(' '), ord('s')):
                if s.analytics_mode:
                    s.show_toast("Turn off Analytics (Tab) first")
                else:
                    s.warmup_active = not s.warmup_active
                    s.ui_visible    = s.warmup_active
                    if s.detector: s.detector.latest_angles = {}
                    s.show_toast(f"Warmup: {'ON' if s.warmup_active else 'OFF'}")

            elif key == ord('p'):
                s.detector.show_roi = not s.detector.show_roi
                s.show_toast(f"ROI: {'ON' if s.detector.show_roi else 'OFF'}")

            elif key == ord('P'):
                s.paused = True
                s.show_toast("Paused — press P to resume")

            elif key == ord('b'):
                if s.ball_tracker:
                    s.show_ball = not s.show_ball
                    s.show_toast(f"Ball tracking: {'ON' if s.show_ball else 'OFF'}")

            elif key == ord('n'):
                if s.racket_detector and s.racket_detector.available:
                    s.show_racket = not s.show_racket
                    s.show_toast(f"Racket detection: {'ON' if s.show_racket else 'OFF'}")
                elif s.racket_detector:
                    s.show_toast("Racket model not found — see models/ setup", category="warning")

            elif key == ord('a'):
                if s.audio_coach:
                    s.audio_coach.enabled = not s.audio_coach.enabled
                    s.show_toast(f"Audio: {'ON' if s.audio_coach.enabled else 'OFF'}")

            elif key == ord('v'):
                if not s.recording:
                    fh2, fw2 = frame.shape[:2]
                    start_recording(s, fw2, fh2, fps)
                    s.show_toast(f"Recording: {s.recording_filename}")
                else:
                    stop_recording(s)
                    s.show_toast(f"Saved: {s.recording_filename}", category="success")

            elif key in (ord('+'), ord('=')):
                s.detector.visibility_threshold = min(1.0, s.detector.visibility_threshold+0.05)
                s.show_toast(f"Confidence: {s.detector.visibility_threshold:.2f}")

            elif key in (ord('-'), ord('_')):
                s.detector.visibility_threshold = max(0.0, s.detector.visibility_threshold-0.05)
                s.show_toast(f"Confidence: {s.detector.visibility_threshold:.2f}")

            elif key in (ord('g'), ord('G')):
                s.presence_grace_max += 1
                s.show_toast(f"Grace: {s.presence_grace_max} frames")

            elif key == ord('j'):
                s.presence_grace_max = max(0, s.presence_grace_max-1)
                s.show_toast(f"Grace: {s.presence_grace_max} frames")

            elif key == ord('c'):
                s.compact_hud = not s.compact_hud
                s.show_toast(f"Layout: {'COMPACT' if s.compact_hud else 'FULL'}")

            elif key == ord('e'):
                if s.detector:
                    f2 = s.detector.export_session()
                    s.show_toast(f"Saved: {f2}", category="success")

            elif key == ord('m'):
                model_idx = (model_idx+1) % len(MODEL_PRESETS)
                s.show_toast(f"Engine: {MODEL_PRESETS[model_idx].upper()}")

            elif key == ord('k'):
                smooth_idx = (smooth_idx+1) % len(SMOOTH_KEYS)
                s.detector.current_preset = SMOOTH_KEYS[smooth_idx]
                s.show_toast(f"Smoothing: {s.detector.current_preset}")

            elif key == ord('l'):
                s.detector._previous_landmarks = None
                s.show_toast("Tracking lock cleared")

            elif key == ord('C'):
                if s.court_cal:
                    if s.calibrating:
                        # Cancel this re-click attempt — restores whatever
                        # calibration existed before (if any). Use [x] to
                        # fully delete calibration instead.
                        s.court_cal._clicks.clear()
                        s.court_cal.calibrating = False
                        s.calibrating = False
                        s.show_toast("Calibration cancelled")
                    else:
                        s.court_cal._clicks.clear()
                        s.court_cal.calibrating = True
                        s.calibrating = True
                        s.show_toast("Click 4 corners: TL → TR → BR → BL")

            elif key == ord('x'):
                if s.court_cal and (s.court_cal.calibrated or s.calibrating):
                    s.court_cal.clear()
                    s.calibrating = False
                    s.show_toast("Court calibration cleared", category="warning")
                elif s.court_cal:
                    s.show_toast("No calibration to clear")

    except KeyboardInterrupt:
        logger.info("Ctrl+C — shutting down.")
    except Exception as exc:
        logger.error(f"Fatal error: {exc}", exc_info=True)
    finally:
        logger.info("Releasing resources...")
        _stop_inference_thread()
        if s.recording:        stop_recording(s)
        if s.audio_coach:      s.audio_coach.stop()
        if s.dashboard:
            try: s.dashboard.close()
            except Exception: pass

        # Session end summary — only if analytics was actually used this run
        if s.analytics_used_this_run and s.body_analytics and s.exporter:
            try:
                saved_csv = ""
                try:
                    saved_csv = s.exporter.save_csv()
                    s.exporter.save_json(summary=_build_session_summary(s))
                except Exception:
                    pass

                total_t = s.body_analytics.time_idle + s.body_analytics.time_moving
                active_pct = (100.0 * s.body_analytics.time_moving / total_t
                             if total_t > 0 else 0.0)
                duration_sec = time.time() - s.analytics_session_start
                m, sec = divmod(int(duration_sec), 60)

                stats = {
                    'strokes':       s.stroke_detector.stroke_count    if s.stroke_detector else 0,
                    'splits':        s.split_detector.split_step_count if s.split_detector  else 0,
                    'lunges':        s.lunge_detector.lunge_count      if s.lunge_detector  else 0,
                    'jumps':         s.jump_detector.jump_count        if s.jump_detector   else 0,
                    'active_pct':    active_pct,
                    'avg_speed':     s.body_analytics.avg_speed,
                    'longest_rally': s.rally.longest_rally if s.rally else 0,
                    'duration':      f"{m}m{sec:02d}s",
                    'saved_file':    saved_csv,
                }

                end_frame = last_good_frame.copy() if last_good_frame is not None \
                           else np.zeros((720,1280,3), dtype=np.uint8)
                end_frame = draw_session_end_overlay(end_frame, stats)
                cv2.imshow(WINDOW_NAME, end_frame)
                cv2.waitKey(4000)   # hold for 4 seconds before closing
            except Exception as e:
                logger.debug(f"Session summary display error: {e}")

        cap.release()
        cv2.destroyAllWindows()
        logger.info("Shutdown complete.")


if __name__ == "__main__":
    main()