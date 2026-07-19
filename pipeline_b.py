# pipeline_b.py — analytics pipeline, called from main.py once per frame
# when analytics mode is on. Renders all the Shift+1..0 panels.
import cv2, time, numpy as np, logging
from typing import Optional, Any
from hud import (draw_panel, _text, _rect, _line, draw_stat_pair,
                 draw_big_number, draw_qual_badge,
                 ACCENT, ORANGE, GREEN, YELLOW, WHITE, GRAY_LT, GRAY_MD,
                 GRAY_DK, RED, DIM_CYAN, BG, HDR_BG,
                 F_BODY, F_TITLE, F_BIG,
                 TS_BODY, TS_SMALL, TS_TITLE, TS_BIG,
                 MARGIN, PANEL_GAP, PAD, PANEL_W, ACCENT_BAR)
from analytics_module import (calculate_body_angles,
                               distance_between_players, BodyAnalytics)
from action_module import (check_ready_stance, classify_movement,
                           SPEED_READY)
from pose_quality import PoseQuality, QUALITY_THRESHOLD

logger = logging.getLogger(__name__)

STROKE_FB_HOLD = 1.5


def run_analytics_pipeline(display_frame, current_dict, s, raw_frame=None):
    if not current_dict:
        return display_frame

    clean_frame = raw_frame if raw_frame is not None else display_frame

    result = current_dict.get("result")
    if not result or not getattr(result, "pose_landmarks", None):
        return display_frame

    fw = current_dict.get("frame_w", display_frame.shape[1])
    fh = current_dict.get("frame_h", display_frame.shape[0])
    now = time.time()

    if s.mp_tracker is None:
        return display_frame

    tracks  = s.mp_tracker.update(result.pose_landmarks, fw, fh)
    if not tracks:
        return display_frame

    primary  = min(tracks, key=lambda t: t.id)
    lms      = primary.landmarks
    torso_px = primary.torso_px
    speed    = primary.speed

    def vis(i):
        return getattr(lms[i], 'visibility', 1.0)

    hip_idx = [i for i in (23, 24) if vis(i) > 0.3]
    cx = (sum(lms[i].x for i in hip_idx)/len(hip_idx)*fw) if hip_idx else primary.center[0]
    cy = (sum(lms[i].y for i in hip_idx)/len(hip_idx)*fh) if hip_idx else primary.center[1]

    angles     = calculate_body_angles(lms, fw, fh)
    left_knee  = angles.get("left_leg")
    right_knee = angles.get("right_leg")

    # ── Pose quality gate ──────────────────────────────────────────────────
    # Lazily create the quality tracker so it survives resets cleanly
    if s.pose_quality is None:
        s.pose_quality = PoseQuality()
    quality = s.pose_quality.update(lms)
    quality_ok = not s.pose_quality.suppress_detectors

    # Note: torso_cal.baseline_px is computed but tracker_module already
    # maintains its own EMA-smoothed torso_px per track, which is what
    # speed is actually calculated from. The calibration gate used to
    # block analytics for 5 seconds without that baseline ever being
    # consumed anywhere, so it's removed here rather than left blocking
    # for no effect.

    # ── Run all detectors — suppressed when tracking quality is too low ────
    if quality_ok:
        if s.body_analytics:  s.body_analytics.update(speed)
        if s.stamina:         s.stamina.update(speed, now)
        if s.jump_detector:   s.jump_detector.update(lms, torso_px, fh)

        split_now = s.split_detector.update(lms, torso_px, fw, fh) if s.split_detector else False
        lunge_now = s.lunge_detector.update(left_knee, right_knee) if s.lunge_detector else False

        stroke_now = False
        if s.stroke_detector:
            try:
                stroke_now = s.stroke_detector.update(lms, torso_px, fw, fh)
            except Exception as e:
                logger.debug(f"StrokeDetector: {e}")
    else:
        # Quality too low — no new detector state, no false events.
        # Still update BodyAnalytics with a zero/held reading so time_idle
        # doesn't quietly stop accumulating during a poor-tracking stretch.
        split_now = False
        lunge_now = False
        stroke_now = False
    ball_near = False
    if quality_ok and s.ball_tracker:
        s.ball_tracker.update(clean_frame)
        if s.ball_tracker.detected:
            for wi in (15, 16):
                if vis(wi) > 0.3:
                    wx, wy = lms[wi].x*fw, lms[wi].y*fh
                    if s.ball_tracker.near_wrist(wx, wy, torso_px):
                        ball_near = True; break

    racket_conf = 0.0
    racket_seen_this_frame = False
    if quality_ok and s.racket_detector and s.racket_detector.available:
        s.racket_detector.detect(clean_frame)
        racket_seen_this_frame = len(s.racket_detector.detections) > 0
        for wi in (15, 16):
            if vis(wi) > 0.3:
                wx, wy = lms[wi].x*fw, lms[wi].y*fh
                rc = s.racket_detector.racket_near_wrist(wx, wy, torso_px)
                racket_conf = max(racket_conf, rc)

    # Ball/racket only ever ADD confidence, never veto a wrist-speed stroke
    # that already fired. Requiring a positive detection on the exact frame
    # a fast swing happens (often blurred) would silently drop real strokes
    # whenever either module is merely loaded but didn't see anything that
    # frame — so the gate only applies when one of them actually detected
    # something right now.
    if stroke_now and (s.ball_tracker and s.ball_tracker.detected or racket_seen_this_frame):
        stroke_now = ball_near or (racket_conf > 0.35)

    if s.recovery:  s.recovery.update(speed, now)
    if s.rally:     s.rally.update(now)

    if stroke_now:
        if s.recovery:  s.recovery.notify_stroke(now)
        if s.rally:     s.rally.notify_stroke(now)
        wsp   = s.stroke_detector.last_wrist_speed if s.stroke_detector else 0
        side  = s.stroke_detector.last_stroke_side if s.stroke_detector else "--"
        slbl  = "HIGH" if wsp > 7 else "MED" if wsp > 4.5 else "LOW"
        sp    = f"  {side}" if side != "--" else ""
        s.stroke_fb_text  = f"STROKE #{s.stroke_detector.stroke_count}{sp}  ·  WRIST {slbl}"
        s.stroke_fb_until = now + STROKE_FB_HOLD
        if s.audio_coach:
            try:
                from audio_module import cue_stroke
                cue_stroke(s.audio_coach, side)
            except Exception: pass

    if lunge_now and s.audio_coach:
        try:
            from audio_module import cue_lunge
            cue_lunge(s.audio_coach, getattr(s.lunge_detector,'last_lunge_dir','--'))
        except Exception: pass

    if split_now and s.audio_coach:
        try:
            from audio_module import cue_split_step
            cue_split_step(s.audio_coach)
        except Exception: pass

    if s.rally and s.rally.rally_just_ended and s.audio_coach:
        try:
            from audio_module import cue_rally_end
            cue_rally_end(s.audio_coach, s.rally.current_rally)
        except Exception: pass

    # Stance audio cue
    if s.audio_coach and s.audio_coach.enabled and speed < SPEED_READY:
        try:
            from audio_module import cue_stance
            sc = check_ready_stance(angles, speed)
            if sc: cue_stance(s.audio_coach, sc)
        except Exception: pass

    # Zone tracking
    zf = cy / fh
    if zf > 0.65:   s.zone_counts["baseline"] += 1
    elif zf > 0.35: s.zone_counts["mid"]      += 1
    else:           s.zone_counts["net"]       += 1

    # Ball / racket overlays (movement arrow removed — velocity computed
    # from raw frame-to-frame hip-center deltas was dominated by pose
    # jitter far more often than real movement, making it visually random)
    if s.ball_tracker and s.show_ball:
        display_frame = s.ball_tracker.draw(display_frame)
    if s.racket_detector and s.show_racket:
        display_frame = s.racket_detector.draw(display_frame)

    # Pose quality indicator — only shown when degraded/poor, so it doesn't
    # clutter the screen during normal tracking
    if s.pose_quality and s.pose_quality.status != "good":
        qcol = s.pose_quality.status_color
        qlbl = "TRACKING DEGRADED" if s.pose_quality.status == "degraded" else "TRACKING POOR — events paused"
        draw_qual_badge(display_frame, MARGIN+ACCENT_BAR+3, 24, qlbl, qcol)

    if stroke_now:
        for wi in (15, 16):
            if vis(wi) > 0.3:
                wx, wy = int(lms[wi].x*fw), int(lms[wi].y*fh)
                cv2.circle(display_frame, (wx,wy), 20, ACCENT, 3, cv2.LINE_AA)

    # ── Dynamic panel stacking ─────────────────────────────────────────────
    left_y  = 50    # below broadcast bug
    right_y = 50
    top_y   = 50

    # ── Shift+1: Court heatmap + zones ────────────────────────────────────
    if s.show_heatmap and s.heatmap:
        s.heatmap.add_point(cx, cy, fw, fh)
        s.heatmap.decay_step()
        display_frame = s.heatmap.get_overlay(display_frame)

        # Court-normalised accumulation, when calibrated — comparable
        # session to session regardless of camera position
        if s.court_cal and s.court_cal.calibrated:
            mx, my = s.court_cal.px_to_metres(cx, cy)
            s.heatmap.add_point_court(mx, my, s.court_cal.court_w_m,
                                      s.court_cal.court_l_m)
            s.heatmap.decay_step_court()
            court_canvas = s.heatmap.get_court_overlay(s.court_cal.court_type)
            ch, cw2 = court_canvas.shape[:2]
            py2 = fh - ch - MARGIN
            px2 = MARGIN
            if py2 > 60:  # only paste if there's room without overlapping panels
                display_frame[py2:py2+ch, px2:px2+cw2] = court_canvas
                cv2.rectangle(display_frame, (px2,py2), (px2+cw2,py2+ch),
                              GRAY_DK, 1)

        total_z = sum(s.zone_counts.values()) or 1
        yz = 6
        zone_defs = [
            ("NET",      s.zone_counts["net"],      int(fh*0.35), ORANGE),
            ("MID",      s.zone_counts["mid"],      int(fh*0.65), ACCENT),
            ("BASELINE", s.zone_counts["baseline"], fh-20,        GREEN),
        ]
        for zlbl, zcount, zdiv_y, zcol in zone_defs:
            _line(display_frame, 0, zdiv_y, fw, zdiv_y, GRAY_DK)
            pct = int(100*zcount/total_z)
            draw_qual_badge(display_frame, MARGIN, zdiv_y-16, f"{zlbl} {pct}%", zcol)

    # ── Shift+2: Recovery timer ────────────────────────────────────────────
    if s.show_recovery and s.recovery:
        ph = 72
        draw_panel(display_frame, MARGIN, left_y, PANEL_W, ph, "recovery", GREEN)
        r = s.recovery
        if r.last_recovery_sec is not None:
            qcol = r.quality_color
            _text(display_frame, f"{r.last_recovery_sec:.2f}s",
                  MARGIN+ACCENT_BAR+PAD, left_y+52,
                  F_BIG, 0.80, qcol, 2)
            draw_qual_badge(display_frame,
                            MARGIN+ACCENT_BAR+PAD+75, left_y+52,
                            r.quality.upper(), qcol)
            if r.avg_recovery_sec:
                _text(display_frame, f"AVG  {r.avg_recovery_sec:.2f}s",
                      MARGIN+ACCENT_BAR+PAD, left_y+68,
                      F_BODY, TS_SMALL, GRAY_MD)
        else:
            _text(display_frame, "waiting for first stroke...",
                  MARGIN+ACCENT_BAR+PAD, left_y+46, F_BODY, TS_SMALL, GRAY_DK)
        left_y += ph + PANEL_GAP

    # ── Shift+3: Intensity sparkline ──────────────────────────────────────
    if s.show_stamina and s.stamina:
        ph = 76
        draw_panel(display_frame, MARGIN, left_y, PANEL_W, ph, "intensity", YELLOW)
        tcol  = s.stamina.trend_color
        trend = s.stamina.trend.upper()
        draw_qual_badge(display_frame, MARGIN+ACCENT_BAR+PAD, left_y+38, trend, tcol)
        _text(display_frame, f"{s.stamina.current_avg:.1f} BL/s",
              MARGIN+ACCENT_BAR+PAD+70, left_y+38, F_BODY, TS_BODY, GRAY_LT)
        # Sparkline
        bars    = list(s.stamina.sparkline)
        max_val = max(bars) if bars else 1.0
        max_val = max(max_val, 0.1)
        bax, bay = MARGIN+ACCENT_BAR+PAD, left_y+52
        bh, bw2 = 14, max(2, (PANEL_W-ACCENT_BAR-PAD*2)//max(len(bars),1))
        for i, val in enumerate(bars):
            filled = int(bh*val/max_val)
            bx2    = bax + i*bw2
            cv2.rectangle(display_frame,(bx2,bay),(bx2+bw2-1,bay+bh),(50,50,50),-1)
            if filled > 0:
                cv2.rectangle(display_frame,(bx2,bay+bh-filled),(bx2+bw2-1,bay+bh),tcol,-1)
        left_y += ph + PANEL_GAP

    # ── Shift+4: Ready stance ─────────────────────────────────────────────
    if s.show_stance:
        cues = check_ready_stance(angles, speed)
        ph   = 72
        draw_panel(display_frame, MARGIN, left_y, PANEL_W, ph, "ready stance", ORANGE)
        if cues is None:
            _text(display_frame, "moving — check when still",
                  MARGIN+ACCENT_BAR+PAD, left_y+46, F_BODY, TS_SMALL, GRAY_DK)
        elif not cues:
            _text(display_frame, "stand fully in frame",
                  MARGIN+ACCENT_BAR+PAD, left_y+46, F_BODY, TS_SMALL, GRAY_DK)
        else:
            qc = {"good": GREEN, "ok": YELLOW, "poor": RED}
            yo = left_y + 38
            for _, (qual, cue) in cues.items():
                _text(display_frame, cue, MARGIN+ACCENT_BAR+PAD, yo,
                      F_BODY, TS_SMALL, qc.get(qual, GRAY_MD))
                yo += 17
        left_y += ph + PANEL_GAP

    # ── Shift+5: Session stats (right column) ────────────────────────────
    if s.show_stats:
        ba = s.body_analytics
        active_pct = 0
        if ba:
            tot = ba.time_idle + ba.time_moving
            if tot > 0: active_pct = int(100*ba.time_moving/tot)
        ph = 154
        draw_panel(display_frame, fw-PANEL_W-MARGIN, right_y, PANEL_W, ph,
                   "session stats", ACCENT)
        rx = fw-PANEL_W-MARGIN+ACCENT_BAR+PAD
        col_items = [
            (f"{s.stroke_detector.stroke_count if s.stroke_detector else 0}",
             "strokes",    ACCENT),
            (f"{s.split_detector.split_step_count if s.split_detector else 0}",
             "split steps",GREEN),
            (f"{s.lunge_detector.lunge_count if s.lunge_detector else 0}",
             "lunges",     ORANGE),
            (f"{s.jump_detector.jump_count if s.jump_detector else 0}",
             "jumps",      YELLOW),
        ]
        col_w2 = (PANEL_W-ACCENT_BAR-PAD*2)//2
        for i, (val, lbl, col) in enumerate(col_items):
            cx2 = rx + (i%2)*col_w2
            cy2 = right_y + 40 + (i//2)*48
            _text(display_frame, val, cx2, cy2,     F_BIG,  0.70, col, 2)
            _text(display_frame, lbl, cx2, cy2+16,  F_BODY, TS_SMALL, GRAY_MD)
        _line(display_frame, fw-PANEL_W-MARGIN+ACCENT_BAR, right_y+106,
              fw-MARGIN-2, right_y+106, GRAY_DK)
        _text(display_frame, f"ACTIVE  {active_pct}%",
              rx, right_y+122, F_BODY, TS_BODY, GREEN)
        _text(display_frame, f"AVG SPD  {ba.avg_speed:.1f} BL/s" if ba else "AVG SPD --",
              rx, right_y+140, F_BODY, TS_BODY, ACCENT)
        right_y += ph + PANEL_GAP

    # ── Shift+6: Dashboard ────────────────────────────────────────────────
    if s.show_dashboard and s.dashboard:
        ba = s.body_analytics
        s.dashboard.feed(
            ba.last_speed if ba else 0,
            ba.last_accel if ba else 0,
            classify_movement(speed),
            s.stroke_detector.stroke_count    if s.stroke_detector else 0,
            s.lunge_detector.lunge_count      if s.lunge_detector  else 0,
            s.split_detector.split_step_count if s.split_detector  else 0,
            s.jump_detector.jump_count        if s.jump_detector   else 0,
            ba.time_idle   if ba else 0,
            ba.time_moving if ba else 0,
        )
        s.dashboard.render()

    # ── Shift+7: Footwork rates ───────────────────────────────────────────
    if s.show_footwork:
        sm = max((now - s.analytics_session_start)/60.0, 1/60)
        sp_r = (s.split_detector.split_step_count if s.split_detector else 0)/sm
        ln_r = (s.lunge_detector.lunge_count      if s.lunge_detector  else 0)/sm
        jp_r = (s.jump_detector.jump_count         if s.jump_detector   else 0)/sm
        phase = s.jump_detector.phase if s.jump_detector else "grounded"
        ph    = 92
        draw_panel(display_frame, MARGIN, left_y, PANEL_W, ph, "footwork", GREEN)
        rows = [
            (f"{s.split_detector.split_step_count if s.split_detector else 0}",
             f"split steps · {sp_r:.1f}/min", GREEN),
            (f"{s.lunge_detector.lunge_count if s.lunge_detector else 0}",
             f"lunges · {ln_r:.1f}/min", ORANGE),
            (f"{s.jump_detector.jump_count if s.jump_detector else 0}",
             f"jumps · {jp_r:.1f}/min  [{phase}]",
             RED if phase in ("rising","peak") else GRAY_MD),
        ]
        yo = left_y + 36
        for val, lbl, col in rows:
            _text(display_frame, val, MARGIN+ACCENT_BAR+PAD, yo, F_TITLE, 0.50, col)
            _text(display_frame, lbl, MARGIN+ACCENT_BAR+PAD+28, yo,
                  F_BODY, TS_SMALL, GRAY_MD)
            yo += 20
        left_y += ph + PANEL_GAP

    # ── Shift+8: Stroke feedback (top-centre) ─────────────────────────────
    if s.show_stroke_fb:
        if now < s.stroke_fb_until and s.stroke_fb_text:
            tw, _ = cv2.getTextSize(s.stroke_fb_text, F_TITLE, 0.60, 2)
            lx = (fw - tw[0])//2 - PAD
            pw2 = tw[0] + PAD*2
            draw_panel(display_frame, lx, top_y, pw2, 42, "stroke", ACCENT)
            _text(display_frame, s.stroke_fb_text, lx+ACCENT_BAR+PAD, top_y+30,
                  F_TITLE, 0.60, WHITE, 2)
            top_y += 42 + PANEL_GAP
        else:
            _text(display_frame, "awaiting stroke...",
                  fw//2-80, top_y+16, F_BODY, TS_SMALL, GRAY_DK)
            top_y += 24 + PANEL_GAP

    # ── Shift+9: Rally counter (top-centre, below stroke) ─────────────────
    if s.show_rally and s.rally:
        r   = s.rally
        pw2 = 280
        lx  = (fw-pw2)//2
        if r.rally_just_ended:
            draw_panel(display_frame, lx, top_y, pw2, 50, "rally", YELLOW)
            _text(display_frame, f"RALLY ENDED  ·  {r.current_rally} STROKES",
                  lx+ACCENT_BAR+PAD, top_y+34, F_TITLE, 0.52, YELLOW, 1)
        elif r.current_rally > 0:
            draw_panel(display_frame, lx, top_y, pw2, 50, "rally", GREEN)
            _text(display_frame, f"RALLY  {r.current_rally}",
                  lx+ACCENT_BAR+PAD, top_y+36, F_BIG, 0.80, GREEN, 2)
            _text(display_frame, f"BEST {r.longest_rally}",
                  lx+ACCENT_BAR+PAD+130, top_y+36, F_BODY, TS_SMALL, GRAY_MD)
        else:
            draw_panel(display_frame, lx, top_y, pw2, 36, "rally", GRAY_DK)
            _text(display_frame, f"RALLY  —  BEST: {r.longest_rally}",
                  lx+ACCENT_BAR+PAD, top_y+24, F_BODY, TS_BODY, GRAY_MD)
        top_y += 50 + PANEL_GAP

    # Record to exporter
    if s.exporter:
        try:
            s.exporter.record(s.global_frame_counter, primary.id, primary.center,
                              speed, s.body_analytics.last_accel if s.body_analytics else 0,
                              classify_movement(speed), angles)
        except Exception: pass

    return display_frame