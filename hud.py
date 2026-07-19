# hud.py — broadcast-style HUD, ESPN/ATP scoreboard look.
# Dark panels, colored left accent bar, cyan/orange palette.
# Left column = analytics panels, right column = warmup stats,
# top-center = events, bottom ticker = analytics status.

import cv2
import time
import textwrap
import numpy as np

# ── Color palette (BGR) ────────────────────────────────────────────────────
BG        = (22,  14,  10)      # near-black navy
HDR_BG    = (38,  28,  20)      # panel header background
ACCENT    = (255, 195,  0)      # electric cyan  — primary brand
ORANGE    = (0,   130, 255)     # orange         — secondary
WHITE     = (255, 255, 255)
GRAY_LT   = (210, 210, 210)
GRAY_MD   = (150, 150, 150)
GRAY_DK   = (75,  75,  75)
GREEN     = (90,  255, 140)     # good / success
YELLOW    = (0,   215, 255)     # warning
RED       = (45,  45,  230)     # danger / live indicator
DIM_CYAN  = (160, 120,   0)     # muted accent for secondary labels

# ── Typography ─────────────────────────────────────────────────────────────
F_TITLE  = cv2.FONT_HERSHEY_DUPLEX
F_BODY   = cv2.FONT_HERSHEY_SIMPLEX
F_BIG    = cv2.FONT_HERSHEY_TRIPLEX

TS_TITLE = 0.40     # panel headers
TS_BODY  = 0.38     # body text
TS_SMALL = 0.33     # captions / dim labels
TS_BIG   = 0.80     # large stat numbers
TS_XLARGE= 1.10     # mode names / event flash

# ── Layout constants ───────────────────────────────────────────────────────
MARGIN      = 10    # outer edge margin
PANEL_GAP   = 7     # gap between stacked panels
PAD         = 7     # internal panel padding
PANEL_W     = 250   # standard panel width
ACCENT_BAR  = 3     # left accent bar width in pixels


# ── Core drawing primitives ────────────────────────────────────────────────

def _rect(img, x, y, w, h, color, alpha=0.88):
    """Filled semi-transparent rectangle."""
    ov = img.copy()
    cv2.rectangle(ov, (x, y), (x+w, y+h), color, -1)
    cv2.addWeighted(ov, alpha, img, 1-alpha, 0, img)


def _line(img, x1, y1, x2, y2, color, thickness=1):
    cv2.line(img, (x1, y1), (x2, y2), color, thickness, cv2.LINE_AA)


def _text(img, text, x, y, font, scale, color, thickness=1):
    """Text with 1px dark shadow for legibility over any background."""
    cv2.putText(img, text, (x+1, y+1), font, scale,
                (5,5,5), thickness+1, cv2.LINE_AA)
    cv2.putText(img, text, (x,   y),   font, scale,
                color, thickness, cv2.LINE_AA)


def _text_size(text, font, scale, thickness=1):
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    return w, h


def draw_panel(img, x: int, y: int, w: int, h: int,
               title: str, accent=ACCENT) -> None:
    """
    Broadcast-style panel:
      Dark background → colored left accent bar → header bar with title
      → thin separator line.
    """
    # Background
    _rect(img, x, y, w, h, BG, 0.90)
    # Outer border (very dim)
    cv2.rectangle(img, (x, y), (x+w, y+h), GRAY_DK, 1)
    # Left accent bar
    cv2.rectangle(img, (x, y), (x+ACCENT_BAR, y+h), accent, -1)
    # Header background strip
    _rect(img, x+ACCENT_BAR, y, w-ACCENT_BAR, 22, HDR_BG, 0.95)
    # Title text
    _text(img, title.upper(), x+ACCENT_BAR+PAD, y+15, F_TITLE, TS_TITLE, accent)
    # Separator line below header
    _line(img, x+ACCENT_BAR, y+22, x+w, y+22, GRAY_DK)


def draw_stat_pair(img, x, y, label, value, val_color=WHITE,
                   label_color=GRAY_MD):
    """label / value pair in compact broadcast style."""
    _text(img, label.upper(), x, y,             F_BODY, TS_SMALL, label_color)
    _text(img, str(value),    x, y+17,           F_BODY, TS_BODY,  val_color, 1)


def draw_big_number(img, x, y, number, label="", color=WHITE):
    """Large centred stat number with small label below."""
    _text(img, str(number), x, y,      F_BIG, TS_BIG, color, 2)
    if label:
        tw, _ = _text_size(str(number), F_BIG, TS_BIG, 2)
        _text(img, label.upper(), x, y+18, F_BODY, TS_SMALL, GRAY_MD)


def draw_qual_badge(img, x, y, text, color):
    """Small coloured pill badge — e.g. FAST, SLOW, GOOD."""
    tw, th = _text_size(text, F_BODY, TS_SMALL)
    cv2.rectangle(img, (x-3, y-th-1), (x+tw+3, y+2), color, -1)
    _text(img, text, x, y, F_BODY, TS_SMALL, (10,10,10), 1)


# ── Broadcast bug (always-on session info, top-left) ──────────────────────

def draw_broadcast_bug(img, fps: float, session_start: float,
                       recording: bool) -> None:
    """
    Top-left persistent session indicator — like a TV network bug.
    Shows: ● LIVE  |  FPS  |  session elapsed
    """
    h, w = img.shape[:2]
    bw, bh = 260, 32
    _rect(img, 0, 0, bw, bh, BG, 0.85)
    _line(img, 0, bh, bw, bh, GRAY_DK)
    _line(img, ACCENT_BAR, 0, ACCENT_BAR, bh, ACCENT)

    # Blinking live dot
    now = time.time()
    dot_col = RED if int(now*2)%2==0 else (80,80,80)
    cv2.circle(img, (ACCENT_BAR+10, 16), 5, dot_col, -1, cv2.LINE_AA)

    label = "REC" if recording else "LIVE"
    _text(img, label, ACCENT_BAR+20, 20, F_BODY, TS_SMALL, RED if recording else GRAY_MD)

    # FPS
    _text(img, f"{int(fps):>3} FPS", ACCENT_BAR+58, 20, F_BODY, TS_SMALL, GRAY_LT)

    # Session timer
    if session_start > 0:
        elapsed = int(now - session_start)
        m, s    = divmod(elapsed, 60)
        timer   = f"{m:02d}:{s:02d}"
        _text(img, timer, ACCENT_BAR+145, 20, F_BODY, TS_SMALL, ACCENT)

    # Separator dots
    _text(img, "▪", ACCENT_BAR+50, 20, F_BODY, TS_SMALL, GRAY_DK)
    _text(img, "▪", ACCENT_BAR+137, 20, F_BODY, TS_SMALL, GRAY_DK)


# ── Analytics ticker bar (bottom) ─────────────────────────────────────────

def draw_analytics_bar(img, s: dict) -> np.ndarray:
    if not s.get('analytics_mode'):
        return img
    h, w = img.shape[:2]
    bh   = 30
    _rect(img, 0, h-bh, w, bh, BG, 0.92)
    _line(img, 0, h-bh, w, h-bh, ACCENT)

    # Left: ANALYTICS label
    _text(img, "ANALYTICS", MARGIN, h-10, F_TITLE, TS_TITLE, ACCENT)

    # Active feature badges
    badge_map = [
        ('show_heatmap',   'HM',     ACCENT),
        ('show_recovery',  'REC',    GREEN),
        ('show_stamina',   'STAM',   YELLOW),
        ('show_stance',    'STANCE', ORANGE),
        ('show_stats',     'STATS',  GRAY_LT),
        ('show_dashboard', 'DASH',   ACCENT),
        ('show_footwork',  'FW',     GREEN),
        ('show_stroke_fb', 'STRK',  ORANGE),
        ('show_rally',     'RALLY',  YELLOW),
    ]
    bx = 120
    for flag, label, col in badge_map:
        if s.get(flag):
            tw, _ = _text_size(label, F_BODY, TS_SMALL)
            cv2.rectangle(img, (bx-2, h-25), (bx+tw+4, h-8), col, -1)
            _text(img, label, bx, h-10, F_BODY, TS_SMALL, (10,10,10))
            bx += tw + 10

    if bx == 120:
        _text(img, "Shift+1..0 to enable features", 120, h-10,
              F_BODY, TS_SMALL, GRAY_DK)
    return img


# ── Help overlay ───────────────────────────────────────────────────────────

def draw_help_overlay(img, s: dict) -> np.ndarray:
    if not s.get('show_help'):
        return img
    fs = {0: 0.32, 1: 0.38, 2: 0.46}[s.get('help_size', 1)]

    # Every control in the program, grouped by where it actually applies.
    # HDR = section heading, everything else is (text, color).
    HDR = DIM_CYAN
    lines = [
        ("CONTROLS",                              ACCENT),
        ("", None),
        ("ALWAYS AVAILABLE", HDR),
        ("h            toggle this help panel",   GRAY_LT),
        ("Shift+H      cycle help text size",      GRAY_LT),
        ("[ / ]        scroll this panel",         GRAY_LT),
        ("mouse wheel  scroll this panel",         GRAY_LT),
        ("ESC or q     quit",                      RED),
        ("", None),
        ("GETTING STARTED", HDR),
        ("i            boot AI (press again to toggle on/off)", GRAY_LT),
        ("Space or s   toggle Warmup mode",         GRAY_LT),
        ("Tab          toggle Analytics mode",      GRAY_LT),
        ("             (Warmup and Analytics are mutually exclusive)", GRAY_MD),
        ("", None),
        ("WARMUP — exercise select", HDR),
        ("0            no exercise",               GRAY_LT),
        ("1            squat",                      GRAY_LT),
        ("2            pushup",                      GRAY_LT),
        ("3            curl",                        GRAY_LT),
        ("", None),
        ("WARMUP — hand gestures", HDR),
        ("palms together, facing camera   same as [Space]", GRAY_LT),
        ("hold up 1 finger, 0.5s          switch to squat",  GRAY_LT),
        ("hold up 2 fingers, 0.5s         switch to pushup", GRAY_LT),
        ("hold up 3 fingers, 0.5s         switch to curl",   GRAY_LT),
        ("", None),
        ("WARMUP — tracking & display", HDR),
        ("p            toggle ROI box",             GRAY_LT),
        ("Shift+P      pause / resume",              GRAY_LT),
        ("+ or =       raise confidence threshold",  GRAY_LT),
        ("- or _       lower confidence threshold",  GRAY_LT),
        ("g / Shift+G  increase tracking grace",      GRAY_LT),
        ("j            decrease tracking grace",      GRAY_LT),
        ("k            cycle smoothing preset",       GRAY_LT),
        ("l            clear tracking lock",          GRAY_LT),
        ("m            cycle model preset",           GRAY_LT),
        ("c            toggle compact layout",        GRAY_LT),
        ("e            export warmup session",        GRAY_LT),
        ("v            start / stop video recording", GRAY_LT),
        ("", None),
        ("ANALYTICS — toggle & reset", HDR),
        ("Tab          toggle analytics on/off",     GRAY_LT),
        ("r            reset all analytics counters", GRAY_LT),
        ("", None),
        ("ANALYTICS — feature panels", HDR),
        ("Shift+1      court heatmap + zones",       GRAY_LT),
        ("Shift+2      recovery timer",               GRAY_LT),
        ("Shift+3      intensity / stamina trend",    GRAY_LT),
        ("Shift+4      ready-stance checker",         GRAY_LT),
        ("Shift+5      session stats panel",          GRAY_LT),
        ("Shift+6      dashboard window (separate)",  GRAY_LT),
        ("Shift+7      footwork rates",               GRAY_LT),
        ("Shift+8      stroke feedback",               GRAY_LT),
        ("Shift+9      rally counter",                 GRAY_LT),
        ("Shift+0      export report (HTML+CSV+JSON)", GRAY_LT),
        ("", None),
        ("ANALYTICS — detection add-ons", HDR),
        ("b            toggle ball tracking overlay", GRAY_LT),
        ("n            toggle racket detection",      GRAY_LT),
        ("             (needs model files — see models/)", GRAY_MD),
        ("a            toggle spoken audio coaching", GRAY_LT),
        ("", None),
        ("COURT CALIBRATION", HDR),
        ("Shift+C      start/cancel corner calibration", GRAY_LT),
        ("             click 4 corners: TL, TR, BR, BL", GRAY_MD),
        ("x            permanently clear calibration", GRAY_LT),
    ]

    lh       = 17
    line_h_px = int(lh * (fs / 0.38))   # scale line spacing with text size
    all_h    = len(lines) * line_h_px
    help_w   = 400   # wider than the standard panel — these lines run long
    px       = img.shape[1] - help_w - MARGIN
    py       = MARGIN
    max_ph   = img.shape[0] - MARGIN * 2 - 20
    ph       = min(all_h + PAD * 3, max_ph)

    draw_panel(img, px, py, help_w, ph, "controls")

    visible_lines = max(1, (ph - 34) // line_h_px)
    scroll = max(0, min(s.get('help_scroll', 0), max(0, len(lines) - visible_lines)))

    yo = py + 30
    clip_bottom = py + ph - 6
    for txt, col in lines[scroll:scroll + visible_lines]:
        if yo > clip_bottom:
            break
        if col is not None and txt:
            _text(img, txt, px + ACCENT_BAR + PAD, yo, F_BODY, fs, col)
        yo += line_h_px

    # Scroll indicator — only shown when content overflows the panel
    if len(lines) > visible_lines:
        bar_x = px + help_w - 6
        bar_top, bar_bot = py + 28, py + ph - 8
        cv2.line(img, (bar_x, bar_top), (bar_x, bar_bot), GRAY_DK, 2)
        track = max(bar_bot - bar_top - 14, 4)
        thumb_y = bar_top + int(track * scroll / max(1, len(lines) - visible_lines))
        cv2.line(img, (bar_x, thumb_y), (bar_x, thumb_y + 14), ACCENT, 3)
        _text(img, "[ ] or wheel to scroll", px + ACCENT_BAR + PAD, py + ph - 6,
              F_BODY, TS_SMALL, GRAY_DK)

    return img


# ── Warmup prompt (shown after [i], before Space or Tab) ──────────────────

def draw_warmup_prompt(img, s: dict) -> np.ndarray:
    if s.get('warmup_active') or s.get('analytics_mode'):
        return img
    if not s.get('ai_loaded') or not s.get('inference_active'):
        return img
    ih, iw = img.shape[:2]
    pw, ph = 440, 70
    ox = (iw-pw)//2
    oy = ih//2 - 60

    _rect(img, ox, oy, pw, ph, BG, 0.92)
    cv2.rectangle(img, (ox, oy), (ox+pw, oy+ph), GRAY_DK, 1)
    _line(img, ox, oy, ox+pw, oy, ACCENT, 2)
    _text(img, "AI ONLINE — SELECT MODE", ox+PAD+8, oy+18,
          F_TITLE, 0.45, ACCENT)
    _text(img, "[Space] = Warmup  /  Rep Counter",
          ox+PAD+8, oy+40, F_BODY, 0.42, GREEN)
    _text(img, "[Tab]   = Analytics  /  Tennis & Pickleball",
          ox+PAD+8, oy+60, F_BODY, 0.42, YELLOW)
    return img


# ── History overlay (shown 6s after analytics start) ──────────────────────

def draw_history_overlay(img, history_lines: list, show_until: float) -> np.ndarray:
    if not history_lines or time.time() > show_until:
        return img
    ih, iw = img.shape[:2]
    lh  = 19
    pw  = min(iw-40, 680)
    ph  = len(history_lines)*lh + 52
    ox  = (iw-pw)//2
    oy  = max(MARGIN, (ih-ph)//2)
    draw_panel(img, ox, oy, pw, ph, "recent sessions", ACCENT)
    for i, line in enumerate(history_lines):
        col = GRAY_LT if i % 2 == 0 else GRAY_MD
        _text(img, line, ox+ACCENT_BAR+PAD, oy+34+i*lh, F_BODY, TS_SMALL, col)
    return img


# ── Paused overlay ─────────────────────────────────────────────────────────

def draw_paused_overlay(img) -> np.ndarray:
    ih, iw = img.shape[:2]
    _rect(img, 0, 0, iw, ih, (0,0,0), 0.55)
    msg = "PAUSED"
    tw, th = _text_size(msg, F_BIG, 1.8, 3)
    cv2.putText(img, msg, ((iw-tw)//2+2, ih//2+th//2+2),
                F_BIG, 1.8, (0,0,0), 5, cv2.LINE_AA)
    cv2.putText(img, msg, ((iw-tw)//2,   ih//2+th//2),
                F_BIG, 1.8, ACCENT, 3, cv2.LINE_AA)
    sm = "Press [P] to resume"
    sw, _ = _text_size(sm, F_BODY, 0.55)
    _text(img, sm, (iw-sw)//2, ih//2+th//2+36, F_BODY, 0.55, GRAY_MD)
    return img


# ── Toast notification ─────────────────────────────────────────────────────

def draw_toast(img, text: str, expiry: float, analytics_mode: bool) -> None:
    if not text or time.time() >= expiry:
        return
    ih, iw = img.shape[:2]
    tw, th = _text_size(text, F_BODY, 0.50)
    tx  = max(MARGIN, (iw-tw)//2)
    ty  = ih - (80 if analytics_mode else 52)
    pw  = min(tw+24, iw-20)
    _rect(img, tx-12, ty-th-4, pw, th+14, BG, 0.92)
    _line(img, tx-12, ty-th-4, tx-12+pw, ty-th-4, ACCENT, 2)
    _text(img, text, tx, ty, F_BODY, 0.50, WHITE)


# ── Warmup HUD panel (right side) ─────────────────────────────────────────

def draw_warmup_panel(img, dict_data: dict, iw: int) -> None:
    from pose_module import MODE_NONE, MODE_SQUAT, MODE_PUSHUP, MODE_CURL
    ex      = dict_data.get("exercise_data", {})
    mode    = ex.get("mode", MODE_NONE)
    fb      = ex.get("feedback", "")
    angs    = ex.get("angles", {})
    reps    = ex.get("rep_count", 0)
    state   = ex.get("state", "IDLE")
    last_q  = ex.get("last_quality")
    avg_q   = ex.get("avg_quality")
    wrapped = textwrap.wrap(fb, width=28) if fb else []

    mode_names   = {MODE_NONE:"STANDBY", 0:"STANDBY",'0':"STANDBY",
                    MODE_SQUAT:"SQUAT",'1':"SQUAT",
                    MODE_PUSHUP:"PUSHUP",'2':"PUSHUP",
                    MODE_CURL:"CURL",'3':"CURL"}
    mode_colors  = {MODE_SQUAT: ACCENT, 1: ACCENT, '1': ACCENT,
                    MODE_PUSHUP: ORANGE, 2: ORANGE, '2': ORANGE,
                    MODE_CURL: GREEN,  3: GREEN, '3': GREEN}
    mn    = mode_names.get(mode, "?")
    mcol  = mode_colors.get(mode, GRAY_MD)

    pw = PANEL_W
    # dynamic height — +26 for the quality bar row when a rep has happened
    quality_row_h = 26 if last_q is not None else 0
    dh = 60 + quality_row_h + len(wrapped)*18 + max(len(angs), 1)*18 + 30
    dx = iw - pw - MARGIN
    dy = MARGIN

    draw_panel(img, dx, dy, pw, dh, mn, mcol)

    # Rep count — large
    _text(img, str(reps), dx+ACCENT_BAR+PAD, dy+58, F_BIG, TS_BIG, mcol, 2)
    tw, _ = _text_size(str(reps), F_BIG, TS_BIG, 2)
    _text(img, "REPS", dx+ACCENT_BAR+PAD+tw+8, dy+52, F_BODY, TS_SMALL, GRAY_MD)

    # State badge
    scol = GREEN if state == "DOWN" else ORANGE
    draw_qual_badge(img, dx+ACCENT_BAR+PAD+tw+8, dy+68, state, scol)

    # ── Rep quality bar ────────────────────────────────────────────────────
    yc = dy + 78
    if last_q is not None:
        bar_x  = dx + ACCENT_BAR + PAD
        bar_y  = yc
        bar_w  = pw - ACCENT_BAR - PAD*2
        bar_h  = 10
        qcol   = GREEN if last_q >= 70 else YELLOW if last_q >= 45 else RED
        cv2.rectangle(img, (bar_x, bar_y), (bar_x+bar_w, bar_y+bar_h), GRAY_DK, -1)
        filled = int(bar_w * max(0, min(last_q, 100)) / 100)
        if filled > 0:
            cv2.rectangle(img, (bar_x, bar_y), (bar_x+filled, bar_y+bar_h), qcol, -1)
        avg_txt = f"  avg {avg_q}" if avg_q is not None else ""
        _text(img, f"QUALITY {last_q}%{avg_txt}", bar_x, bar_y+bar_h+13,
              F_BODY, TS_SMALL, qcol)
        yc += 32

    # Feedback
    for ln in wrapped:
        _text(img, ln, dx+ACCENT_BAR+PAD, yc, F_BODY, TS_SMALL, GRAY_LT)
        yc += 16
    if not wrapped:
        yc += 16

    # Divider
    _line(img, dx+ACCENT_BAR, yc+4, dx+pw-2, yc+4, GRAY_DK)

    # Angles
    hl = {MODE_SQUAT:["knee","hip"],MODE_PUSHUP:["elbow","hip","shoulder"],
          MODE_CURL:["elbow"]}.get(mode, [])
    yo = yc + 18
    for jn, av in angs.items():
        col = ACCENT if any(h in jn for h in hl) else GRAY_MD
        lbl = jn.replace('_',' ').title()
        val = f"{int(av)}\u00b0" if av is not None else "--"
        _text(img, f"{lbl}: {val}", dx+ACCENT_BAR+PAD, yo, F_BODY, TS_SMALL, col)
        yo += 17


# ── MAIN DRAW HUD ──────────────────────────────────────────────────────────

def draw_hud(img, fps: float, dict_data, s: dict) -> np.ndarray:
    ih, iw = img.shape[:2]
    now    = time.time()

    session_start = s.get('analytics_session_start', 0)

    # ── Broadcast bug (always visible) ─────────────────────────────────────
    draw_broadcast_bug(img, fps,
                       session_start if s.get('analytics_mode') else 0,
                       s.get('recording', False))

    # ── System offline ──────────────────────────────────────────────────────
    if not s.get('ai_loaded') or not s.get('inference_active'):
        pw, ph = 300, 58
        ox = (iw-pw)//2; oy = ih//2-60
        draw_panel(img, ox, oy, pw, ph, "system", RED)
        _text(img, "SYSTEM OFFLINE", ox+ACCENT_BAR+PAD, oy+42,
              F_BIG, 0.55, RED, 2)
        _text(img, "Press  [i]  to boot AI engine",
              ox+ACCENT_BAR+PAD, oy+62, F_BODY, TS_BODY, GRAY_MD)
        draw_analytics_bar(img, s)
        draw_help_overlay(img, s)
        draw_toast(img, s.get('toast_text',''), s.get('toast_expiry',0),
                   s.get('analytics_mode', False))
        return img

    # ── Choose-mode prompt ──────────────────────────────────────────────────
    if not s.get('warmup_active') and not s.get('analytics_mode'):
        draw_warmup_prompt(img, s)
        draw_help_overlay(img, s)
        draw_toast(img, s.get('toast_text',''), s.get('toast_expiry',0), False)
        return img

    # ── Compact mode: just bug + toast ─────────────────────────────────────
    if s.get('compact_hud'):
        draw_analytics_bar(img, s)
        draw_toast(img, s.get('toast_text',''), s.get('toast_expiry',0),
                   s.get('analytics_mode', False))
        return draw_help_overlay(img, s)

    # ── Lock status indicator (Warmup only — meaningless in Analytics) ──────
    if dict_data and s.get('warmup_active'):
        lock = "LOCKED" if dict_data.get("roi") else "SEARCHING"
        lcol = GREEN if lock == "LOCKED" else RED
        draw_qual_badge(img, MARGIN+ACCENT_BAR+3, 42, lock, lcol)
        if s.get('detector_preset'):
            _text(img, s['detector_preset'], MARGIN+ACCENT_BAR+3+70, 42,
                  F_BODY, TS_SMALL, DIM_CYAN)

    # ── Warmup panel (right side) ──────────────────────────────────────────
    if s.get('warmup_active') and dict_data and "exercise_data" in dict_data:
        draw_warmup_panel(img, dict_data, iw)

    # ── Toast ──────────────────────────────────────────────────────────────
    draw_toast(img, s.get('toast_text',''), s.get('toast_expiry',0),
               s.get('analytics_mode', False))

    # ── Analytics ticker and overlays ──────────────────────────────────────
    draw_analytics_bar(img, s)
    draw_history_overlay(img, s.get('history_lines',[]),
                         s.get('history_show_until', 0))
    return draw_help_overlay(img, s)


# ── Session end summary overlay ────────────────────────────────────────────

def draw_session_end_overlay(img, stats: dict) -> np.ndarray:
    """5-second post-session summary in broadcast style."""
    ih, iw = img.shape[:2]
    _rect(img, 0, 0, iw, ih, (0,0,0), 0.72)

    pw, ph = 520, 280
    ox = (iw-pw)//2
    oy = (ih-ph)//2
    draw_panel(img, ox, oy, pw, ph, "session complete", GREEN)

    _text(img, "SESSION COMPLETE", ox+ACCENT_BAR+PAD, oy+44, F_BIG, 0.70, GREEN, 2)

    # Stat grid — 4 columns
    stats_row = [
        ("STROKES",    str(stats.get('strokes', 0)),     ACCENT),
        ("SPLIT STEPS",str(stats.get('splits', 0)),      YELLOW),
        ("LUNGES",     str(stats.get('lunges', 0)),      ORANGE),
        ("JUMPS",      str(stats.get('jumps', 0)),       GREEN),
    ]
    col_w = (pw - ACCENT_BAR - PAD*2) // 4
    for i, (label, val, col) in enumerate(stats_row):
        cx = ox + ACCENT_BAR + PAD + i*col_w
        _text(img, val,   cx, oy+110, F_BIG,  0.90, col, 2)
        _text(img, label, cx, oy+130, F_BODY, TS_SMALL, GRAY_MD)

    # Second row
    row2 = [
        ("ACTIVE",    f"{stats.get('active_pct',0):.0f}%",  GREEN),
        ("AVG SPEED", f"{stats.get('avg_speed',0):.1f} BL/s", ACCENT),
        ("LONGEST RALLY", str(stats.get('longest_rally',0)), YELLOW),
        ("DURATION",  stats.get('duration','--'),           GRAY_LT),
    ]
    for i, (label, val, col) in enumerate(row2):
        cx = ox + ACCENT_BAR + PAD + i*col_w
        _text(img, val,   cx, oy+172, F_TITLE, 0.55, col, 1)
        _text(img, label, cx, oy+190, F_BODY,  TS_SMALL, GRAY_MD)

    _line(img, ox+ACCENT_BAR, oy+200, ox+pw-2, oy+200, GRAY_DK)
    saved = stats.get('saved_file', '')
    if saved:
        _text(img, f"Saved: {saved}", ox+ACCENT_BAR+PAD, oy+222,
              F_BODY, TS_SMALL, DIM_CYAN)
    _text(img, "Closing in a moment...", ox+ACCENT_BAR+PAD, oy+244,
          F_BODY, TS_SMALL, GRAY_DK)

    return img