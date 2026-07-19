# dashboard_module.py
#
# Separate OpenCV window with live speed/accel graphs plus event counts
# and an "active %" summary — a real per-session number, unlike frame
# counts per pose label. Graph scales (SPEED_MAX_VAL/ACCEL_MAX_VAL) are
# tuned for torso-lengths/sec units, not raw pixel speed.

import cv2
import base64
import numpy as np
from collections import deque


class LiveDashboard:
    """Separate OpenCV window with live analytics graphs and counters."""

    WINDOW = "Analytics Dashboard"
    W, H   = 800, 500

    # Display ranges for the rolling graphs (torso-lengths/sec and /sec^2)
    SPEED_MAX_VAL = 6.0
    ACCEL_MAX_VAL = 8.0

    def __init__(self):
        self.speed_buf: deque = deque(maxlen=200)
        self.accel_buf: deque = deque(maxlen=200)
        self.label_buf: deque = deque(maxlen=200)   # movement-tier timeline

        self._open = False

        self.stroke_count = 0
        self.lunge_count  = 0
        self.split_count  = 0
        self.jump_count   = 0
        self.time_idle    = 0.0
        self.time_moving  = 0.0

    def open(self) -> None:
        cv2.namedWindow(self.WINDOW, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW, self.W, self.H)
        self._open = True

    def close(self) -> None:
        if self._open:
            try:
                cv2.destroyWindow(self.WINDOW)
            except Exception:
                pass
        self._open = False

    def reset(self) -> None:
        self.speed_buf.clear()
        self.accel_buf.clear()
        self.label_buf.clear()
        self.stroke_count = 0
        self.lunge_count  = 0
        self.split_count  = 0
        self.jump_count   = 0
        self.time_idle    = 0.0
        self.time_moving  = 0.0

    def feed(self, speed: float, accel: float, movement_label: str,
            stroke_count: int, lunge_count: int, split_count: int,
            jump_count: int, time_idle: float, time_moving: float) -> None:
        self.speed_buf.append(speed)
        self.accel_buf.append(accel)
        self.label_buf.append(movement_label)

        self.stroke_count = stroke_count
        self.lunge_count  = lunge_count
        self.split_count  = split_count
        self.jump_count   = jump_count
        self.time_idle    = time_idle
        self.time_moving  = time_moving

    def _draw_graph(self, canvas, buf, y_top, y_bot, color, label, max_val):
        if not buf:
            cv2.putText(canvas, label, (5, y_top + 14),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
            return

        pts   = list(buf)
        w_per = (self.W - 80) / max(len(pts) - 1, 1)
        mid_y = (y_top + y_bot) // 2

        prev = None
        for i, v in enumerate(pts):
            x = int(40 + i * w_per)
            clamped = float(np.clip(v / max_val, -1, 1))
            y = int(mid_y - clamped * (y_bot - y_top) / 2)
            if prev:
                cv2.line(canvas, prev, (x, y), color, 1, cv2.LINE_AA)
            prev = (x, y)

        cv2.line(canvas, (40, mid_y), (self.W - 40, mid_y), (60, 60, 60), 1)
        cv2.putText(canvas, label, (5, y_top + 14),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"+/-{max_val:.0f}", (self.W - 75, y_top + 14),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.35, (120, 120, 120), 1, cv2.LINE_AA)

    def render(self) -> None:
        if not self._open:
            return

        canvas = np.zeros((self.H, self.W, 3), dtype=np.uint8)

        self._draw_graph(canvas, self.speed_buf, 20, 140,
                        (0, 220, 255), "Speed (torso-lengths/s)",
                        self.SPEED_MAX_VAL)
        self._draw_graph(canvas, self.accel_buf, 160, 280,
                        (100, 255, 100), "Accel (torso-lengths/s^2)",
                        self.ACCEL_MAX_VAL)

        # ── Discrete event counters (left column) ──────────────────────────
        sx, sy = 40, 300
        cv2.putText(canvas, f"Strokes:     {self.stroke_count}", (sx, sy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 1)
        cv2.putText(canvas, f"Lunges:      {self.lunge_count}", (sx, sy + 26),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 60, 200), 1)
        cv2.putText(canvas, f"Split-steps: {self.split_count}", (sx, sy + 52),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 1)
        cv2.putText(canvas, f"Jumps:       {self.jump_count}", (sx, sy + 78),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1)

        # ── Time breakdown (right column) ───────────────────────────────────
        x2 = 360
        total = self.time_idle + self.time_moving
        active_pct = (100.0 * self.time_moving / total) if total > 0 else 0.0

        cv2.putText(canvas, f"Idle:    {self.time_idle:5.1f}s", (x2, sy),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 255), 1)
        cv2.putText(canvas, f"Active:  {self.time_moving:5.1f}s", (x2, sy + 26),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)
        cv2.putText(canvas, f"Active %: {active_pct:4.0f}%", (x2, sy + 52),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 100), 1)

        # ── Movement-tier timeline strip ────────────────────────────────────
        strip_y = 460
        label_colors = {
            "ready":      (50, 255, 100),
            "shuffling":  (0, 220, 220),
            "jogging":    (0, 200, 255),
            "sprinting":  (0, 60, 255),
            "stroke":     (0, 230, 255),
            "lunge":      (255, 60, 200),
            "split-step": (0, 200, 80),
        }
        pts   = list(self.label_buf)
        bar_w = max(1, (self.W - 80) // max(len(pts), 1))
        for i, lbl in enumerate(pts):
            x   = 40 + i * bar_w
            col = label_colors.get(lbl, (80, 80, 80))
            cv2.rectangle(canvas, (x, strip_y), (x + bar_w, strip_y + 16), col, -1)
        cv2.putText(canvas, "Movement timeline", (5, strip_y + 12),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1)

        cv2.imshow(self.WINDOW, canvas)


# ── HTML session report ─────────────────────────────────────────────────
def export_html_report(speed_data, accel_data, heatmap_img, action_counts,
                       time_idle: float, time_moving: float,
                       filename: str = "session_report.html") -> str:
    """
    action_counts: dict like {"Strokes": n, "Lunges": n, "Split-steps": n,
    "Jumps": n} — the discrete event totals from action_module detectors.
    speed_data / accel_data: lists in torso-lengths/sec and /sec^2.
    """
    ok, buf = cv2.imencode(".png", heatmap_img)
    b64 = base64.b64encode(buf).decode() if ok else ""

    total = time_idle + time_moving
    active_pct = (100.0 * time_moving / total) if total > 0 else 0.0

    speed_js  = ",".join(f"{v:.2f}" for v in speed_data)
    accel_js  = ",".join(f"{v:.2f}" for v in accel_data)
    labels_js = ",".join(f'"{k}"' for k in action_counts)
    values_js = ",".join(str(v) for v in action_counts.values())

    summary_rows = "".join(
        f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in action_counts.items()
    )

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Session Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body{{font-family:sans-serif;background:#111;color:#eee;padding:20px}}
canvas{{max-width:700px;display:block;margin:20px auto}}
img{{display:block;margin:20px auto;max-width:700px;border:1px solid #333}}
table{{margin:10px auto;border-collapse:collapse}}
td{{padding:4px 16px;border-bottom:1px solid #333}}
h1,h2{{text-align:center}}
.summary{{text-align:center;color:#aaa}}
</style>
</head><body>
<h1>Session Report</h1>
<p class="summary">
  Idle: <b>{time_idle:.1f}s</b> &nbsp;|&nbsp;
  Active: <b>{time_moving:.1f}s</b> &nbsp;|&nbsp;
  Active: <b>{active_pct:.0f}%</b>
</p>
<table>{summary_rows}</table>

<h2>Speed (torso-lengths / second)</h2>
<canvas id="speed"></canvas>

<h2>Acceleration (torso-lengths / second^2)</h2>
<canvas id="accel"></canvas>

<h2>Event Counts</h2>
<canvas id="actions"></canvas>

<h2>Movement Heatmap</h2>
<img src="data:image/png;base64,{b64}" alt="heatmap">

<script>
new Chart(document.getElementById('speed'), {{
  type: 'line',
  data: {{
    labels: [...Array([{speed_js}].length).keys()],
    datasets: [{{label: 'Speed', data: [{speed_js}], borderColor: '#00dcff',
                 borderWidth: 1, pointRadius: 0}}]
  }},
  options: {{plugins: {{legend: {{labels: {{color: '#eee'}}}}}},
             scales: {{x: {{ticks: {{color: '#999'}}}}, y: {{ticks: {{color: '#999'}}}}}}}}
}});
new Chart(document.getElementById('accel'), {{
  type: 'line',
  data: {{
    labels: [...Array([{accel_js}].length).keys()],
    datasets: [{{label: 'Acceleration', data: [{accel_js}], borderColor: '#88ff88',
                 borderWidth: 1, pointRadius: 0}}]
  }},
  options: {{plugins: {{legend: {{labels: {{color: '#eee'}}}}}},
             scales: {{x: {{ticks: {{color: '#999'}}}}, y: {{ticks: {{color: '#999'}}}}}}}}
}});
new Chart(document.getElementById('actions'), {{
  type: 'bar',
  data: {{
    labels: [{labels_js}],
    datasets: [{{label: 'Count', data: [{values_js}], backgroundColor: '#ff7755'}}]
  }},
  options: {{plugins: {{legend: {{labels: {{color: '#eee'}}}}}},
             scales: {{x: {{ticks: {{color: '#999'}}}}, y: {{ticks: {{color: '#999'}}}}}}}}
}});
</script>
</body></html>"""

    with open(filename, "w") as f:
        f.write(html)
    return filename