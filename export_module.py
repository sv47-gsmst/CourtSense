# export_module.py
# CSV fieldnames are built dynamically from whatever keys actually show
# up in the records, so angle columns like torso/left_arm never go missing.
import csv
import json
import time
from datetime import datetime

_BASE_FIELDS = ["frame", "track_id", "cx", "cy",
                "speed", "accel", "pose", "timestamp"]


def _ts() -> str:
    """Returns a compact timestamp string for use in filenames."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")


class SessionExporter:
    def __init__(self):
        self.records: list[dict] = []

    def record(self, frame_idx: int, track_id: int, center: tuple,
               speed: float, accel: float,
               pose_label: str, angles: dict) -> None:
        row = {
            "frame":     frame_idx,
            "track_id":  track_id,
            "cx":        round(center[0], 1),
            "cy":        round(center[1], 1),
            "speed":     round(speed, 2),
            "accel":     round(accel, 2),
            "pose":      pose_label,
            "timestamp": round(time.time(), 3),
        }
        for k, v in angles.items():
            row[k] = round(v, 1) if v is not None else None
        self.records.append(row)

    def save_csv(self, path: str = "") -> str:
        # Default to timestamped filename so sessions never overwrite each other
        if not path:
            path = f"session_{_ts()}.csv"
        if not self.records:
            return path
        all_keys: list[str] = list(_BASE_FIELDS)
        extra: list[str] = []
        for row in self.records:
            for k in row:
                if k not in all_keys and k not in extra:
                    extra.append(k)
        fieldnames = all_keys + extra
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames,
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(self.records)
        return path

    def save_json(self, path: str = "", summary: dict = None) -> str:
        # Optional summary trailer: real event counts (strokes/lunges/etc)
        # from the live detectors, appended as one extra record. Without
        # this, the only "pose" field ever written is a movement-speed
        # label (ready/shuffling/jogging/sprinting) — never "stroke" or
        # "lunge" — so trying to infer event counts by searching for those
        # substrings later (as load_history() used to) always finds zero.
        if not path:
            path = f"session_{_ts()}.json"
        data = list(self.records)
        if summary:
            data.append({"__summary__": True, **summary})
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return path

    def reset(self) -> None:
        self.records = []


# ── Session history ─────────────────────────────────────────────────────
from dataclasses import dataclass, field
from typing import List, Optional
import os
import glob


@dataclass
class SessionSummary:
    """Lightweight summary of one past session loaded from its JSON file."""
    filename:      str
    date_str:      str
    stroke_count:  int   = 0
    lunge_count:   int   = 0
    split_count:   int   = 0
    jump_count:    int   = 0
    active_pct:    float = 0.0
    avg_speed:     float = 0.0
    duration_sec:  float = 0.0


def load_history(folder: str = ".", max_sessions: int = 7) -> List[SessionSummary]:
    """
    Scans `folder` for session_*.json files, loads up to `max_sessions`
    of the most recent ones, and returns a list of SessionSummary objects.

    The JSON files written by SessionExporter.save_json() contain one
    record per analytics frame. We aggregate across the whole file to
    produce per-session totals, which are then displayed as a startup
    summary when analytics mode is first enabled.

    Any file that fails to parse (corrupted, wrong format) is silently
    skipped so a bad file never prevents the program from starting.
    """
    pattern = os.path.join(folder, "session_*.json")
    paths   = sorted(glob.glob(pattern), reverse=True)[:max_sessions]

    summaries: List[SessionSummary] = []
    for path in paths:
        try:
            with open(path, "r") as f:
                records = json.load(f)
            if not records:
                continue

            fname    = os.path.basename(path)
            # Parse date from filename: session_YYYYMMDD_HHMMSS.json
            try:
                ts_part  = fname.replace("session_", "").replace(".json", "")
                date_obj = datetime.strptime(ts_part, "%Y%m%d_%H%M%S")
                date_str = date_obj.strftime("%d %b  %H:%M")
            except ValueError:
                date_str = fname

            # Real event counts if this file has a summary trailer
            # (written by save_json since this fix), else fall back to
            # inferring from the per-frame "pose" field for older files —
            # that inference only ever finds zero since "pose" is a
            # movement-speed label, never "stroke"/"lunge"/"split", but
            # it's kept as a graceful fallback rather than crashing on
            # files saved before this fix existed.
            summary_row = None
            if records and isinstance(records[-1], dict) and records[-1].get("__summary__"):
                summary_row = records[-1]
                records = records[:-1]

            poses       = [r.get("pose", "") for r in records]
            speeds      = [r.get("speed", 0.0) for r in records if r.get("speed") is not None]
            timestamps  = [r.get("timestamp", 0.0) for r in records if r.get("timestamp")]

            if summary_row:
                stroke_count = summary_row.get("strokes", 0)
                lunge_count  = summary_row.get("lunges", 0)
                split_count  = summary_row.get("splits", 0)
                jump_count   = summary_row.get("jumps", 0)
                active_pct   = round(summary_row.get("active_pct", 0.0), 1)
                avg_speed    = round(summary_row.get("avg_speed", 0.0), 2)
            else:
                stroke_count = sum(1 for p in poses if "stroke" in p)
                lunge_count  = sum(1 for p in poses if "lunge"  in p)
                split_count  = sum(1 for p in poses if "split"  in p)
                jump_count   = 0
                moving  = sum(1 for p in poses if p not in ("ready", ""))
                total   = len(poses) or 1
                active_pct = round(100.0 * moving / total, 1)
                avg_speed  = round(sum(speeds) / len(speeds), 2) if speeds else 0.0

            duration = (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0

            summaries.append(SessionSummary(
                filename     = fname,
                date_str     = date_str,
                stroke_count = stroke_count,
                lunge_count  = lunge_count,
                split_count  = split_count,
                jump_count   = jump_count,
                active_pct   = active_pct,
                avg_speed    = avg_speed,
                duration_sec = round(duration, 1),
            ))
        except Exception:
            continue   # silently skip bad files

    return summaries


def format_history_lines(summaries: List[SessionSummary]) -> List[str]:
    """
    Returns a list of short strings suitable for drawing on screen,
    one per past session, most recent first.
    """
    if not summaries:
        return ["No previous sessions found"]
    lines = []
    for s in summaries:
        dur = f"{int(s.duration_sec//60)}m{int(s.duration_sec%60):02d}s"
        lines.append(
            f"{s.date_str}  |  {dur}  |  "
            f"Strokes:{s.stroke_count}  Active:{s.active_pct:.0f}%  "
            f"Spd:{s.avg_speed:.1f}BL/s"
        )
    return lines