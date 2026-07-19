# review.py
#
# Standalone session review tool. Run separately from main.py:
#     python review.py
#     python review.py --days 14        (look back further than default 7)
#     python review.py --plot           (also save a PNG trend chart)
#
# Loads every session_*.json in the current folder, prints a trend table,
# and highlights whether recent sessions show improvement or decline
# compared to the earlier average.

import os
import sys
import glob
import json
import argparse
from datetime import datetime
from typing import List, Dict, Any


def load_all_sessions(folder: str = ".") -> List[Dict[str, Any]]:
    """Loads every session_*.json, returns list of {filename, date, records}."""
    pattern = os.path.join(folder, "session_*.json")
    paths   = sorted(glob.glob(pattern))
    sessions = []

    for path in paths:
        try:
            with open(path) as f:
                records = json.load(f)
            if not records:
                continue
            fname = os.path.basename(path)
            try:
                ts_part  = fname.replace("session_", "").replace(".json", "")
                date_obj = datetime.strptime(ts_part, "%Y%m%d_%H%M%S")
            except ValueError:
                date_obj = datetime.fromtimestamp(os.path.getmtime(path))
            sessions.append({"filename": fname, "date": date_obj, "records": records})
        except Exception as e:
            print(f"  (skipped {path}: {e})")

    return sessions


def summarise(records: List[dict]) -> Dict[str, Any]:
    """Aggregate one session's records into summary stats. Uses the
    __summary__ trailer (real event counts from the live detectors) when
    present; older files without one fall back to inferring from the
    per-frame "pose" field, which only ever holds a movement-speed label
    so that fallback always reports zero events — kept for files saved
    before the trailer existed rather than crashing on them."""
    summary_row = None
    if records and isinstance(records[-1], dict) and records[-1].get("__summary__"):
        summary_row = records[-1]
        records = records[:-1]

    poses      = [r.get("pose", "") for r in records]
    speeds     = [r.get("speed", 0.0) for r in records if r.get("speed") is not None]
    timestamps = [r.get("timestamp", 0.0) for r in records if r.get("timestamp")]

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
        moving = sum(1 for p in poses if p not in ("ready", ""))
        total  = len(poses) or 1
        active_pct = round(100.0 * moving / total, 1)
        avg_speed  = round(sum(speeds) / len(speeds), 2) if speeds else 0.0

    duration_s = (max(timestamps) - min(timestamps)) if len(timestamps) >= 2 else 0.0

    return {
        "strokes":     stroke_count,
        "lunges":      lunge_count,
        "splits":      split_count,
        "jumps":       jump_count,
        "active_pct":  active_pct,
        "avg_speed":   avg_speed,
        "duration_s":  duration_s,
    }


def fmt_duration(sec: float) -> str:
    m, s = divmod(int(sec), 60)
    return f"{m}m{s:02d}s"


def trend_arrow(recent_avg: float, earlier_avg: float) -> str:
    if earlier_avg <= 0:
        return "  "
    ratio = recent_avg / earlier_avg
    if ratio > 1.10: return "↑"
    if ratio < 0.90: return "↓"
    return "→"


def print_table(sessions: List[Dict[str, Any]]) -> None:
    print()
    print(f"{'DATE':<17} {'DURATION':<9} {'STROKES':<8} {'LUNGES':<7} "
          f"{'SPLITS':<7} {'JUMPS':<6} {'ACTIVE%':<8} {'AVG SPD':<8}")
    print("-" * 78)

    summaries = []
    for sess in sessions:
        summ = summarise(sess["records"])
        summaries.append(summ)
        date_str = sess["date"].strftime("%d %b  %H:%M")
        print(f"{date_str:<17} {fmt_duration(summ['duration_s']):<9} "
              f"{summ['strokes']:<8} {summ['lunges']:<7} {summ['splits']:<7} "
              f"{summ['jumps']:<6} {summ['active_pct']:<8.1f} {summ['avg_speed']:<8.2f}")

    print("-" * 78)

    if len(summaries) >= 4:
        half = len(summaries) // 2
        earlier = summaries[:half]
        recent  = summaries[half:]

        def avg(key, lst):
            return sum(x[key] for x in lst) / len(lst) if lst else 0

        print()
        print("TREND  (first half of history vs second half)")
        for key, label in [("strokes","Strokes/session"),
                           ("active_pct","Active %"),
                           ("avg_speed","Avg speed (BL/s)")]:
            e_avg = avg(key, earlier)
            r_avg = avg(key, recent)
            arrow = trend_arrow(r_avg, e_avg)
            print(f"  {label:<20} {e_avg:>7.2f}  →  {r_avg:>7.2f}   {arrow}")
    else:
        print()
        print("Need at least 4 sessions for a meaningful trend comparison.")

    print()
    print(f"Total sessions found: {len(sessions)}")


def save_plot(sessions: List[Dict[str, Any]], out_path: str = "session_trend.png") -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot. "
             "Run: pip install matplotlib --break-system-packages")
        return

    summaries = [summarise(s["records"]) for s in sessions]
    dates     = [s["date"] for s in sessions]

    fig, axes = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    fig.patch.set_facecolor("#16120a")
    for ax in axes:
        ax.set_facecolor("#16120a")
        ax.tick_params(colors="#999999")
        for spine in ax.spines.values():
            spine.set_color("#444444")

    axes[0].plot(dates, [s["strokes"] for s in summaries], marker="o", color="#00c3ff")
    axes[0].set_title("Strokes per session", color="#dddddd")

    axes[1].plot(dates, [s["active_pct"] for s in summaries], marker="o", color="#5aff8c")
    axes[1].set_title("Active %", color="#dddddd")

    axes[2].plot(dates, [s["avg_speed"] for s in summaries], marker="o", color="#ffd700")
    axes[2].set_title("Avg speed (body-lengths/sec)", color="#dddddd")

    plt.tight_layout()
    plt.savefig(out_path, facecolor=fig.get_facecolor())
    print(f"Saved trend chart: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Review past tracking sessions")
    parser.add_argument("--days", type=int, default=0,
                        help="Only show sessions from the last N days (0 = all)")
    parser.add_argument("--plot", action="store_true",
                        help="Save a PNG trend chart (requires matplotlib)")
    parser.add_argument("--folder", type=str, default=".",
                        help="Folder to scan for session_*.json files")
    args = parser.parse_args()

    print("Loading sessions...")
    sessions = load_all_sessions(args.folder)

    if args.days > 0:
        cutoff = datetime.now().timestamp() - args.days * 86400
        sessions = [s for s in sessions if s["date"].timestamp() >= cutoff]

    if not sessions:
        print("No session_*.json files found in this folder.")
        sys.exit(0)

    print_table(sessions)

    if args.plot:
        save_plot(sessions)


if __name__ == "__main__":
    main()