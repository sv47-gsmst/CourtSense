# court_calibration.py
#
# Click 4 court corners to compute a homography, giving real metre
# coordinates instead of raw camera pixels. Once calibrated this
# powers real player distances, a bird's-eye heatmap, and zone
# boundaries measured in actual court dimensions rather than screen
# fractions.
#
# Press [C] to start, click corners in order TL -> TR -> BR -> BL,
# saves to config.json automatically on the 4th click.

import cv2
import json
import numpy as np
import os
from typing import List, Tuple, Optional

# Supported court types — width × length in metres
COURT_TYPES = {
    "pickleball": (6.10, 13.41),
    "tennis_singles": (8.23, 23.77),
    "tennis_doubles": (10.97, 23.77),
    "badminton": (6.10, 13.40),
    "generic": (10.0, 20.0),
}

# Bird's-eye canvas size in pixels (proportional to court dimensions)
BIRDEYE_W = 180
BIRDEYE_H = 360


class CourtCalibrator:
    def __init__(self, court_type: str = "pickleball",
                 config_path: str = "config.json"):
        self.court_type  = court_type
        self.config_path = config_path
        cw, cl           = COURT_TYPES.get(court_type, COURT_TYPES["generic"])
        self.court_w_m   = cw
        self.court_l_m   = cl

        self._clicks: List[Tuple[int, int]] = []   # pixel coords, in order
        self.calibrated  = False
        self._H: Optional[np.ndarray] = None       # 3×3 homography matrix
        self._H_inv: Optional[np.ndarray] = None   # inverse (for bird's-eye)
        self.calibrating = False                    # True while collecting clicks

        # Try loading existing calibration
        self._load()

    # ── Real-world destination corners (metres, TL/TR/BR/BL) ──────────────
    @property
    def _dst_pts(self) -> np.ndarray:
        w, l = self.court_w_m, self.court_l_m
        return np.array([[0, 0], [w, 0], [w, l], [0, l]],
                        dtype=np.float32)

    # ── Mouse callback — called by OpenCV ─────────────────────────────────
    def on_mouse(self, event, x: int, y: int, flags, param) -> None:
        if not self.calibrating:
            return
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(self._clicks) < 4:
                self._clicks.append((x, y))
                if len(self._clicks) == 4:
                    self._compute_homography()

    def _compute_homography(self) -> None:
        src = np.array(self._clicks, dtype=np.float32)
        dst = self._dst_pts
        self._H, _    = cv2.findHomography(src, dst)
        self._H_inv, _ = cv2.findHomography(dst, src)
        if self._H is not None:
            self.calibrated  = True
            self.calibrating = False
            self._save()

    # ── Coordinate transforms ──────────────────────────────────────────────
    def px_to_metres(self, px: float, py: float) -> Tuple[float, float]:
        """Convert pixel position to court metres (0,0 = TL corner)."""
        if not self.calibrated or self._H is None:
            return (0.0, 0.0)
        pt  = np.array([[[px, py]]], dtype=np.float32)
        out = cv2.perspectiveTransform(pt, self._H)
        return (float(out[0][0][0]), float(out[0][0][1]))

    def metres_to_birdeye_px(self, mx: float, my: float) -> Tuple[int, int]:
        """Convert court metres to bird's-eye canvas pixel position."""
        bx = int(np.clip(mx / self.court_w_m * BIRDEYE_W, 0, BIRDEYE_W - 1))
        by = int(np.clip(my / self.court_l_m * BIRDEYE_H, 0, BIRDEYE_H - 1))
        return (bx, by)

    def distance_metres(self, px1, py1, px2, py2) -> float:
        """Real-world distance in metres between two pixel positions."""
        m1 = self.px_to_metres(px1, py1)
        m2 = self.px_to_metres(px2, py2)
        return float(np.hypot(m1[0]-m2[0], m1[1]-m2[1]))

    # ── Save / load ────────────────────────────────────────────────────────
    def _save(self) -> None:
        cfg = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    cfg = json.load(f)
            except Exception:
                pass
        cfg["court_calibration"] = {
            "court_type": self.court_type,
            "clicks":     self._clicks,
            "H":          self._H.tolist() if self._H is not None else None,
        }
        with open(self.config_path, "w") as f:
            json.dump(cfg, f, indent=2)

    def _load(self) -> None:
        if not os.path.exists(self.config_path):
            return
        try:
            with open(self.config_path) as f:
                cfg = json.load(f)
            cal = cfg.get("court_calibration")
            if not cal or cal.get("H") is None:
                return
            self.court_type = cal.get("court_type", self.court_type)
            cw, cl          = COURT_TYPES.get(self.court_type, COURT_TYPES["generic"])
            self.court_w_m  = cw
            self.court_l_m  = cl
            self._clicks    = [tuple(c) for c in cal["clicks"]]
            self._H         = np.array(cal["H"], dtype=np.float32)
            self._H_inv, _  = cv2.findHomography(self._dst_pts,
                                                  np.array(self._clicks, dtype=np.float32))
            self.calibrated = True
        except Exception:
            pass

    def clear(self) -> None:
        # Full delete, both in memory and config.json — cancelling mid-
        # calibration only clears the clicks being collected right now,
        # any earlier calibration would still be sitting there and drawing.
        self._clicks     = []
        self._H           = None
        self._H_inv       = None
        self.calibrated   = False
        self.calibrating  = False

        if os.path.exists(self.config_path):
            try:
                with open(self.config_path) as f:
                    cfg = json.load(f)
                if "court_calibration" in cfg:
                    del cfg["court_calibration"]
                    with open(self.config_path, "w") as f:
                        json.dump(cfg, f, indent=2)
            except Exception:
                pass

    # ── Draw overlays ──────────────────────────────────────────────────────
    def draw_calibration_overlay(self, img: np.ndarray) -> np.ndarray:
        """Draws calibration UI during corner collection."""
        ih, iw = img.shape[:2]
        labels = ["1: Top-Left", "2: Top-Right", "3: Bottom-Right", "4: Bottom-Left"]
        colours = [(0,255,0), (0,200,255), (255,60,200), (255,200,0)]
        n = len(self._clicks)

        # Instruction banner
        remaining = labels[n] if n < 4 else "Done"
        cv2.rectangle(img, (0, ih-52), (iw, ih), (15,15,15), -1)
        cv2.putText(img, f"CALIBRATION MODE — click corner  {remaining}",
                    (12, ih-30), cv2.FONT_HERSHEY_DUPLEX, 0.55,
                    (0,220,255), 1, cv2.LINE_AA)
        cv2.putText(img, "Order: TL → TR → BR → BL   |   [C] to cancel",
                    (12, ih-10), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                    (150,150,150), 1, cv2.LINE_AA)

        # Draw collected clicks
        for i, (cx, cy) in enumerate(self._clicks):
            col = colours[i]
            cv2.circle(img, (cx, cy), 10, col, 2, cv2.LINE_AA)
            cv2.circle(img, (cx, cy),  3, col, -1)
            cv2.putText(img, str(i+1), (cx+12, cy+5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1, cv2.LINE_AA)

        # Draw quadrilateral after 4 clicks
        if n == 4:
            pts = np.array(self._clicks, dtype=np.int32).reshape((-1,1,2))
            cv2.polylines(img, [pts], isClosed=True, color=(0,220,255), thickness=2)

        return img

    def draw_court_boundary(self, img: np.ndarray) -> np.ndarray:
        """When calibrated, draws the court boundary outline on the live feed."""
        if not self.calibrated or len(self._clicks) != 4:
            return img
        pts = np.array(self._clicks, dtype=np.int32).reshape((-1,1,2))
        cv2.polylines(img, [pts], isClosed=True, color=(0,200,80), thickness=1,
                      lineType=cv2.LINE_AA)
        return img

    def draw_birdeye_panel(self, img: np.ndarray,
                           player_positions: List[Tuple[float,float]],
                           panel_x: int, panel_y: int) -> np.ndarray:
        """
        Draws a top-down bird's-eye court diagram with player dots.
        player_positions: list of (pixel_x, pixel_y) camera-space positions.
        """
        canvas = np.zeros((BIRDEYE_H + 20, BIRDEYE_W + 10, 3), dtype=np.uint8)
        canvas[:] = (25, 35, 30)

        # Court outline
        cv2.rectangle(canvas, (4, 4), (BIRDEYE_W+5, BIRDEYE_H+15),
                      (0, 180, 80), 1)
        # Centre line
        cy = (BIRDEYE_H + 20) // 2
        cv2.line(canvas, (4, cy), (BIRDEYE_W+5, cy), (0,140,60), 1)

        # Court type label
        cv2.putText(canvas, self.court_type.replace("_"," "),
                    (5, BIRDEYE_H+18), cv2.FONT_HERSHEY_SIMPLEX,
                    0.28, (100,150,100), 1)

        # Player dots
        colours = [(0,230,255), (255,80,180), (100,255,100), (255,200,0)]
        for i, (px, py) in enumerate(player_positions):
            mx, my = self.px_to_metres(px, py)
            bx, by = self.metres_to_birdeye_px(mx, my)
            col    = colours[i % len(colours)]
            cv2.circle(canvas, (bx+4, by+4), 5, col, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"P{i+1}", (bx+8, by+8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1)

        # Paste onto main frame
        ph, pw = canvas.shape[:2]
        y2 = min(panel_y + ph, img.shape[0])
        x2 = min(panel_x + pw, img.shape[1])
        img[panel_y:y2, panel_x:x2] = canvas[:y2-panel_y, :x2-panel_x]
        cv2.rectangle(img, (panel_x, panel_y), (x2-1, y2-1), (60,80,80), 1)
        return img