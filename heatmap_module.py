# Camera-space heatmap plus a court-normalized version for when
# court calibration is active.
#
# get_overlay() caches the coloured layer and only rebuilds it when
# add_point()/decay_step() actually changed something — otherwise it's
# just normalize+colormap+blend on every frame even while the player's
# standing still, which adds up.
#
# The camera-space heatmap shows where a player stood in pixel
# coordinates, which breaks across sessions if the camera moves and
# gets distorted by perspective (someone close to the camera occupies
# more pixels per metre than someone far away). add_point_court() uses
# the court calibrator's homography to accumulate into a fixed grid in
# real metres instead, so get_court_overlay() is comparable session to
# session regardless of camera position.
import cv2
import numpy as np

COURT_GRID_W = 120   # maps to court width
COURT_GRID_H = 240   # maps to court length


class HeatmapAccumulator:
    def __init__(self, decay: float = 0.998):
        self.decay   = decay
        self._map    = None
        self._w      = 0
        self._h      = 0
        self._dirty  = False          # True when _map changed since last rebuild
        self._cached_overlay = None   # last built coloured layer (BGR uint8)

        # Court-space accumulator (only used when court calibration is active)
        self._court_map    = np.zeros((COURT_GRID_H, COURT_GRID_W), dtype=np.float32)
        self._court_dirty  = False
        self._court_cached = None

    def _ensure_map(self, w: int, h: int) -> None:
        if self._map is None or self._w != w or self._h != h:
            self._map   = np.zeros((h, w), dtype=np.float32)
            self._w, self._h = w, h
            self._dirty = True
            self._cached_overlay = None

    def add_point(self, x: float, y: float,
                  frame_w: int, frame_h: int,
                  radius: int = 24, intensity: float = 1.0) -> None:
        self._ensure_map(frame_w, frame_h)
        ix = int(np.clip(x, 0, self._w - 1))
        iy = int(np.clip(y, 0, self._h - 1))
        cv2.circle(self._map, (ix, iy), radius, intensity, -1)
        self._dirty = True

    def decay_step(self) -> None:
        if self._map is not None:
            self._map   *= self.decay
            self._dirty  = True

    def get_overlay(self, frame: np.ndarray, alpha: float = 0.45) -> np.ndarray:
        fh, fw = frame.shape[:2]
        self._ensure_map(fw, fh)

        if self._map.max() < 1e-6:
            return frame

        # Rebuild the coloured layer only when something changed
        if self._dirty or self._cached_overlay is None:
            norm    = cv2.normalize(self._map, None, 0, 255, cv2.NORM_MINMAX)
            norm_u8 = norm.astype(np.uint8)
            colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
            mask    = (norm_u8 > 8).astype(np.float32)
            mask_3  = np.stack([mask] * 3, axis=2)
            # Store the pre-multiplied overlay and its mask so blending is
            # just one addWeighted call on subsequent identical frames
            self._cached_overlay = (colored.astype(np.float32),
                                    mask_3, alpha)
            self._dirty = False

        colored_f, mask_3, alpha = self._cached_overlay
        blended = (frame.astype(np.float32) * (1.0 - alpha * mask_3)
                   + colored_f * alpha * mask_3)
        return blended.astype(np.uint8)

    def reset(self) -> None:
        if self._map is not None:
            self._map[:] = 0
        self._dirty          = True
        self._cached_overlay = None
        self._court_map[:]   = 0
        self._court_dirty    = True
        self._court_cached   = None

    # ── Court-space accumulation ────────────────────────────────────────────
    def add_point_court(self, court_x_m: float, court_y_m: float,
                        court_w_m: float, court_l_m: float,
                        radius: int = 6, intensity: float = 1.0) -> None:
        """
        Accumulates into the court-metre grid. Caller (pipeline_b.py) is
        responsible for converting camera pixels to court metres first via
        CourtCalibrator.px_to_metres() — this method just handles the
        metres → grid-cell mapping and painting.
        """
        gx = int(np.clip(court_x_m / max(court_w_m, 0.1) * COURT_GRID_W,
                         0, COURT_GRID_W - 1))
        gy = int(np.clip(court_y_m / max(court_l_m, 0.1) * COURT_GRID_H,
                         0, COURT_GRID_H - 1))
        cv2.circle(self._court_map, (gx, gy), radius, intensity, -1)
        self._court_dirty = True

    def decay_step_court(self) -> None:
        self._court_map  *= self.decay
        self._court_dirty = True

    def get_court_overlay(self, court_type_label: str = "") -> np.ndarray:
        """
        Renders the court-space heatmap as a standalone top-down image
        (not overlaid on a camera frame — this is a flat court diagram).
        Returns a BGR uint8 image of size (COURT_GRID_H+30, COURT_GRID_W, 3).
        """
        canvas = np.zeros((COURT_GRID_H + 30, COURT_GRID_W, 3), dtype=np.uint8)
        canvas[:] = (20, 28, 24)

        if self._court_map.max() > 1e-6:
            if self._court_dirty or self._court_cached is None:
                norm    = cv2.normalize(self._court_map, None, 0, 255, cv2.NORM_MINMAX)
                norm_u8 = norm.astype(np.uint8)
                colored = cv2.applyColorMap(norm_u8, cv2.COLORMAP_JET)
                mask    = (norm_u8 > 6).astype(np.float32)
                self._court_cached = (colored, mask)
                self._court_dirty  = False
            colored, mask = self._court_cached
            mask_3 = np.stack([mask]*3, axis=2)
            court_area = canvas[:COURT_GRID_H, :COURT_GRID_W].astype(np.float32)
            blended = court_area * (1-0.75*mask_3) + colored.astype(np.float32)*0.75*mask_3
            canvas[:COURT_GRID_H, :COURT_GRID_W] = blended.astype(np.uint8)

        # Court boundary
        cv2.rectangle(canvas, (2, 2), (COURT_GRID_W-2, COURT_GRID_H-2),
                      (0, 180, 80), 1)
        cv2.line(canvas, (2, COURT_GRID_H//2), (COURT_GRID_W-2, COURT_GRID_H//2),
                 (0, 140, 60), 1)

        if court_type_label:
            cv2.putText(canvas, court_type_label.replace("_"," "),
                       (5, COURT_GRID_H+18), cv2.FONT_HERSHEY_SIMPLEX,
                       0.32, (110,150,110), 1, cv2.LINE_AA)

        return canvas