# Racket detection using SSD MobileNet v2, pretrained on COCO, run
# through OpenCV's DNN module (no PyTorch/TF needed at runtime).
#
# COCO class 43 is "tennis racket" so this is real detection, not a
# proxy. Worth knowing though: COCO's racket images are almost all
# strung tennis rackets, so pickleball paddles (solid face) detect
# less reliably. It's used as an extra signal alongside ball tracking
# and wrist speed for stroke confirmation, not the only check, mostly
# because of that gap.
#
# Needs two files that aren't bundled here (they're ~65MB, can't ship
# binaries through chat) — frozen_inference_graph.pb and
# ssd_mobilenet_v2_coco.pbtxt, both from the OpenCV DNN model zoo:
# https://github.com/opencv/opencv_extra/tree/master/testdata/dnn
# Drop them in models/. If they're missing, .available just stays
# False and nothing else in the program is affected.

import os
import cv2
import numpy as np
from typing import List, Tuple

MODEL_DIR      = "models"
GRAPH_PATH     = os.path.join(MODEL_DIR, "frozen_inference_graph.pb")
CONFIG_PATH    = os.path.join(MODEL_DIR, "ssd_mobilenet_v2_coco.pbtxt")

COCO_TENNIS_RACKET = 43
COCO_SPORTS_BALL   = 37   # ball_tracker.py already handles balls via colour,
                          # kept here for reference

CONF_THRESHOLD = 0.35     # rackets are thin/small, so threshold is lower
                          # than a typical detector to avoid missing them
INPUT_SIZE     = 300


class RacketDetector:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.available = False
        self.net       = None
        self._graph    = os.path.join(model_dir, "frozen_inference_graph.pb")
        self._config   = os.path.join(model_dir, "ssd_mobilenet_v2_coco.pbtxt")

        self.detections: List[Tuple[int,int,int,int,float]] = []  # x1,y1,x2,y2,conf
        self._try_load()

    def _try_load(self) -> None:
        if not (os.path.exists(self._graph) and os.path.exists(self._config)):
            return   # silently unavailable until weights are downloaded
        try:
            self.net = cv2.dnn.readNetFromTensorflow(self._graph, self._config)
            self.available = True
        except Exception:
            self.available = False
            self.net = None

    def detect(self, frame: np.ndarray) -> List[Tuple[int,int,int,int,float]]:
        """
        Runs COCO detection, filters to the "tennis racket" class only.
        Returns list of (x1,y1,x2,y2,confidence) racket bounding boxes.
        """
        self.detections = []
        if not self.available or self.net is None:
            return self.detections

        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, size=(INPUT_SIZE, INPUT_SIZE),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        out = self.net.forward()

        for i in range(out.shape[2]):
            conf      = float(out[0, 0, i, 2])
            class_id  = int(out[0, 0, i, 1])
            if conf < CONF_THRESHOLD or class_id != COCO_TENNIS_RACKET:
                continue
            x1 = int(out[0, 0, i, 3] * w)
            y1 = int(out[0, 0, i, 4] * h)
            x2 = int(out[0, 0, i, 5] * w)
            y2 = int(out[0, 0, i, 6] * h)
            self.detections.append((x1, y1, x2, y2, conf))

        return self.detections

    def racket_near_wrist(self, wrist_x: float, wrist_y: float,
                          torso_px: float) -> float:
        """
        Returns the confidence of the nearest detected racket box to the
        given wrist position, or 0.0 if none are within 0.8 torso-lengths
        of the wrist. Used as an additional stroke-confirmation signal.
        """
        if not self.detections:
            return 0.0
        best_conf = 0.0
        pad = 0.8 * max(torso_px, 1.0)
        for (x1, y1, x2, y2, conf) in self.detections:
            cx, cy = (x1+x2)/2, (y1+y2)/2
            dist   = float(np.hypot(cx-wrist_x, cy-wrist_y))
            if dist < pad and conf > best_conf:
                best_conf = conf
        return best_conf

    def draw(self, img: np.ndarray) -> np.ndarray:
        """Draw detected racket boxes."""
        for (x1, y1, x2, y2, conf) in self.detections:
            cv2.rectangle(img, (x1,y1), (x2,y2), (0,180,255), 2, cv2.LINE_AA)
            cv2.putText(img, f"racket {conf:.2f}", (x1, max(y1-6,12)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,180,255), 1, cv2.LINE_AA)
        return img