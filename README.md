# CourtSense 🎾

Real-time human motion tracking and sports analytics using a single camera. No wearables or sensors required.

Runs in two independent modes:
* **Warmup Mode** – Rep-counting workout tracker
* **Analytics Mode** – Tennis/pickleball performance analysis

> **Note:** Warmup and Analytics cannot run simultaneously.

---

## 🏋️‍♂️ Warmup Mode
Counts squats, pushups, and curls using joint-angle tracking.

### Features
* **Noise-Tolerant Detection:** Works with fast or slow movement.
* **Rep Quality Bar:** Scores each rep on depth and tempo.
* **Hand-Gesture Switching:** Hold up 1/2/3 fingers or bring palms together.
* **Camera Tuning:** Confidence threshold, smoothing presets, tracking lock.

---

## 📊 Analytics Mode
Tracks up to three players for tennis and pickleball.

### Features
* **Stroke Detection:** Forehand/backhand classification, validated with ball + racket tracking.
* **Footwork Tracking:** Split-steps, lunges (with direction), jumps; includes per-minute rates.
* **Recovery Timing:** Measures time to return to ready stance after each shot.
* **Rally Counting:** Current rally and longest rally.
* **Stamina Trend:** Rolling intensity compared to early-session baseline.
* **Ready-Stance Coaching:** Live feedback on knee bend and body lean.
* **Court Heatmaps:** Camera-perspective and calibrated bird’s-eye view.
* **Spoken Coaching Cues:** Offline text-to-speech.
* **Session Recording:** Saves annotated video.
* **Session History:** Logs every session; `review.py` shows long-term trends.

> **Note:** All speed/distance metrics are normalized to torso-lengths per second (not raw pixels).

---

## 🧠 How It Works
* **Pose & Hand Tracking:** MediaPipe Pose Landmarker + Hand Landmarker
* **Ball Detection:** HSV color masking
* **Racket Detection:** SSD MobileNet v2 (COCO) via OpenCV DNN
* **Court Calibration:** Click 4 corners → homography → real-world coordinates
* **Threaded Inference:** Pose detection runs on a background thread
* **Confidence Gating:** Suppresses events when tracking quality drops

---

Note: ReadME was polished using AI

## 🚀 Setup
```bash
# Create and activate virtual environment
python -m venv venv
venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
