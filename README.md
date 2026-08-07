# CourtSense

CourtSense is a single-camera setup for tracking human motion and generating sports analytics. No wearables or sensors on the players — just a webcam.

There are two main modes, though you can only run one at a time.

## Warmup Mode

This is a workout tracker that counts your reps. It uses joint angles to track squats, pushups, and bicep curls, and handles different movement speeds without breaking. There's a progress bar that grades your reps based on depth and tempo.

You don't have to touch the keyboard to switch exercises — hold up one, two, or three fingers, or put your palms together. You can also tweak the camera settings for smoothing, tracking lock, and confidence thresholds.

## Analytics Mode

This mode is built for tennis and pickleball, and tracks up to three people on the court.

By tracking both the racket and the ball, it can classify whether you hit a forehand or a backhand. It also logs your footwork — jumps, split-steps, directional lunges — and calculates how often you do them per minute.

It also measures recovery timing (how fast you get back to a ready stance after a shot), rally counts (current streak and longest of the session), and stamina (comparing your current movement intensity against your baseline from the start of the session). There's live stance feedback too, checking knee bend and body lean.

Speed and distance are measured in torso-lengths per second instead of raw pixels, since that stays consistent regardless of how far you are from the camera.

The tool generates court heatmaps from both the standard camera angle and a calibrated top-down view, and uses offline text-to-speech for coaching cues. Sessions save as annotated video files plus data logs — run `review.py` later to check your long-term trends.

## How It Works

- MediaPipe handles the hand and pose landmarks.
- OpenCV DNN runs an SSD MobileNet v2 model to detect the racket.
- Ball tracking uses basic HSV color masking.
- To calibrate the court, you click the four corners on screen. A homography matrix translates those pixels into real-world coordinates.
- Pose inference runs on a background thread so it doesn't bottleneck the rest of the program.
- If tracking confidence drops too low, the system just ignores those events instead of letting bad data mess up your stats.

## Setup

Create a virtual environment and activate it (Windows):

```bash
python -m venv venv
venv\Scripts\activate
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```
