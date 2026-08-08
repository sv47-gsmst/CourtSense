# CourtSense

CourtSense turns a regular webcam into a sports coach and workout tracker. You don't need any wearables or sensors stuck to your body. Just point a camera at yourself and it figures out what you are doing.

It has two main modes right now. They don't run at the same time yet, so you just pick one to start.

## Warmup Mode

This is basically a smart rep counter. It watches your joint angles while you do squats, pushups, or bicep curls. It doesn't matter if you go slow and controlled or fast and sloppy, it keeps up either way. There is a progress bar that grades each rep based on how deep you went and how consistent your tempo was, so you can't cheat and get free credit for half reps.

You never have to touch your keyboard while you work out. If you hold up 1, 2, or 3 fingers, or clap your hands together, it automatically switches exercises. If the camera starts losing track of you, there are settings you can tweak like smoothing and confidence thresholds to fix it.

## Analytics Mode

This mode is built for tennis and pickleball. It can track up to three people on the court at once.

Since it tracks the racket and the ball at the same time, it actually knows if you just hit a forehand or a backhand. It also watches your footwork the entire time and counts things like jumps, split steps, and lunges per minute.

While you play, it also tracks a bunch of other stuff:
* How fast you get back into a ready stance after a shot
* Your current rally streak and the longest one you hit all session
* A rough stamina estimate that compares how you are moving right now to your baseline from earlier
* Live feedback on your knee bend and body lean

Speed and distance are measured in torso lengths per second instead of raw pixels. Pixel distance changes depending on how far you stand from the camera, so doing it this way keeps the numbers accurate no matter what. It also generates a heatmap of everywhere you moved on the court and can read coaching tips out loud using text to speech. Every session gets saved with a video and a data log. You can run the review script later whenever you want to see how you are trending over time.

## How It Works

* MediaPipe handles the pose and hand tracking.
* Racket detection runs on an SSD MobileNet v2 model using OpenCV.
* Ball tracking is just basic HSV color masking. I kept it simple on purpose.
* For court calibration, you just click the four corners of the court on your screen. It builds a homography matrix out of those points to turn pixel coordinates into real world coordinates.
* The pose tracking runs on its own background thread so it doesn't lag the rest of the app.
* If the tracking confidence drops on a specific frame, it just throws that frame out so bad data doesn't ruin your stats.

## Setup

You will want to make a virtual environment first. If you are on Windows, run this:

```bash
python -m venv venv
venv\Scripts\activate
```

Then grab the dependencies:

```bash
pip install -r requirements.txt
```
Once that is done, just run the script (run.bat) and pick a mode.
Or run the following in your project location terminal. 
```bash
python main.py
```

Side note: If your laptop only has 8GB of RAM, it will still work. Having all three tracking systems running at once might make your fan sound like a jet engine, but mine does it all the time and it survives just fine.
