# ARAIP Project Part 2 – e-puck colour search + CIFAR-10 CNN (Webots)

5FTC2068 Applied Robotics and AI Project, Assignments 3/4/5 – Aisur Omar
Webots R2025a · Python 3 · e-puck

## Folder layout
```
worlds/colour_search_cnn.wbt     red/blue/green block search (demo for the presentation)
worlds/cnn_camera_test.wbt       cat in front + dog behind the e-puck, for the CNN camera test
controllers/colour_search_cnn/   state-machine controller + cnn_module.py (TFLite, background thread)
controllers/cnn_camera_test/     captures front/rear camera frames and classifies them
colab/ARAIP_Part2_CIFAR10_CNN.ipynb   CNN training, accuracy, own images, Webots image, TFLite optimisation
models/                          put cifar10_cnn_int8.tflite here (exported from Colab)
captures/                        camera PNGs saved by the controllers (created automatically)
```

## How to run
1. Open `worlds/colour_search_cnn.wbt` in Webots and press Run. The console prints
   `I see red/blue/green` on the first encounter of each block, then a summary.
2. Open `worlds/cnn_camera_test.wbt` and press Run. Two frames are saved to `captures/`.
   Upload them in section 7 of the Colab notebook to get the CNN predictions.
3. Optional on-robot inference: copy `cifar10_cnn_int8.tflite` from Colab into `models/` and
   `pip install numpy tflite-runtime` (or `tensorflow`) for the Python used by Webots.
   The controller then prints `[CNN] <class> (<confidence>)` lines without slowing the robot down.

## Environment / hardware notes
- Blocks: `Solid` + `Box 0.3 x 0.3 x 0.1 m`, boundingObject = same box (IR sensors detect them),
  PBRAppearance roughness 1 / metalness 0 (no reflections), positions generated with seed 2068.
- Plain light-grey sky (Background node) so the sky never looks red. Floor is a black and white chequered texture (worlds/textures/checker.png, 0.25 m squares): black is below the brightness limit and white has no saturation, so it never matches a colour range.
- e-puck: 2 wheel motors (max 6.28 rad/s), IR sensors ps0–ps7 (threshold 80),
  LEDs led0–led9, camera at its default 52×39 px.

## Colour detection
HSV thresholds (hue: red 0–15°/345–360°, green 90–150°, blue 200–260°; saturation ≥ 0.60,
value ≥ 0.20). Only every 2nd pixel is checked (pre-filter). A block counts as found when its
colour fills ≥ 25 % of the image for 3 frames in a row.

## Controller states
SCAN (360° look-around) → WANDER (random walk) → APPROACH (visual servoing) → AVOID (IR) →
REPORT (message + summary + LEDs) → BACKOFF → … → DONE when all three are found.
