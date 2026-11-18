"""
colour_search_cnn.py - Webots controller for the e-puck robot
5FTC2068 Applied Robotics and AI Project - Assignment 3/4/5 (Project Part 2)
Author: Aisur Omar

What it does
------------
1. Searches the arena for a red, a blue and a green block (0.3 x 0.3 x 0.1 m).
   On the FIRST encounter of each block it prints "I see red/blue/green",
   followed by a summary of everything found so far.
2. Saves camera frames and (if a TFLite model is in ../../models/) runs the
   CIFAR-10 CNN on them in a background thread, so the CNN never slows the
   robot's 64 ms control loop.

Design (state machine)
----------------------
  SCAN     - spin on the spot for one full turn to look around (cheap way to
             find colours before driving anywhere)
  WANDER   - drive forward, random turn every few seconds (coverage)
  APPROACH - visual servoing towards the largest un-found colour
  AVOID    - IR proximity sensors steer away from walls
  REPORT   - print message + summary, blink LEDs
  BACKOFF  - reverse and turn away from the block just reported
  DONE     - all three colours found, stop

Hardware notes (e-puck): 2 stepper-motor wheels (max 6.28 rad/s),
8 IR proximity sensors ps0..ps7, 10 LEDs led0..led9, front colour camera
(default 52 x 39 px; the CNN camera-test world raises it to 160 x 120).

Keyboard (click the 3D view first):  C = save camera frame now
"""

import math
import random

from controller import Robot, Keyboard

from cnn_module import CNNClassifier, save_snapshot

# --------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------
TIME_STEP = 64                 # ms
MAX_SPEED = 6.28               # rad/s
CRUISE = 0.6 * MAX_SPEED

# HSV colour ranges (hue in degrees, sat/val in 0..1). HSV separates colour
# from brightness, so the shaded side of a block is still classified right.
HSV_RANGES = {
    "red":   [(0, 15), (345, 360)],
    "green": [(90, 150)],
    "blue":  [(200, 260)],
}
MIN_SAT = 0.60                 # grey walls, sky and floor have low saturation
MIN_VAL = 0.20                 # ignore very dark pixels (black floor tiles)
PIXEL_STRIDE = 2               # camera pre-filter: check every 2nd pixel

SEEN_RATIO = 0.01              # colour visible -> approach it
FOUND_RATIO = 0.25             # colour fills 25 % of the view -> found (close to it)
CONFIRM_FRAMES = 3             # consecutive frames needed (noise rejection)
AVOID_OFF_RATIO = 0.10         # target this big -> ignore IR (it IS the block)
OBSTACLE_THRESHOLD = 80.0

SCAN_STEPS = 45                # about one full turn on the spot
SCAN_EVERY = 12.0              # seconds of wandering between scans
SAVE_EVERY = 5.0               # seconds between automatic camera snapshots
COLOURS = ("red", "blue", "green")


# --------------------------------------------------------------------------
# Perception
# --------------------------------------------------------------------------
def rgb_to_hsv(r, g, b):
    """r,g,b in 0..255 -> (hue deg, sat 0..1, val 0..1)."""
    mx, mn = max(r, g, b), min(r, g, b)
    v = mx / 255.0
    if mx == 0:
        return 0.0, 0.0, 0.0
    d = mx - mn
    s = d / mx
    if d == 0:
        return 0.0, s, v
    if mx == r:
        h = 60.0 * (((g - b) / d) % 6)
    elif mx == g:
        h = 60.0 * (((b - r) / d) + 2)
    else:
        h = 60.0 * (((r - g) / d) + 4)
    return h, s, v


def classify_pixel(r, g, b):
    h, s, v = rgb_to_hsv(r, g, b)
    if s < MIN_SAT or v < MIN_VAL:
        return None
    for name, ranges in HSV_RANGES.items():
        for lo, hi in ranges:
            if lo <= h < hi:
                return name
    return None


def detect_colour_blocks(image, width, height):
    """Return {colour: (ratio_of_image, horizontal_centre_0_to_1)}."""
    counts = {c: 0 for c in COLOURS}
    xsum = {c: 0 for c in COLOURS}
    total = 0
    for y in range(0, height, PIXEL_STRIDE):
        row = y * width * 4
        for x in range(0, width, PIXEL_STRIDE):
            i = row + 4 * x
            c = classify_pixel(image[i + 2], image[i + 1], image[i])  # BGRA
            total += 1
            if c:
                counts[c] += 1
                xsum[c] += x
    out = {}
    for c in COLOURS:
        ratio = counts[c] / float(total)
        centre = (xsum[c] / counts[c]) / float(width) if counts[c] else 0.5
        out[c] = (ratio, centre)
    return out


# --------------------------------------------------------------------------
# Robot wrapper
# --------------------------------------------------------------------------
class EpuckColourSearcher:
    def __init__(self):
        self.robot = Robot()
        self.left = self.robot.getDevice("left wheel motor")
        self.right = self.robot.getDevice("right wheel motor")
        for m in (self.left, self.right):
            m.setPosition(float("inf"))
            m.setVelocity(0.0)
        self.ps = []
        for i in range(8):
            s = self.robot.getDevice("ps%d" % i)
            s.enable(TIME_STEP)
            self.ps.append(s)
        self.camera = self.robot.getDevice("camera")
        self.camera.enable(TIME_STEP)
        self.cam_w = self.camera.getWidth()
        self.cam_h = self.camera.getHeight()
        self.leds = []
        for i in range(10):
            try:
                self.leds.append(self.robot.getDevice("led%d" % i))
            except Exception:
                pass
        self.keyboard = Keyboard()
        self.keyboard.enable(TIME_STEP)

        self.found = {c: None for c in COLOURS}     # colour -> time found
        self.order = []
        self.confirm = {c: 0 for c in COLOURS}
        self.state = "SCAN"
        self.state_steps = 0
        self.turn_dir = 1
        self.next_turn = self._t() + random.uniform(2.5, 6.0)
        self.next_scan = self._t() + SCAN_EVERY
        self.next_save = 0.0
        self.cnn = CNNClassifier(period=1.0)
        print("[init] camera %dx%d | %s" % (self.cam_w, self.cam_h, self.cnn.status))

    # -- helpers -----------------------------------------------------------
    def _t(self):
        return self.robot.getTime()

    def set_speed(self, l, r):
        self.left.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, l)))
        self.right.setVelocity(max(-MAX_SPEED, min(MAX_SPEED, r)))

    def obstacles(self):
        v = [s.getValue() for s in self.ps]
        right = max(v[0], v[1], v[2]) > OBSTACLE_THRESHOLD
        left = max(v[5], v[6], v[7]) > OBSTACLE_THRESHOLD
        return left, right

    def blink(self, on):
        for led in self.leds:
            led.set(1 if on else 0)

    def change_state(self, new):
        self.state, self.state_steps = new, 0

    # -- reporting ---------------------------------------------------------
    def report(self, colour):
        self.found[colour] = self._t()
        self.order.append(colour)
        print("I see %s" % colour)
        self.print_summary()
        save_snapshot(self.camera, "block_%s" % colour, self._t())

    def print_summary(self):
        print("------ Summary of blocks encountered ------")
        for c in COLOURS:
            t = self.found[c]
            print("  %-5s : %s" % (c, "FOUND at %.1f s" % t if t is not None
                                   else "not yet found"))
        print("  Total : %d / 3   Order: %s" %
              (len(self.order), " -> ".join(self.order) or "-"))
        print("-------------------------------------------")

    # -- CNN ---------------------------------------------------------------
    def cnn_step(self, image, colours_visible):
        t = self._t()
        # Pre-filter: only classify frames that are NOT a plain colour block
        # (blocks are handled by the cheap HSV detector).
        if not colours_visible:
            self.cnn.submit(image, self.cam_w, self.cam_h, t)
        res = self.cnn.pop_result()
        if res:
            label, conf, ms = res
            print("[CNN] %s (%.0f%%)  inference %.1f ms" % (label, conf * 100, ms))
        if t >= self.next_save:
            save_snapshot(self.camera, "frame", t)
            self.next_save = t + SAVE_EVERY
        if self.keyboard.getKey() in (ord("C"), ord("c")):
            print("[capture] saved", save_snapshot(self.camera, "manual", t))

    # -- main loop ---------------------------------------------------------
    def run(self):
        while self.robot.step(TIME_STEP) != -1:
            self.state_steps += 1
            image = self.camera.getImage()
            seen = detect_colour_blocks(image, self.cam_w, self.cam_h)
            visible = any(seen[c][0] >= SEEN_RATIO for c in COLOURS)
            self.cnn_step(image, visible)

            # 1) REPORT (highest priority)
            for c in COLOURS:
                if self.found[c] is not None:
                    continue
                self.confirm[c] = self.confirm[c] + 1 if seen[c][0] >= FOUND_RATIO else 0
                if self.confirm[c] >= CONFIRM_FRAMES:
                    self.report(c)
                    self.change_state("BACKOFF")
                    self.turn_dir = random.choice((-1, 1))
            if len(self.order) == 3:
                self.set_speed(0, 0)
                self.blink(True)
                print("Search complete - all three blocks found in %.1f s" % self._t())
                break

            # 2) BACKOFF after a report
            if self.state == "BACKOFF":
                self.blink(self.state_steps % 4 < 2)
                if self.state_steps < 12:
                    self.set_speed(-CRUISE, -CRUISE)
                elif self.state_steps < 24:
                    self.set_speed(self.turn_dir * CRUISE, -self.turn_dir * CRUISE)
                else:
                    self.blink(False)
                    self.change_state("SCAN")
                continue

            # 3) APPROACH an un-found colour
            target = max((c for c in COLOURS if self.found[c] is None),
                         key=lambda c: seen[c][0])
            ratio, centre = seen[target]
            left_obs, right_obs = self.obstacles()
            obstacle = left_obs or right_obs
            if ratio >= SEEN_RATIO and (not obstacle or ratio >= AVOID_OFF_RATIO):
                err = centre - 0.5
                base = CRUISE * (0.5 if ratio > 0.12 else 1.0)  # slow down near it
                self.set_speed(base + err * 2 * MAX_SPEED, base - err * 2 * MAX_SPEED)
                if self.state != "APPROACH":
                    self.change_state("APPROACH")
                continue

            # 4) AVOID walls
            if obstacle:
                if left_obs and right_obs:
                    self.set_speed(CRUISE, -CRUISE)
                elif left_obs:
                    self.set_speed(CRUISE, -CRUISE * 0.3)
                else:
                    self.set_speed(-CRUISE * 0.3, CRUISE)
                continue

            # 5) SCAN: one turn on the spot to look around
            if self.state == "SCAN":
                self.set_speed(0.5 * CRUISE, -0.5 * CRUISE)
                if self.state_steps >= SCAN_STEPS:
                    self.change_state("WANDER")
                    self.next_scan = self._t() + SCAN_EVERY
                continue

            # 6) WANDER (lowest priority)
            if self.state != "WANDER":
                self.change_state("WANDER")
            if self._t() >= self.next_scan:
                self.change_state("SCAN")
                continue
            if self._t() >= self.next_turn:
                self.turn_dir = random.choice((-1, 1))
                self.next_turn = self._t() + random.uniform(2.5, 6.0)
                self.turn_until = self._t() + random.uniform(0.4, 1.2)
            if getattr(self, "turn_until", 0) > self._t():
                self.set_speed(self.turn_dir * CRUISE, -self.turn_dir * CRUISE)
            else:
                self.set_speed(CRUISE, CRUISE)


if __name__ == "__main__":
    EpuckColourSearcher().run()
