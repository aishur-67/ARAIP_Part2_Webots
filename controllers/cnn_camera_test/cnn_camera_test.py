"""
cnn_camera_test.py - CNN prediction on images from the e-puck camera
5FTC2068 Applied Robotics and AI Project - Assignment 3/4/5 (Project Part 2)
Author: Aisur Omar

World: worlds/cnn_camera_test.wbt (a cat in front of the robot and a dog
behind it). The robot:
  1. waits 1 s for the scene to settle, captures the front view (cat),
  2. turns 180 degrees, captures the rear object (dog),
  3. classifies both frames with the CIFAR-10 CNN (TFLite, background
     thread) if a model is present in ../../models/,
  4. saves every frame to captures/ so it can also be classified in Colab.
"""

import math
import os
import sys

from controller import Robot

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "..", "colour_search_cnn"))
from cnn_module import CNNClassifier, save_snapshot  # noqa: E402

TIME_STEP = 64
WHEEL_RADIUS = 0.0205      # m (e-puck)
AXLE_LENGTH = 0.052        # m (e-puck)
TURN_SPEED = 2.0           # rad/s wheel speed while turning


def main():
    robot = Robot()
    left = robot.getDevice("left wheel motor")
    right = robot.getDevice("right wheel motor")
    for m in (left, right):
        m.setPosition(float("inf"))
        m.setVelocity(0.0)
    camera = robot.getDevice("camera")
    camera.enable(TIME_STEP)
    w, h = camera.getWidth(), camera.getHeight()
    cnn = CNNClassifier(period=0.0)
    print("[init] camera %dx%d | %s" % (w, h, cnn.status))

    # time needed to spin 180 deg on the spot
    robot_rate = 2 * WHEEL_RADIUS * TURN_SPEED / AXLE_LENGTH   # rad/s
    turn_time = math.pi / robot_rate

    plan = [("front_object", 1.0), ("turn", turn_time), ("settle", 0.5),
            ("rear_object", 0.5), ("done", 0.0)]
    phase, t_phase = 0, robot.getTime()

    while robot.step(TIME_STEP) != -1:
        name, dur = plan[phase]
        t = robot.getTime()

        res = cnn.pop_result()
        if res:
            print("[CNN] prediction: %s  (confidence %.0f%%, %.1f ms)" %
                  (res[0], res[1] * 100, res[2]))

        if name == "turn":
            left.setVelocity(TURN_SPEED)
            right.setVelocity(-TURN_SPEED)
        else:
            left.setVelocity(0.0)
            right.setVelocity(0.0)

        if t - t_phase < dur:
            continue

        if name in ("front_object", "rear_object"):
            path = save_snapshot(camera, name, t)
            print("[capture] %s -> %s" % (name, path))
            cnn.submit(camera.getImage(), w, h, t)
        if name == "done":
            if not cnn.busy:
                print("Camera test finished. Images are in the captures/ folder.")
                break
            continue
        phase += 1
        t_phase = t


if __name__ == "__main__":
    main()
