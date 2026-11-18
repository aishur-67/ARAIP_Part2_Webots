"""
cnn_module.py - CIFAR-10 CNN inference for the Webots e-puck
5FTC2068 Applied Robotics and AI Project - Assignment 3/4/5 (Project Part 2)

The CNN is trained off-board in Google Colab and exported as a quantised
TensorFlow Lite file (models/cifar10_cnn_int8.tflite or the float16 one).

Optimisation ideas used here so the CNN does not slow the robot down:
  1. Off-board training, on-board inference only.
  2. A small quantised TFLite model (about 4x smaller than the Keras model).
  3. Inference runs in a background thread, so the control loop (64 ms)
     never waits for the CNN.
  4. Rate limiting: the CNN runs at most once every CNN_PERIOD seconds.
  5. Pre-filter: only frames that pass a cheap check (enough non-floor
     pixels / object present) are sent to the CNN.
If no TFLite runtime is installed, the module still saves camera frames
so they can be classified in Colab instead.
"""

import os
import threading
import time

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(HERE, "..", ".."))
MODEL_CANDIDATES = [
    os.path.join(PROJECT_DIR, "models", "cifar10_cnn_int8.tflite"),
    os.path.join(PROJECT_DIR, "models", "cifar10_cnn_fp16.tflite"),
    os.path.join(PROJECT_DIR, "models", "cifar10_cnn.tflite"),
]
CAPTURE_DIR = os.path.join(PROJECT_DIR, "captures")

try:
    import numpy as np
except ImportError:  # numpy is needed for inference only
    np = None


def _load_interpreter():
    """Return (interpreter, path) or (None, reason)."""
    if np is None:
        return None, "numpy not installed"
    model_path = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
    if model_path is None:
        return None, "no .tflite model found in models/"
    Interpreter = None
    try:
        from tflite_runtime.interpreter import Interpreter  # light runtime
    except ImportError:
        try:
            import tensorflow as tf
            Interpreter = tf.lite.Interpreter
        except ImportError:
            return None, "tflite_runtime / tensorflow not installed"
    interp = Interpreter(model_path=model_path, num_threads=1)
    interp.allocate_tensors()
    return interp, model_path


def bgra_to_rgb_array(raw, width, height):
    """Webots camera bytes (BGRA) -> HxWx3 uint8 RGB numpy array."""
    img = np.frombuffer(raw, dtype=np.uint8).reshape((height, width, 4))
    return img[:, :, [2, 1, 0]].copy()


def preprocess(rgb):
    """Centre-crop to a square and resize to 32x32 (nearest neighbour,
    so no extra library such as OpenCV/PIL is required)."""
    h, w, _ = rgb.shape
    s = min(h, w)
    y0, x0 = (h - s) // 2, (w - s) // 2
    sq = rgb[y0:y0 + s, x0:x0 + s]
    idx = (np.arange(32) * s / 32).astype(int)
    return sq[idx][:, idx]


class CNNClassifier:
    """Non-blocking CIFAR-10 classifier for camera frames."""

    def __init__(self, period=1.0, min_confidence=0.40):
        self.period = period
        self.min_confidence = min_confidence
        self.interpreter, info = _load_interpreter()
        self.enabled = self.interpreter is not None
        self.status = ("CNN loaded: " + os.path.basename(info)) if self.enabled \
            else ("CNN disabled (" + info + ") - frames will only be saved")
        self.last_run = -1e9
        self.busy = False
        self.latest = None          # (label, confidence, inference_ms)
        self._lock = threading.Lock()
        os.makedirs(CAPTURE_DIR, exist_ok=True)

    def ready(self, sim_time):
        return (not self.busy) and (sim_time - self.last_run >= self.period)

    def submit(self, raw, width, height, sim_time):
        """Start a background inference on one frame (returns immediately)."""
        if not self.enabled or not self.ready(sim_time):
            return False
        self.busy = True
        self.last_run = sim_time
        frame = bytes(raw)          # copy: Webots reuses the buffer
        threading.Thread(target=self._infer, args=(frame, width, height),
                         daemon=True).start()
        return True

    def _infer(self, raw, width, height):
        try:
            x = preprocess(bgra_to_rgb_array(raw, width, height))
            inp = self.interpreter.get_input_details()[0]
            out = self.interpreter.get_output_details()[0]
            if inp["dtype"] == np.uint8 or inp["dtype"] == np.int8:
                scale, zero = inp["quantization"]
                xin = (x / 255.0 / scale + zero).astype(inp["dtype"])
            else:
                xin = (x / 255.0).astype(np.float32)
            t0 = time.perf_counter()
            self.interpreter.set_tensor(inp["index"], xin[np.newaxis, ...])
            self.interpreter.invoke()
            y = self.interpreter.get_tensor(out["index"])[0].astype(np.float32)
            ms = (time.perf_counter() - t0) * 1000.0
            if out["dtype"] in (np.uint8, np.int8):
                scale, zero = out["quantization"]
                y = (y - zero) * scale
            if y.min() < 0 or abs(y.sum() - 1.0) > 0.05:   # logits -> softmax
                e = np.exp(y - y.max())
                y = e / e.sum()
            k = int(np.argmax(y))
            with self._lock:
                self.latest = (CIFAR10_CLASSES[k], float(y[k]), ms)
        except Exception as exc:  # never crash the robot because of the CNN
            with self._lock:
                self.latest = ("error: %s" % exc, 0.0, 0.0)
        finally:
            self.busy = False

    def pop_result(self):
        with self._lock:
            r, self.latest = self.latest, None
        return r


def save_snapshot(camera, tag, sim_time):
    """Save the current camera frame as PNG (for Colab prediction)."""
    os.makedirs(CAPTURE_DIR, exist_ok=True)
    path = os.path.join(CAPTURE_DIR, "%s_t%06.1f.png" % (tag, sim_time))
    camera.saveImage(path, 100)
    return path
