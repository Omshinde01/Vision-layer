"""
main.py - feeds camera frames into perception.main() and prints the JSON.
No window, no overlay. Runs headless.
It only sends RAW camera frames to perception.py and prints the JSON it returns.

Run:
    python main.py --target cup            # headless, JSON only
    python main.py --target cup --debug    # + overlay window + FPS

Type commands in the terminal while it runs:
    t <name>   change target object   (e.g.  t bottle)
    g          grab  (freeze original_pos)
    r          release / reset grab
    q          quit
"""

import os
os.environ.setdefault("OB_LOG_LEVEL", "none")

import argparse
import threading
import time

import cv2
from pyorbbecsdk import Pipeline, Config, OBSensorType

import perception


# ------------------------------ CAMERA -------------------------------
class GeminiCamera:
    """Only starts the camera and hands out RAW frames. No conversion here."""

    def __init__(self):
        self.pipeline = Pipeline()
        self.depth_profile = None

    def start(self):
        cfg = Config()
        dp = self.pipeline.get_stream_profile_list(OBSensorType.DEPTH_SENSOR)
        cp = self.pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
        if dp is None or dp.get_count() == 0:
            raise RuntimeError("No depth profiles found.")
        if cp is None or cp.get_count() == 0:
            raise RuntimeError("No color profiles found.")
        self.depth_profile = dp.get_default_video_stream_profile()
        cfg.enable_stream(self.depth_profile)
        cfg.enable_stream(cp.get_default_video_stream_profile())
        self.pipeline.start(cfg)

    def read(self):
        """Raw FrameSet (or None on timeout)."""
        return self.pipeline.wait_for_frames(1000)

    def stop(self):
        try:
            self.pipeline.stop()
        except Exception:
            pass


# ---------------------------- CONTROL INPUT ---------------------------
class Controls:
    """Terminal commands -> shared state (target, grab, quit)."""

    def __init__(self, target=None):
        self.target = target.strip().lower() if target else None
        self.grab = 0
        self.quit = False
        self._lock = threading.Lock()

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def snapshot(self):
        with self._lock:
            return self.target, self.grab, self.quit

    def _loop(self):
        while True:
            try:
                line = input().strip()
            except (EOFError, KeyboardInterrupt):
                return
            if not line:
                continue
            cmd, _, arg = line.partition(" ")
            cmd = cmd.lower()
            with self._lock:
                if cmd == "t" and arg:
                    self.target, self.grab = arg.strip().lower(), 0
                elif cmd == "g":
                    self.grab = 1
                elif cmd == "r":
                    self.grab = 0
                elif cmd == "q":
                    self.quit = True
                    return


# -------------------------------- MAIN --------------------------------
WINDOW_NAME = "main.py DEBUG - YOLO Tracking + 3D"


def run(target=None, debug=False):
    controls = Controls(target)
    controls.start()
    cam = GeminiCamera()
    cam.start()
    print("Running. Commands: t <name> | g | r | q"
          + ("  (window keys: I / G / R / Q)" if debug else ""), flush=True)

    if debug:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    last, fps, prev = None, 0.0, time.perf_counter()
    try:
        while True:
            tgt, grab, quit_ = controls.snapshot()
            if quit_:
                break
            frames = cam.read()
            if frames is None:
                continue

            # raw frames in -> JSON out (all processing is inside perception)
            res = perception.main(frames, target_object=tgt, grab_state=grab,
                                  depth_profile=cam.depth_profile,
                                  return_debug=debug)
            out, dbg = res if debug else (res, None)

            if out != last:          # avoid duplicate lines
                print(out, flush=True)
                last = out

            if not debug:
                continue

            now = time.perf_counter()
            dt, prev = now - prev, now
            if dt > 0:
                fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt

            cv2.imshow(WINDOW_NAME, perception.draw_debug(dbg, fps))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("i"):
                print("\nType in terminal:  t <object name>", flush=True)
            elif key == ord("g"):
                controls.grab = 1
            elif key == ord("r"):
                controls.grab = 0
            elif key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        if debug:
            cv2.destroyAllWindows()
        cam.stop()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=None, help="object class to track")
    ap.add_argument("--debug", action="store_true",
                    help="show inference overlay window (boxes, XYZ, placement, FPS)")
    args = ap.parse_args()
    run(args.target, args.debug)