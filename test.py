"""
test.py - DEBUG runner. Shows the inference overlay window.
Uses the exact same perception.main() as main.py.

Keys (in the video window):
    I -> type a new target in the terminal
    G -> grab (freeze original_pos)
    R -> reset grab
    Q / ESC -> quit
"""

import os
os.environ.setdefault("OB_LOG_LEVEL", "none")

import threading
import time

import cv2

import perception
from main import GeminiCamera

DEBUG = True                      # False -> no window, JSON only (same as main.py)
PRINT_JSON = True
WINDOW_NAME = "Gemini E - YOLO Tracking + 3D (DEBUG)"

target = None
grab = 0
_pending = {"value": None}
_asking = threading.Event()


def _ask_target():
    if _asking.is_set():
        return
    _asking.set()

    def worker():
        try:
            print("\nEnter object to track: ", end="", flush=True)
            _pending["value"] = input().strip().lower()
        except (EOFError, KeyboardInterrupt):
            pass
        finally:
            _asking.clear()

    threading.Thread(target=worker, daemon=True).start()


def run():
    global target, grab
    cam = GeminiCamera()
    cam.start()
    if target is None:
        _ask_target()

    if DEBUG:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    last, fps, prev = None, 0.0, time.perf_counter()
    try:
        while True:
            frames = cam.read()
            if frames is None:
                continue

            if _pending["value"]:
                target, grab = _pending["value"], 0
                _pending["value"] = None
                print(f"\nNow tracking: {target}")

            out, dbg = perception.main(frames, target_object=target, grab_state=grab,
                                       depth_profile=cam.depth_profile,
                                       return_debug=True)
            if PRINT_JSON and out != last:
                print(out, flush=True)
                last = out

            now = time.perf_counter()
            dt, prev = now - prev, now
            if dt > 0:
                fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt

            if not DEBUG:
                continue

            cv2.imshow(WINDOW_NAME, perception.draw_debug(dbg, fps))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("i"):
                _ask_target()
            elif key == ord("g"):
                grab = 1
            elif key == ord("r"):
                grab = 0
            elif key in (ord("q"), 27):
                break
    except KeyboardInterrupt:
        pass
    finally:
        cv2.destroyAllWindows()
        cam.stop()


if __name__ == "__main__":
    run()