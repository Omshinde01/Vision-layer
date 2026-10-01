"""
main.py - feeds camera frames into perception.main() and prints the JSON.
Headless by default; --debug adds the overlay window, FPS and diagnostics.
All debug / visualization / logging / keyboard code lives here, not in perception.py.

Run:
    python main.py --target cup
    python main.py --target cup --debug

Type commands in the terminal while it runs:
    t <name>   change target object   (e.g.  t bottle)
    g          grab  (freeze original_pos and placement footprint)
    r          release / reset grab
    q          quit
"""

import os
os.environ.setdefault("OB_LOG_LEVEL", "none")

import argparse
import sys
import threading
import time

import cv2
import numpy as np
from pyorbbecsdk import Pipeline, Config, OBSensorType

import perception

# ---- Debug overlay toggles (drawing only, never affect JSON or placement) ----
DRAW_ALL_OBSTACLE_BOXES = True      # every overlapping obstacle, blocking (red) or safe (green)
DRAW_PLACEMENT_BBOX = True          # placement bbox(es) evaluated
DRAW_FOOTPRINT = True               # free-space footprint (incl. buffer) checked on the surface
DRAW_BASE_POINT = True              # target base point (bottom-centre)

FONT = cv2.FONT_HERSHEY_SIMPLEX
WINDOW_NAME = "main.py DEBUG - YOLO Tracking + 3D"
_UNIT = perception.MEASUREMENT_UNIT
_ND = {"m": 3, "cm": 1, "mm": 0}[_UNIT]   # display decimals only


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
        """Return raw FrameSet, or None on timeout."""
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

    def set_grab(self, value):
        with self._lock:
            self.grab = 1 if int(value) else 0

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
                    # Changing target also releases any previous grab state.
                    self.target = arg.strip().lower()
                    self.grab = 0

                elif cmd == "g":
                    self.grab = 1

                elif cmd == "r":
                    self.grab = 0

                elif cmd == "q":
                    self.quit = True
                    return


# ------------------------ DEBUG OVERLAY (optional) --------------------
def _txt(img, text, pos, color=(255, 255, 255)):
    cv2.putText(img, text, pos, FONT, 0.45, color, 1, cv2.LINE_AA)


def draw_debug(dbg, fps=None):
    """Annotated BGR image from the details dict of perception.main(return_details=True)."""
    image = dbg.get("color")
    out = image.copy() if image is not None else np.zeros((480, 640, 3), np.uint8)
    payload = dbg["payload"]
    sel = dbg.get("selected")
    if sel and not sel.get("depth_invalid"):
        x1, y1, x2, y2 = sel["bbox"]
        cx, cy = sel["center_px"]
        pl = sel["placement"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        _txt(out, f'{payload["object"]["object_name"]} ID:{sel["track_id"]} '
                  f'{sel["confidence"]:.2f}', (x1, max(15, y1 - 6)), (0, 255, 0))
        cv2.drawMarker(out, (cx, cy), (255, 0, 0), cv2.MARKER_CROSS, 12, 2)
        cv2.drawMarker(out, (x1, cy), (255, 255, 0), cv2.MARKER_CROSS, 12, 2)
        cv2.drawMarker(out, (x2, cy), (0, 165, 255), cv2.MARKER_CROSS, 12, 2)
        if DRAW_ALL_OBSTACLE_BOXES:
            for ob in pl.get("all_obstacles", []):
                ox1, oy1, ox2, oy2 = ob["bbox"]
                col = (0, 0, 255) if ob["blocking"] else (0, 200, 0)
                cv2.rectangle(out, (ox1, oy1), (ox2, oy2), col, 2)
                dd = ob.get("depth_diff")
                dtxt = f" dZ={dd:.{_ND}f}{_UNIT}" if dd is not None else " dZ=?"
                _txt(out, f'{ob["class_name"]} {ob.get("slot", "")} '
                          f'{"BLOCK" if ob["blocking"] else "SAFE"}{dtxt}',
                     (ox1, max(15, oy1 - 6)), col)
        else:
            for key, tag in (("obstacle20", "+20"), ("obstacle40", "+40")):
                ob = pl.get(key)
                if ob:
                    ox1, oy1, ox2, oy2 = ob["bbox"]
                    cv2.rectangle(out, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                    _txt(out, f'{ob["class_name"]} {tag}', (ox1, max(15, oy1 - 6)), (0, 0, 255))
        if DRAW_PLACEMENT_BBOX:
            for key in ("bbox20", "bbox40"):
                if pl.get(key) is not None:
                    px1, py1, px2, py2 = pl[key]
                    cv2.rectangle(out, (px1, py1), (px2, py2), (160, 0, 160), 1)
        if DRAW_FOOTPRINT:
            for key in ("footprint20", "footprint40", "footprint"):
                if pl.get(key) is not None:
                    fx1, fy1, fx2, fy2 = pl[key]
                    cv2.rectangle(out, (fx1, fy1), (fx2, fy2), (255, 200, 0), 1)
        if DRAW_BASE_POINT and sel.get("base") is not None:
            cv2.drawMarker(out, (cx, y2), (0, 255, 255), cv2.MARKER_DIAMOND, 10, 2)
        if pl["pixel"] is not None:
            cv2.drawMarker(out, pl["pixel"], (0, 0, 0), cv2.MARKER_CROSS, 22, 2)
            _txt(out, "Place loc", (pl["pixel"][0] + 8, pl["pixel"][1] - 8), (0, 0, 0))
        _txt(out, pl["message"], (x1, max(15, y1 - 24)), (0, 255, 255))
        h = out.shape[0]
        y = min(h - 76, y2 + 18)
        for i, (tag, v, c) in enumerate([("C", sel["center"], (255, 0, 0)),
                                         ("L", sel["left"], (255, 255, 0)),
                                         ("R", sel["right"], (0, 165, 255)),
                                         ("P", sel["target"], (0, 0, 0))]):
            if v is not None:
                _txt(out, f"{tag} {v[0]:+.{_ND}f},{v[1]:+.{_ND}f},{v[2]:.{_ND}f} {_UNIT}",
                     (x1, y + 17 * i), c)
    elif sel and sel.get("depth_invalid"):
        x1, y1, x2, y2 = sel["bbox"]
        _txt(out, "Depth: INVALID", (x1, y2 + 18), (0, 0, 255))

    if dbg.get("locked_id") is not None and dbg.get("missing"):
        _txt(out, f'Locked ID:{dbg["locked_id"]} not visible '
                  f'({dbg["missing"]}/{perception.REACQUIRE_AFTER_MISSING_FRAMES})',
             (8, 72), (0, 165, 255))
    _txt(out, f'Target: {dbg["target"]}' if dbg.get("target") else "No target selected", (8, 20))
    _txt(out, f'Grab: {dbg.get("grab", 0)}', (8, 38))
    if fps is not None:
        _txt(out, f"FPS: {fps:.1f}", (8, 56))
    _txt(out, "G: grab | R: reset | Q: quit", (8, 90))
    return out


# -------------------------------- MAIN --------------------------------
def _report_engine_state(shown, announced_k):
    """Print new engine warnings (stderr); in debug also the loaded intrinsics once."""
    eng = perception.get_engine()
    if eng is None:
        return shown, announced_k
    for msg in eng.warnings[shown:]:
        print(f"[perception] {msg}", file=sys.stderr, flush=True)
    return len(eng.warnings), announced_k


def run(target=None, debug=False):
    controls = Controls(target)
    controls.start()

    cam = GeminiCamera()
    cam.start()

    print(
        "Running. Commands: t <name> | g | r | q"
        + ("  (window keys: I / G / R / Q)" if debug else ""),
        flush=True,
    )

    if debug:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    last = None
    fps = 0.0
    prev = time.perf_counter()
    shown_warnings = 0
    k_announced = False

    try:
        while True:
            tgt, grab, quit_ = controls.snapshot()

            if quit_:
                break

            frames = cam.read()
            if frames is None:
                continue

            # RAW frames -> perception.py (all detection, depth, 3D, tracking and
            # placement logic stays there).
            res = perception.main(
                frames,
                target_object=tgt,
                grab_state=grab,
                depth_profile=cam.depth_profile,
                return_details=debug,
            )

            out, dbg = res if debug else (res, None)

            shown_warnings, _ = _report_engine_state(shown_warnings, k_announced)
            if debug and not k_announced:
                K = perception.get_engine().intrinsics
                if K is not None:
                    print("Depth intrinsics: FX={:.2f}, FY={:.2f}, CX={:.2f}, CY={:.2f}"
                          .format(*K), file=sys.stderr, flush=True)
                k_announced = True

            # Avoid printing identical JSON repeatedly.
            if out != last:
                print(out, flush=True)
                last = out

            if not debug:
                continue

            now = time.perf_counter()
            dt, prev = now - prev, now

            if dt > 0:
                fps = 1.0 / dt if fps == 0 else 0.9 * fps + 0.1 / dt

            cv2.imshow(WINDOW_NAME, draw_debug(dbg, fps))

            key = cv2.waitKey(1) & 0xFF

            if key == ord("i"):
                print("\nType in terminal:  t <object name>", flush=True)

            elif key == ord("g"):
                controls.set_grab(1)

            elif key == ord("r"):
                controls.set_grab(0)

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

    ap.add_argument(
        "--target",
        default=None,
        help="object class to track",
    )

    ap.add_argument(
        "--debug",
        action="store_true",
        help="show inference overlay window "
             "(boxes, XYZ, placement, free-space footprint, FPS)",
    )

    args = ap.parse_args()
    run(args.target, args.debug)