"""
perception.py - standalone YOLO tracking + 3D placement.

ALL processing lives here: raw SDK frame -> numpy/BGR conversion, depth scale,
camera intrinsics, YOLO tracking, 3D math, placement, JSON.
The caller (main.py) only hands over the raw frames.

Usage:
    import perception
    json_str = perception.main(frames, target_object="cup", grab_state=0,
                               depth_profile=depth_profile)

Inputs
    frames        : raw Orbbec FrameSet from pipeline.wait_for_frames()
                    (None / missing color or depth is fine - last good frame is reused)
    target_object : class name to track (e.g. "cup"); None = nothing selected
    grab_state    : 0 = original_pos/target_pos follow the object,
                    1 = freeze original_pos and target_pos
    depth_profile : raw Orbbec depth stream profile (used once to read intrinsics);
                    optional, approximate intrinsics are used if omitted

Output
    JSON string (robot-facing nested JSON).

Coordinates: X = right, Y = down, Z = forward, meters.
"""

import json
import sys

import cv2
import numpy as np
from ultralytics import YOLO

# ========================= TUNABLE PARAMETERS =========================
MODEL_PATH = "yolo26n.pt"
CONFIDENCE_THRESHOLD = 0.45
IOU_THRESHOLD = 0.50
TRACKER = "bytetrack.yaml"
TRACK_PERSIST = True
YOLO_IMAGE_SIZE = 640
DEVICE = "cpu"
REACQUIRE_AFTER_MISSING_FRAMES = 10

DEPTH_ROI_PERCENT = 0.60
RIGHT_OFFSET_M = 0.20
MIN_DEPTH_M = 0.10
MAX_DEPTH_M = 10.0
FALLBACK_FX_SCALE = 0.90
FALLBACK_FY_SCALE = 0.90
# =====================================================================

FONT = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------ JSON ---------------------------------
def _xyz(v):
    return None if v is None else [round(float(x), 4) for x in v]


def build_nested_json(detected, object_name, object_id, confidence,
                      center_xyz, left_xyz, right_xyz,
                      original_xyz, target_xyz, description):
    return {
        "status": "SUCCESS" if detected else "NO_OBJECT",
        "object": {
            "object_detected": 1 if detected else 0,
            "object_name": object_name if detected else None,
            "object_id": object_id if detected else None,
            "confidence": round(float(confidence), 4)
            if detected and confidence is not None else None,
            "coordinates": {
                "center": _xyz(center_xyz),
                "left": _xyz(left_xyz),
                "right": _xyz(right_xyz),
            },
            "original_pos": _xyz(original_xyz),
            "target_pos": _xyz(target_xyz),
            "description": description,
        },
    }


# ----------------------- RAW FRAME CONVERSION ------------------------
def depth_to_numpy(frame):
    """Orbbec depth frame -> (HxW uint16 array, depth_scale)."""
    scale = float(frame.get_depth_scale())
    w, h = frame.get_width(), frame.get_height()
    data = np.asarray(frame.get_data())
    if data.dtype == np.uint8:
        if data.size != w * h * 2:
            raise RuntimeError(f"Unexpected depth byte count: {data.size}, expected {w * h * 2}")
        depth = data.view(np.uint16).reshape(h, w)
    else:
        depth = data.astype(np.uint16, copy=False).reshape(h, w)
    return np.ascontiguousarray(depth), scale


def color_to_bgr(frame):
    """Orbbec color frame -> HxWx3 uint8 BGR array (or None)."""
    w, h = frame.get_width(), frame.get_height()
    fmt = str(frame.get_format()).upper()
    data = np.asarray(frame.get_data())
    if data.dtype == np.uint8 and data.size == w * h * 3:
        img = data.reshape(h, w, 3).copy()
        if "RGB" in fmt and "BGR" not in fmt:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        return np.ascontiguousarray(img)
    if data.dtype == np.uint8 and data.size == w * h * 2:
        return cv2.cvtColor(data.reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
    if data.dtype == np.uint8 and data.size:
        return cv2.imdecode(data.reshape(-1, 1), cv2.IMREAD_COLOR)
    return None


def load_intrinsics(depth_profile):
    """Depth stream profile -> (fx, fy, cx, cy) or None."""
    try:
        i = depth_profile.get_intrinsic()
        K = (float(i.fx), float(i.fy), float(i.cx), float(i.cy))
        print(f"Depth intrinsics: FX={K[0]:.2f}, FY={K[1]:.2f}, CX={K[2]:.2f}, CY={K[3]:.2f}",
              file=sys.stderr)
        return K
    except Exception as e:
        print(f"Warning: camera intrinsics unavailable ({type(e).__name__}: {e}); "
              f"using approximate intrinsics.", file=sys.stderr)
        return None


# --------------------------- GEOMETRY --------------------------------
def _resolve_k(K, depth_shape):
    dh, dw = depth_shape[:2]
    if K is not None:
        return tuple(float(v) for v in K)
    return (dw * FALLBACK_FX_SCALE, dh * FALLBACK_FY_SCALE, dw / 2.0, dh / 2.0)


def _rgb_to_depth_pixel(x, y, color_shape, depth_shape):
    ch, cw = color_shape[:2]
    dh, dw = depth_shape[:2]
    dx = round(x * (dw - 1) / max(cw - 1, 1))
    dy = round(y * (dh - 1) / max(ch - 1, 1))
    return max(0, min(dw - 1, int(dx))), max(0, min(dh - 1, int(dy)))


def _median_bbox_depth(bbox, color, depth, depth_scale):
    x1, y1, x2, y2 = map(int, bbox)
    ch, cw = color.shape[:2]
    x1, x2 = max(0, min(cw - 1, x1)), max(0, min(cw - 1, x2))
    y1, y2 = max(0, min(ch - 1, y1)), max(0, min(ch - 1, y2))
    if x2 <= x1 or y2 <= y1:
        return None

    f = DEPTH_ROI_PERCENT
    rx1 = int(x1 + (x2 - x1) * (1 - f) / 2)
    rx2 = int(x2 - (x2 - x1) * (1 - f) / 2)
    ry1 = int(y1 + (y2 - y1) * (1 - f) / 2)
    ry2 = int(y2 - (y2 - y1) * (1 - f) / 2)

    dh, dw = depth.shape
    dx1 = int(round(rx1 * (dw - 1) / max(cw - 1, 1)))
    dx2 = int(round(rx2 * (dw - 1) / max(cw - 1, 1)))
    dy1 = int(round(ry1 * (dh - 1) / max(ch - 1, 1)))
    dy2 = int(round(ry2 * (dh - 1) / max(ch - 1, 1)))
    dx1, dx2 = max(0, min(dw - 1, dx1)), max(0, min(dw, dx2))
    dy1, dy2 = max(0, min(dh - 1, dy1)), max(0, min(dh, dy2))
    if dx2 <= dx1 or dy2 <= dy1:
        return None

    d = depth[dy1:dy2, dx1:dx2].astype(np.float32) * depth_scale / 1000.0
    valid = d[(d >= MIN_DEPTH_M) & (d <= MAX_DEPTH_M)]
    return float(np.median(valid)) if valid.size else None


def _pixel_to_3d(x, y, depth_m, color, depth, K):
    if depth_m is None:
        return None
    fx, fy, cx, cy = _resolve_k(K, depth.shape)
    dx, dy = _rgb_to_depth_pixel(x, y, color.shape, depth.shape)
    return ((dx - cx) * depth_m / fx, (dy - cy) * depth_m / fy, float(depth_m))


def _point_3d_to_rgb_pixel(xyz, color, depth, K):
    if xyz is None:
        return None
    X, Y, Z = xyz
    if Z <= 0:
        return None
    fx, fy, cx, cy = _resolve_k(K, depth.shape)
    if fx == 0 or fy == 0:
        return None
    dh, dw = depth.shape
    ch, cw = color.shape[:2]
    if dw <= 1 or dh <= 1:
        return None
    dx = X * fx / Z + cx
    dy = Y * fy / Z + cy
    px = int(round(dx * (cw - 1) / (dw - 1)))
    py = int(round(dy * (ch - 1) / (dh - 1)))
    if not (0 <= px < cw and 0 <= py < ch):
        return None
    return px, py


def _in_frame(p, w, h):
    return p is not None and 0 <= p[0] < w and 0 <= p[1] < h


def _find_obstacle(slot_xyz, dets, selected_index, color, depth, K):
    """Point is occupied only if its projected pixel lies inside another bbox."""
    pixel = _point_3d_to_rgb_pixel(slot_xyz, color, depth, K)
    if pixel is None:
        return None
    px, py = pixel
    best, best_score = None, float("inf")
    for det in dets:
        if det["index"] == selected_index:
            continue
        x1, y1, x2, y2 = det["bbox"]
        if not (x1 <= px <= x2 and y1 <= py <= y2):
            continue
        score = ((x1 + x2) / 2 - px) ** 2 + ((y1 + y2) / 2 - py) ** 2
        if score < best_score:
            best_score, best = score, dict(det)
    return best


def _choose_placement(target_xyz, right_xyz, dets, selected_index, color, depth, K):
    """+20 cm right of object, then +40 cm, else fall back to original XYZ."""
    p20_xyz = (right_xyz[0] + RIGHT_OFFSET_M, right_xyz[1], right_xyz[2])
    p40_xyz = (right_xyz[0] + 2 * RIGHT_OFFSET_M, right_xyz[1], right_xyz[2])
    p20 = _point_3d_to_rgb_pixel(p20_xyz, color, depth, K)
    p40 = _point_3d_to_rgb_pixel(p40_xyz, color, depth, K)
    h, w = color.shape[:2]
    p20_in, p40_in = _in_frame(p20, w, h), _in_frame(p40, w, h)

    ob20 = _find_obstacle(p20_xyz, dets, selected_index, color, depth, K) if p20_in else None
    ob40 = None
    if ob20 is not None and p40_in:
        ob40 = _find_obstacle(p40_xyz, dets, selected_index, color, depth, K)

    base = {"p20_pixel": p20, "p40_pixel": p40, "p40_in": p40_in,
            "obstacle20": ob20, "obstacle40": ob40}

    if not p20_in:
        return {**base, "xyz": target_xyz, "pixel": None,
                "message": "20cm is out of frame, using original object position"}
    if ob20 is None:
        return {**base, "xyz": p20_xyz, "pixel": p20,
                "message": "Free to keep object at +20cm"}

    n20 = ob20.get("class_name", "object")
    if not p40_in:
        return {**base, "xyz": target_xyz, "pixel": None,
                "message": f"There is {n20} at +20cm, +40cm is out of frame, blocked"}
    if ob40 is None:
        return {**base, "xyz": p40_xyz, "pixel": p40,
                "message": f"There is {n20} at +20cm, +40cm is free to keep object"}

    n40 = ob40.get("class_name", "object")
    return {**base, "xyz": target_xyz, "pixel": None,
            "message": f"There is {n20} at +20cm and {n40} at +40cm, no place to keep object"}


# ---------------------------- ENGINE ---------------------------------
class PerceptionEngine:
    """Holds YOLO model + tracking/grab state. One instance per camera."""

    def __init__(self, model_path=MODEL_PATH, conf=CONFIDENCE_THRESHOLD,
                 iou=IOU_THRESHOLD, tracker=TRACKER, imgsz=YOLO_IMAGE_SIZE,
                 device=DEVICE):
        self.model = YOLO(model_path)
        self.conf, self.iou, self.tracker = float(conf), float(iou), tracker
        self.imgsz, self.device = int(imgsz), device
        self.target = None
        self.grab_state = 0
        self.locked_id = None
        self.missing = 0
        self.original_pos = None
        self.original_right = None     # frozen right-edge XYZ at grab
        self.target_pos_fixed = None   # frozen placement XYZ at grab
        self.target_is_slot = False    # True if fixed target is a real +20/+40 slot
        # latest converted sensor data (reused if a frameset lacks a stream)
        self.latest_color = None
        self.latest_depth = None
        self.depth_scale = 1.0
        self.intrinsics = None
        self._k_loaded = False

    # -- raw frame ingestion --
    def ingest(self, frames, depth_profile=None):
        if depth_profile is not None and not self._k_loaded:
            self.intrinsics = load_intrinsics(depth_profile)
            self._k_loaded = True
        if frames is None:
            return
        try:
            df = frames.get_depth_frame()
            if df is not None:
                self.latest_depth, self.depth_scale = depth_to_numpy(df)
            cf = frames.get_color_frame()
            if cf is not None:
                c = color_to_bgr(cf)
                if c is not None:
                    self.latest_color = c
        except Exception as e:      # keep last good frames
            print(f"[FRAME ERROR] {type(e).__name__}: {e}", file=sys.stderr)

    def process(self, frames, target_object=None, grab_state=None, depth_profile=None):
        """Raw frames in -> (payload_dict, debug_dict)."""
        self.ingest(frames, depth_profile)
        payload, dbg = self.process_arrays(self.latest_color, self.latest_depth,
                                           target_object, grab_state,
                                           self.depth_scale, self.intrinsics)
        dbg.update({"color": self.latest_color, "payload": payload,
                    "target": self.target, "grab": self.grab_state})
        return payload, dbg

    # -- state helpers --
    def set_target(self, name):
        name = name.strip().lower() if name else None
        if name == self.target:
            return
        self.target = name
        self.locked_id, self.missing = None, 0
        self.grab_state = 0
        self._clear_grab()
        try:
            self.model.predictor = None  # clear tracker memory
        except Exception:
            pass

    def set_grab(self, value):
        value = 1 if int(value) else 0
        if self.grab_state == 1 and value == 0:
            self._clear_grab()
        self.grab_state = value

    def reset(self):
        self.grab_state = 0
        self._clear_grab()

    def _clear_grab(self):
        self.original_pos = None
        self.original_right = None
        self.target_pos_fixed = None
        self.target_is_slot = False

    def _grab_placement(self, original, ref_right, dets, idx, color, depth, K):
        """After grab: keep the fixed target while it stays free; if it becomes
        obstructed (or none was fixed yet), run the base +20/+40/original logic
        from the frozen original position and fix the new result."""
        h, w = color.shape[:2]
        replanned = False
        if self.target_pos_fixed is not None and self.target_is_slot:
            pix = _point_3d_to_rgb_pixel(self.target_pos_fixed, color, depth, K)
            blocker = (_find_obstacle(self.target_pos_fixed, dets, idx, color, depth, K)
                       if _in_frame(pix, w, h) else None)
            if blocker is None:
                return {"xyz": self.target_pos_fixed, "pixel": pix,
                        "p20_pixel": None, "p40_pixel": None, "p40_in": True,
                        "obstacle20": None, "obstacle40": None,
                        "message": "Fixed target position is free"}, "fixed"
            replanned = True

        pl = _choose_placement(original, ref_right, dets, idx, color, depth, K)
        self.target_pos_fixed = pl["xyz"]
        self.target_is_slot = pl["xyz"] != original
        return pl, ("replanned" if replanned else "planned")

    # -- main entry --
    def process_arrays(self, color, depth, target_object=None, grab_state=None,
                       depth_scale=1.0, intrinsics=None):
        """Already-converted arrays in -> (payload_dict, debug_dict)."""
        if target_object != self.target:
            self.set_target(target_object)
        if grab_state is not None:
            self.set_grab(grab_state)

        dbg = {"detections": [], "selected": None, "locked_id": self.locked_id,
               "missing": self.missing}

        if self.target is None:
            return build_nested_json(False, None, None, None, None, None, None,
                                     None, None, "No target object selected"), dbg
        if color is None or depth is None:
            return build_nested_json(False, self.target, None, None, None, None, None,
                                     None, None, "No frame available"), dbg

        result = self.model.track(source=color, imgsz=self.imgsz, conf=self.conf,
                                  iou=self.iou, device=self.device,
                                  tracker=self.tracker, persist=TRACK_PERSIST,
                                  verbose=False)[0]

        dets, targets = [], []
        boxes = result.boxes
        if boxes is not None and len(boxes):
            for i in range(len(boxes)):
                name = str(result.names[int(boxes.cls[i].item())])
                det = {
                    "index": i, "class_name": name,
                    "confidence": float(boxes.conf[i].item()),
                    "bbox": tuple(int(v) for v in boxes.xyxy[i].cpu().numpy()),
                    "track_id": int(boxes.id[i].item()) if boxes.id is not None else None,
                }
                dets.append(det)
                if name.lower() == self.target:
                    targets.append(det)
        dbg["detections"] = dets

        # ---- lock / reacquire logic ----
        selected = None
        if self.locked_id is not None:
            for d in targets:
                if d["track_id"] == self.locked_id:
                    selected, self.missing = d, 0
                    break
            if selected is None:
                self.missing += 1
                if self.missing >= REACQUIRE_AFTER_MISSING_FRAMES:
                    self.locked_id, self.missing = None, 0

        if self.locked_id is None and selected is None:
            for d in targets:
                if d["track_id"] is not None:
                    self.locked_id, self.missing, selected = d["track_id"], 0, d
                    break
            if selected is None and targets:
                selected = targets[0]

        dbg["locked_id"], dbg["missing"] = self.locked_id, self.missing
        frozen = self.original_pos if self.grab_state == 1 else None
        frozen_target = (self.target_pos_fixed if self.target_pos_fixed is not None
                         else frozen) if self.grab_state == 1 else None

        if selected is None:
            return build_nested_json(False, self.target, self.locked_id, None,
                                     None, None, None, frozen, frozen_target,
                                     "Target object not detected"), dbg

        # ---- 3D computation ----
        h, w = color.shape[:2]
        x1, y1, x2, y2 = selected["bbox"]
        x1, x2 = max(0, min(w - 1, x1)), max(0, min(w - 1, x2))
        y1, y2 = max(0, min(h - 1, y1)), max(0, min(h - 1, y2))
        bbox = (x1, y1, x2, y2)
        conf, track_id = selected["confidence"], self.locked_id

        median_depth = _median_bbox_depth(bbox, color, depth, depth_scale) \
            if (x2 > x1 and y2 > y1) else None
        if median_depth is None:
            dbg["selected"] = {"bbox": bbox, "depth_invalid": True,
                               "confidence": conf, "track_id": track_id}
            return build_nested_json(False, self.target, track_id, None,
                                     None, None, None, frozen, frozen_target,
                                     "Depth invalid for target"), dbg

        cx_px, cy_px = (x1 + x2) // 2, (y1 + y2) // 2
        center = _pixel_to_3d(cx_px, cy_px, median_depth, color, depth, intrinsics)
        left = _pixel_to_3d(x1, cy_px, median_depth, color, depth, intrinsics)
        right = _pixel_to_3d(x2, cy_px, median_depth, color, depth, intrinsics)

        # ---- grab / original_pos / target_pos ----
        mode = "dynamic"
        if self.grab_state == 0:
            original = tuple(center)
            placement = _choose_placement(original, right, dets, selected["index"],
                                          color, depth, intrinsics)
        else:
            if self.original_pos is None:      # freeze at first valid grab frame
                self.original_pos = tuple(float(v) for v in center)
                self.original_right = tuple(float(v) for v in right)
            original = self.original_pos
            placement, mode = self._grab_placement(original, self.original_right, dets,
                                                   selected["index"], color, depth,
                                                   intrinsics)
        target_xyz = placement["xyz"]

        # Description logic (same as original test.py)
        if self.grab_state == 0:
            desc = "Object detected; position is updating dynamically"
        elif target_xyz is not None and original is not None and target_xyz != original:
            desc = "Valid target position available"
        else:
            desc = "No valid target position available; using original position"

        dbg["selected"] = {
            "bbox": bbox, "center_px": (cx_px, cy_px), "confidence": conf,
            "track_id": track_id, "center": center, "left": left, "right": right,
            "target": target_xyz, "placement": placement,
        }
        payload = build_nested_json(True, self.target, track_id, conf, center, left,
                                    right, original, target_xyz, desc)
        return payload, dbg


# ------------------------ DEBUG OVERLAY (optional) --------------------
def _txt(img, text, pos, color=(255, 255, 255)):
    cv2.putText(img, text, pos, FONT, 0.45, color, 1, cv2.LINE_AA)


def draw_debug(dbg, fps=None):
    """Return the annotated BGR image for a debug dict from perception.main(return_debug=True)."""
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
        for key, tag in (("obstacle20", "+20"), ("obstacle40", "+40")):
            ob = pl.get(key)
            if ob:
                ox1, oy1, ox2, oy2 = ob["bbox"]
                cv2.rectangle(out, (ox1, oy1), (ox2, oy2), (0, 0, 255), 2)
                _txt(out, f'{ob["class_name"]} {tag}', (ox1, max(15, oy1 - 6)), (0, 0, 255))
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
                _txt(out, f"{tag} {v[0]:+.2f},{v[1]:+.2f},{v[2]:.2f}", (x1, y + 17 * i), c)
    elif sel and sel.get("depth_invalid"):
        x1, y1, x2, y2 = sel["bbox"]
        _txt(out, "Depth: INVALID", (x1, y2 + 18), (0, 0, 255))

    if dbg.get("locked_id") is not None and dbg.get("missing"):
        _txt(out, f'Locked ID:{dbg["locked_id"]} not visible '
                  f'({dbg["missing"]}/{REACQUIRE_AFTER_MISSING_FRAMES})', (8, 72), (0, 165, 255))
    _txt(out, f'Target: {dbg["target"]}' if dbg.get("target") else "No target selected", (8, 20))
    _txt(out, f'Grab: {dbg.get("grab", 0)}', (8, 38))
    if fps is not None:
        _txt(out, f"FPS: {fps:.1f}", (8, 56))
    _txt(out, "G: grab | R: reset | Q: quit", (8, 90))
    return out


# ------------------------- PUBLIC ENTRY POINT -------------------------
_engine = None


def main(frames, target_object=None, grab_state=0, depth_profile=None,
         return_debug=False, **engine_kwargs):
    """
    Raw camera frames in -> JSON string out.

    engine_kwargs (first call only): model_path, conf, iou, tracker, imgsz, device
    return_debug=True -> returns (json_str, debug_dict); pass debug_dict to draw_debug().
    """
    global _engine
    if _engine is None:
        _engine = PerceptionEngine(**engine_kwargs)
    payload, dbg = _engine.process(frames, target_object, grab_state, depth_profile)
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return (encoded, dbg) if return_debug else encoded


def get_engine():
    return _engine