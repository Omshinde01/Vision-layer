"""

perception.py - YOLO tracking + 3D placement (production pipeline).



Raw SDK frame -> numpy/BGR conversion, depth scale, camera intrinsics, YOLO

tracking, 3D math, placement validation, JSON. No printing, drawing or logging.



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

    JSON string (robot-facing nested JSON). `status` is "SUCCESS" or "NO_OBJECT".



Optional introspection (for callers that want to render or log):

    perception.main(..., return_details=True) -> (json_str, details_dict)

    perception.get_engine().warnings          -> list of non-fatal warnings



Coordinates: X = right, Y = down, Z = forward, in MEASUREMENT_UNIT (m / cm / mm).

"""



import json



import cv2

import numpy as np

from ultralytics import YOLO



# ========================= TUNABLE PARAMETERS =========================

MODEL_PATH = "yolo26n.pt"

CONFIDENCE_THRESHOLD = 0.55

IOU_THRESHOLD = 0.50

TRACKER = "bytetrack.yaml"

TRACK_PERSIST = True

YOLO_IMAGE_SIZE = 640

DEVICE = "cpu"

REACQUIRE_AFTER_MISSING_FRAMES = 10



# Class names that must NEVER be detected (case-insensitive). Removed at model level.

EXCLUDED_CLASSES = ["dining table", "chair", "person", "tv"]



DEPTH_ROI_PERCENT = 0.60

FALLBACK_FX_SCALE = 0.90

FALLBACK_FY_SCALE = 0.90



# ---- Measurement unit: "m", "cm" or "mm" ----

MEASUREMENT_UNIT = "cm"



_MM_PER_UNIT_TABLE = {"mm": 1.0, "cm": 10.0, "m": 1000.0}

MEASUREMENT_UNIT = str(MEASUREMENT_UNIT).strip().lower()

if MEASUREMENT_UNIT not in _MM_PER_UNIT_TABLE:

    raise ValueError(f"MEASUREMENT_UNIT must be one of {list(_MM_PER_UNIT_TABLE)}, "

                     f"got {MEASUREMENT_UNIT!r}")

MM_PER_UNIT = _MM_PER_UNIT_TABLE[MEASUREMENT_UNIT]





def _q(value, unit):

    """Write a distance in any unit ("mm"/"cm"/"m"); returns it in MEASUREMENT_UNIT."""

    return float(value) * _MM_PER_UNIT_TABLE[unit] / MM_PER_UNIT





# ---- Distances (value + unit, auto-converted to MEASUREMENT_UNIT) ----

RIGHT_OFFSET = _q(20, "cm")                     # placement step: +20 cm, then +40 cm

MIN_DEPTH = _q(20, "cm")                        # valid depth range

MAX_DEPTH = _q(10, "m")

SURFACE_DEPTH_THRESHOLD = _q(5, "cm")           # legacy base-point |dZ| <= this -> same surface

OBSTACLE_DEPTH_SAFETY_THRESHOLD = _q(15, "cm")  # obstacle |dZ| > this -> not blocking

BASE_POINT_PATCH_PX = 5                         # half-size (depth px) of base-point depth patch

ALLOW_PLACEMENT_IF_BASE_DEPTH_INVALID = False   # legacy base-point check



# ---- Feature toggles ----

ENABLE_SURFACE_VALIDATION = False       # legacy base-point same-surface check

ENABLE_OBSTACLE_CHECK = True            # master switch: can any detected obstacle block a slot?

ENABLE_BBOX_OVERLAP = True              # True: placement-bbox overlap; False: slot pixel in bbox

ENABLE_OBSTACLE_DEPTH_AVOIDANCE = True  # obstacle clearly in front/behind is not blocking

ENABLE_FREE_SPACE_CHECK = True          # depth-based supporting-surface / free-space check



# ---- Free-space (supporting surface) check ----

# The support plane is fitted (RANSAC) to depth points in a band around the target's

# base edge. A slot is FREE only if its footprint (target width x footprint depth,

# plus PLACEMENT_BUFFER on every side) lies on that plane, has enough valid depth,

# contains no raised obstruction and is flat at the plane height.

PLACEMENT_BUFFER = _q(2, "cm")              # margin checked around the placement footprint

FOOTPRINT_DEPTH_RATIO = 0.5                 # footprint image-height as a fraction of target width

SURFACE_PLANE_TOLERANCE = _q(1.5, "cm")     # |height above plane| <= this -> pixel is on surface

SURFACE_MAX_POINT_DEVIATION = _q(1.0, "cm")

SURFACE_MAX_MEDIAN_DEVIATION = _q(0.6, "cm")

MIN_FLAT_SURFACE_RATIO = 0.95

SURFACE_SCAN_STEP_PX = 2

SURFACE_SCAN_MIN_POINTS = 80

SURFACE_SCAN_RANSAC_ITERATIONS = 120

SURFACE_SCAN_MAX_SAMPLES = 2500

SURFACE_SCAN_MIN_INLIER_RATIO = 0.60
# Grid-based placement-surface validation.
# The footprint is divided into cells; every sufficiently valid cell must be flat.
# ---- Depth-only physical surface analysis ----
# A point within this distance of the common support plane is considered surface.
SURFACE_TOLERANCE = _q(0.8, "cm")
SURFACE_PLANE_TOLERANCE = SURFACE_TOLERANCE

# Raised/depressed depth beyond these values is treated as physical obstruction
# or a surface drop rather than normal RGB-D noise.
OBSTRUCTION_HEIGHT_THRESHOLD = _q(1.5, "cm")
DEPRESSION_DEPTH_THRESHOLD = _q(1.5, "cm")

# Allowed fraction of footprint points that may be raised/depressed.
MAX_OBSTRUCTION_RATIO = 0.02

# Required fraction of footprint points belonging to the support plane.

# Robust planarity limits.
SURFACE_RMS_TOLERANCE = _q(0.8, "cm")
SURFACE_P95_TOLERANCE = _q(1.5, "cm")



MIN_SURFACE_COVERAGE = 0.90                 # min fraction of footprint inside the surface contour

MIN_VALID_DEPTH_RATIO = 0.80                # min fraction of footprint pixels with valid depth

MAX_OBSTRUCTION_RATIO = 0.02                # max fraction of footprint covered by raised contours


SURFACE_MORPH_KERNEL_PX = 3                 # close/open kernel for depth speckle

SURFACE_CONTEXT_RATIO = 0.5                 # extra context around the footprint for contours

PLANE_BAND_SIDE_RATIO = 0.5                 # plane-fit band: extra width each side (x target width)

PLANE_BAND_HEIGHT_RATIO = 0.5               # plane-fit band: height below base (x target width)

PLANE_BAND_MIN_PX = 12                      # plane-fit band: minimum height (depth px)

PLANE_MIN_POINTS = 150                      # min valid points to accept a plane

PLANE_MIN_INLIER_RATIO = 0.5                # min RANSAC inlier fraction

PLANE_RANSAC_ITERATIONS = 100

PLANE_MAX_SAMPLES = 2000                    # points subsampled for RANSAC

ALLOW_PLACEMENT_IF_SURFACE_UNKNOWN = False  # False = reject when the surface can't be fitted
# Support plane is fitted from a ring OUTSIDE the candidate footprint.
# This prevents a notebook/book/object from becoming the reference plane.
PLACEMENT_PLANE_RING_RATIO = 1.5
PLACEMENT_PLANE_RING_MIN_PX = 35
PLACEMENT_PLANE_MIN_POINTS = 150
# Maximum 3D distance between the original object's support point and the
# candidate placement support plane for them to be considered the same surface.
SAME_SURFACE_MAX_3D_DISTANCE = _q(2.0, "cm")


# ---- Adaptive RGB-D surface analysis ----
SURFACE_MIN_TOLERANCE = _q(0.8, "cm")
SURFACE_MAX_ADAPTIVE_TOLERANCE = _q(3.0, "cm")
SURFACE_NOISE_SIGMA_MULTIPLIER = 3.0
MIN_NOISE_SAMPLE_POINTS = 100

OBSTRUCTION_MIN_HEIGHT = _q(1.5, "cm")
DEPRESSION_MIN_DEPTH = _q(1.5, "cm")
OBSTRUCTION_TOLERANCE_MULTIPLIER = 2.0
DEPRESSION_TOLERANCE_MULTIPLIER = 2.0

MAX_DEPRESSION_RATIO = 0.02
MIN_SURFACE_RATIO = 0.90
SURFACE_MAX_P95_TOLERANCE = _q(3.0, "cm")
P95_TOLERANCE_MULTIPLIER = 3.0

OBSTRUCTION_MORPH_KERNEL_PX = 3
MIN_OBSTRUCTION_AREA_PX = 20

ADAPTIVE_CONTEXT_RATIO = 1.0
ADAPTIVE_CONTEXT_MIN_PX = 30

SUPPORT_CONTEXT_SIDE_RATIO = 1.5
SUPPORT_CONTEXT_HEIGHT_RATIO = 1.5
SUPPORT_CONTEXT_ABOVE_RATIO = 0.25
SUPPORT_CONTEXT_MIN_PX = 40


# =====================================================================





# ------------------------------ JSON ---------------------------------

def _xyz(v):

    return None if v is None else [round(float(x), 4) for x in v]





def build_nested_json(detected, object_name, object_id, confidence,

                      center_xyz, left_xyz, right_xyz,

                      original_xyz, target_xyz, description):

    return {

        "status": "SUCCESS" if detected else "NO_OBJECT",

        "object": {

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

        return (float(i.fx), float(i.fy), float(i.cx), float(i.cy))

    except Exception:

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





def _depth_rect(bbox, color_shape, depth_shape):

    """RGB-pixel box -> float box in depth-pixel coordinates."""

    ch, cw = color_shape[:2]

    dh, dw = depth_shape[:2]

    sx = (dw - 1) / max(cw - 1, 1)

    sy = (dh - 1) / max(ch - 1, 1)

    return bbox[0] * sx, bbox[1] * sy, bbox[2] * sx, bbox[3] * sy





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



    d = depth[dy1:dy2, dx1:dx2].astype(np.float64) * depth_scale / MM_PER_UNIT

    valid = d[(d >= MIN_DEPTH) & (d <= MAX_DEPTH)]

    return float(np.median(valid)) if valid.size else None





def _valid_depth(z):

    return z is not None and bool(np.isfinite(z)) and MIN_DEPTH <= z <= MAX_DEPTH





def _sample_point_depth(x, y, color_shape, depth, depth_scale):

    """Robust (patch-median) depth, in MEASUREMENT_UNIT, at an RGB pixel."""

    ch, cw = color_shape[:2]

    if not (0 <= x < cw and 0 <= y < ch):

        return None

    dx, dy = _rgb_to_depth_pixel(x, y, color_shape, depth.shape)

    dh, dw = depth.shape

    r = int(BASE_POINT_PATCH_PX)

    patch = depth[max(0, dy - r):min(dh, dy + r + 1), max(0, dx - r):min(dw, dx + r + 1)]

    if patch.size == 0:

        return None

    d = patch.astype(np.float64) * depth_scale / MM_PER_UNIT

    valid = d[(d >= MIN_DEPTH) & (d <= MAX_DEPTH)]

    return float(np.median(valid)) if valid.size else None





def _pixel_to_3d(x, y, z, color, depth, K):

    """Pixel + depth z (MEASUREMENT_UNIT) -> (X, Y, Z) in MEASUREMENT_UNIT."""

    if z is None:

        return None

    fx, fy, cx, cy = _resolve_k(K, depth.shape)

    dx, dy = _rgb_to_depth_pixel(x, y, color.shape, depth.shape)

    return ((dx - cx) * z / fx, (dy - cy) * z / fy, float(z))





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





def _placement_bbox(pixel, bw, bh):

    """Bounding box (x1, y1, x2, y2) of target size centred on a placement pixel."""

    x1 = int(pixel[0] - bw // 2)

    y1 = int(pixel[1] - bh // 2)

    return (x1, y1, x1 + int(bw), y1 + int(bh))





def _bbox_overlap(a, b):

    return (min(a[2], b[2]) - max(a[0], b[0]) > 0 and

            min(a[3], b[3]) - max(a[1], b[1]) > 0)





def _same_surface(ref_base, pbbox, color, depth, depth_scale):

    """Legacy check: base-point (bottom-centre) depth of original vs placement bbox."""

    z_ref = ref_base[2] if ref_base is not None else None

    base_px = ((pbbox[0] + pbbox[2]) // 2, pbbox[3])

    z_new = _sample_point_depth(base_px[0], base_px[1], color.shape, depth, depth_scale)

    if not _valid_depth(z_ref) or not _valid_depth(z_new):

        return ALLOW_PLACEMENT_IF_BASE_DEPTH_INVALID

    return abs(z_new - z_ref) <= SURFACE_DEPTH_THRESHOLD





# ----------------------- SUPPORT SURFACE / FREE SPACE ----------------

def _crop_cloud(depth, depth_scale, K, x1, y1, x2, y2):

    """Back-project a depth crop. Returns X, Y, Z (MEASUREMENT_UNIT) and a validity mask."""

    fx, fy, cx, cy = _resolve_k(K, depth.shape)

    z = depth[y1:y2, x1:x2].astype(np.float64) * depth_scale / MM_PER_UNIT

    valid = (z >= MIN_DEPTH) & (z <= MAX_DEPTH)

    u = np.arange(x1, x2, dtype=np.float64)[None, :]

    v = np.arange(y1, y2, dtype=np.float64)[:, None]

    return (u - cx) * z / fx, (v - cy) * z / fy, z, valid





def _fit_plane(points):

    """RANSAC plane fit. Returns (normal, c) with n.P = c, normal oriented toward the

    camera (c < 0), or None. Height above the plane is n.P - c (positive = toward camera)."""

    n_pts = len(points)

    if n_pts < PLANE_MIN_POINTS:

        return None

    rng = np.random.default_rng(0)

    idx = rng.integers(0, n_pts, size=(PLANE_RANSAC_ITERATIONS, 3))

    p0, p1, p2 = points[idx[:, 0]], points[idx[:, 1]], points[idx[:, 2]]

    normals = np.cross(p1 - p0, p2 - p0)

    norms = np.linalg.norm(normals, axis=1)

    ok = norms > 1e-9

    if not ok.any():

        return None

    normals = normals[ok] / norms[ok, None]

    offsets = np.einsum("ij,ij->i", normals, p0[ok])

    dist = np.abs(points @ normals.T - offsets[None, :])

    inliers_mask = dist <= SURFACE_PLANE_TOLERANCE

    best = int(np.argmax(inliers_mask.sum(axis=0)))

    inl = points[inliers_mask[:, best]]

    if len(inl) < PLANE_MIN_POINTS or len(inl) / n_pts < PLANE_MIN_INLIER_RATIO:

        return None

    centroid = inl.mean(axis=0)

    n = np.linalg.svd(inl - centroid, full_matrices=False)[2][2]

    c = float(n @ centroid)

    if c > 0:

        n, c = -n, -c

    return n, c





def _estimate_support_plane(bbox, dets, color_shape, depth, depth_scale, K):
    """Estimate the dominant supporting surface from a larger depth region."""
    dh, dw = depth.shape
    x1, y1, x2, y2 = _depth_rect(bbox, color_shape, depth.shape)
    bw = max(x2-x1, 1.0)
    bh = max(y2-y1, 1.0)
    side=max(bw*SUPPORT_CONTEXT_SIDE_RATIO, SUPPORT_CONTEXT_MIN_PX)
    below=max(bw*SUPPORT_CONTEXT_HEIGHT_RATIO, SUPPORT_CONTEXT_MIN_PX)
    X1=max(0,int(x1-side)); X2=min(dw,int(x2+side))
    Y1=max(0,int(y2-bh*SUPPORT_CONTEXT_ABOVE_RATIO)); Y2=min(dh,int(y2+below))
    if X2-X1<10 or Y2-Y1<10: return None

    keep=np.ones((Y2-Y1,X2-X1),dtype=bool)
    for det in dets:
        ax,ay,bx,by=_depth_rect(det["bbox"],color_shape,depth.shape)
        ax=max(0,int(ax)-X1); bx=min(X2-X1,int(bx)+1-X1)
        ay=max(0,int(ay)-Y1); by=min(Y2-Y1,int(by)+1-Y1)
        if bx>ax and by>ay: keep[ay:by,ax:bx]=False

    Xc,Yc,Zc,valid=_crop_cloud(depth,depth_scale,K,X1,Y1,X2,Y2)
    mask=valid & keep
    pts=np.stack([Xc[mask],Yc[mask],Zc[mask]],axis=1)
    if len(pts)<PLANE_MIN_POINTS: return None
    if len(pts)>PLANE_MAX_SAMPLES:
        pts=pts[np.random.default_rng(1).choice(len(pts),PLANE_MAX_SAMPLES,replace=False)]
    return _fit_plane(pts)


def _fit_local_plane(points):
    """RANSAC plane fit directly on the proposed placement footprint."""
    if len(points) < SURFACE_SCAN_MIN_POINTS:
        return None
    rng = np.random.default_rng(7)
    best_count, best_inliers = 0, None
    for _ in range(SURFACE_SCAN_RANSAC_ITERATIONS):
        ids = rng.choice(len(points), 3, replace=False)
        p0, p1, p2 = points[ids]
        normal = np.cross(p1 - p0, p2 - p0)
        norm = np.linalg.norm(normal)
        if norm < 1e-9:
            continue
        normal /= norm
        c = float(normal @ p0)
        residual = np.abs(points @ normal - c)
        inliers = residual <= SURFACE_PLANE_TOLERANCE
        count = int(inliers.sum())
        if count > best_count:
            best_count, best_inliers = count, inliers
    if best_inliers is None or best_count < SURFACE_SCAN_MIN_POINTS:
        return None
    if best_count / len(points) < SURFACE_SCAN_MIN_INLIER_RATIO:
        return None
    inlier_points = points[best_inliers]
    centroid = inlier_points.mean(axis=0)
    normal = np.linalg.svd(inlier_points - centroid, full_matrices=False)[2][2]
    norm = np.linalg.norm(normal)
    if norm < 1e-9:
        return None
    normal /= norm
    c = float(normal @ centroid)
    if c > 0:
        normal, c = -normal, -c
    return normal, c



def _estimate_placement_plane(
    fx1, fy1, fx2, fy2,
    depth, depth_scale, K,
    dets, color_shape
):
    """
    Estimate the real supporting surface from an annulus around the proposed
    placement footprint.

    The footprint itself is deliberately excluded. This is critical when the
    footprint contains a flat object such as a notebook: fitting the plane
    using the footprint could incorrectly make the notebook look like the
    supporting table.
    """
    dh, dw = depth.shape

    pad = int(max(
        (fx2 - fx1) * PLACEMENT_PLANE_RING_RATIO,
        PLACEMENT_PLANE_RING_MIN_PX,
    ))

    rx1 = max(0, fx1 - pad)
    ry1 = max(0, fy1 - pad)
    rx2 = min(dw, fx2 + pad)
    ry2 = min(dh, fy2 + pad)

    if rx2 <= rx1 or ry2 <= ry1:
        return None, None

    Xc, Yc, Zc, valid = _crop_cloud(
        depth, depth_scale, K, rx1, ry1, rx2, ry2
    )

    keep = valid.copy()

    # Remove the complete candidate footprint.
    local_fx1 = fx1 - rx1
    local_fy1 = fy1 - ry1
    local_fx2 = fx2 - rx1
    local_fy2 = fy2 - ry1

    keep[
        max(0, local_fy1):min(ry2 - ry1, local_fy2),
        max(0, local_fx1):min(rx2 - rx1, local_fx2)
    ] = False

    # Remove known detected object boxes from the support-plane ring.
    for det in dets:
        ax, ay, bx, by = _depth_rect(
            det["bbox"], color_shape, depth.shape
        )

        ax -= rx1
        bx -= rx1
        ay -= ry1
        by -= ry1

        ax = max(0, int(ax))
        ay = max(0, int(ay))
        bx = min(rx2 - rx1, int(bx) + 1)
        by = min(ry2 - ry1, int(by) + 1)

        if bx > ax and by > ay:
            keep[ay:by, ax:bx] = False

    pts = np.stack(
        [Xc[keep], Yc[keep], Zc[keep]],
        axis=1
    )

    if len(pts) < PLACEMENT_PLANE_MIN_POINTS:
        return None, None

    if len(pts) > PLANE_MAX_SAMPLES:
        pts = pts[
            np.random.default_rng(7).choice(
                len(pts), PLANE_MAX_SAMPLES, replace=False
            )
        ]

    plane = _fit_plane(pts)
    return plane, (rx1, ry1, rx2, ry2)



def _check_original_and_placement_surface(original_ref, placement_plane):
    """
    Check whether the original object's support point lies on the proposed
    placement support plane. This uses true 3D point-to-plane distance rather
    than comparing raw Z values, so a tilted table can still count as one
    continuous surface.
    """
    if original_ref is None:
        return False, "place location is not on same surface"

    original_base = original_ref.get("base")
    if original_base is None:
        return False, "place location is not on same surface"

    if placement_plane is None:
        return False, "place location is not on same surface"

    try:
        base = np.asarray(original_base, dtype=np.float64)
        n, c = placement_plane

        if base.shape != (3,) or not np.all(np.isfinite(base)):
            return False, "place location is not on same surface"

        diff = abs(float(np.dot(n, base) - c))

        if diff > SAME_SURFACE_MAX_3D_DISTANCE:
            return False, (
                f"place location is not on same surface "
                f"(surface diff={diff:.2f}{MEASUREMENT_UNIT})"
            )

        return True, (
            f"same surface (surface diff={diff:.2f}{MEASUREMENT_UNIT})"
        )
    except Exception:
        return False, "place location is not on same surface"


def _scan_local_surface(fx1, fy1, fx2, fy2, depth, depth_scale, K, support_plane, context_depth):
    """Check placement footprint against one support plane using adaptive depth noise."""
    dh,dw=depth.shape
    fx1,fy1=max(0,int(fx1)),max(0,int(fy1))
    fx2,fy2=min(dw,int(fx2)),min(dh,int(fy2))
    if fx2<=fx1 or fy2<=fy1: return False,{"reason":"invalid placement area"}
    if support_plane is None: return False,{"reason":"support plane unavailable"}

    n,c=support_plane
    X,Y,Z,valid=_crop_cloud(depth,depth_scale,K,fx1,fy1,fx2,fy2)
    if valid.size==0: return False,{"reason":"empty placement area"}
    valid_ratio=float(valid.mean())
    if valid_ratio<MIN_VALID_DEPTH_RATIO:
        return False,{"reason":"insufficient depth data","valid_ratio":valid_ratio}

    residual=n[0]*X+n[1]*Y+n[2]*Z-c
    rv=residual[valid]
    if rv.size<10: return False,{"reason":"too few valid depth points"}
    abs_rv=np.abs(rv)

    adaptive_tol=SURFACE_MIN_TOLERANCE
    if context_depth is not None:
        cx1,cy1,cx2,cy2=context_depth
        cx1,cy1=max(0,int(cx1)),max(0,int(cy1))
        cx2,cy2=min(dw,int(cx2)),min(dh,int(cy2))
        if cx2>cx1 and cy2>cy1:
            Xn,Yn,Zn,vn=_crop_cloud(depth,depth_scale,K,cx1,cy1,cx2,cy2)
            rn=n[0]*Xn+n[1]*Yn+n[2]*Zn-c
            sr=np.abs(rn[vn])
            if sr.size>=MIN_NOISE_SAMPLE_POINTS:
                med=float(np.median(sr))
                mad=float(np.median(np.abs(sr-med)))
                sigma=1.4826*mad
                adaptive_tol=max(SURFACE_MIN_TOLERANCE,min(
                    SURFACE_MAX_ADAPTIVE_TOLERANCE,
                    med+SURFACE_NOISE_SIGMA_MULTIPLIER*sigma))

    raised_threshold=max(OBSTRUCTION_MIN_HEIGHT,adaptive_tol*OBSTRUCTION_TOLERANCE_MULTIPLIER)
    depressed_threshold=max(DEPRESSION_MIN_DEPTH,adaptive_tol*DEPRESSION_TOLERANCE_MULTIPLIER)
    surface_ratio=float(np.mean(abs_rv<=adaptive_tol))
    raised_ratio=float(np.mean(rv>raised_threshold))
    depressed_ratio=float(np.mean(rv<-depressed_threshold))
    rms=float(np.sqrt(np.mean(rv**2)))
    p95=float(np.percentile(abs_rv,95))

    detail={"valid_ratio":round(valid_ratio,4),"adaptive_tolerance":round(adaptive_tol,4),
            "surface_ratio":round(surface_ratio,4),"raised_ratio":round(raised_ratio,4),
            "depressed_ratio":round(depressed_ratio,4),"rms_error":round(rms,4),
            "p95_error":round(p95,4),"median_abs_error":round(float(np.median(abs_rv)),4)}

    raised_map=valid & (residual>raised_threshold)
    k=np.ones((max(1,int(OBSTRUCTION_MORPH_KERNEL_PX)),)*2,np.uint8)
    raised_u8=cv2.morphologyEx(raised_map.astype(np.uint8)*255,cv2.MORPH_OPEN,k)
    raised_u8=cv2.morphologyEx(raised_u8,cv2.MORPH_CLOSE,k)
    contours,_=cv2.findContours(raised_u8,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    significant=0; largest=0.0
    for cn in contours:
        area=float(cv2.contourArea(cn)); largest=max(largest,area)
        if area>=MIN_OBSTRUCTION_AREA_PX: significant+=1
    detail["obstruction_components"]=significant
    detail["largest_obstruction_px"]=round(largest,1)

    if significant>0:
        detail["reason"]="obstruction detected"; return False,detail
    if depressed_ratio>MAX_DEPRESSION_RATIO:
        detail["reason"]="surface depression detected"; return False,detail
    if surface_ratio<MIN_SURFACE_RATIO:
        detail["reason"]="surface is not sufficiently planar"; return False,detail
    if p95>max(SURFACE_MAX_P95_TOLERANCE,adaptive_tol*P95_TOLERANCE_MULTIPLIER):
        detail["reason"]="surface is uneven"; return False,detail
    detail["reason"]="free and planar"
    return True,detail



def _check_free_space(pbbox, ref, color_shape, depth, depth_scale, K):
    """
    Validate candidate placement:
      1. Estimate the support plane from the area around the candidate.
      2. Confirm the original object and candidate are on the same surface.
      3. Check the candidate footprint for raised/depressed obstructions.
    """
    ch, cw = color_shape[:2]
    dh, dw = depth.shape

    fx, fy, _, _ = _resolve_k(K, depth.shape)
    sx = (dw - 1) / max(cw - 1, 1)
    sy = (dh - 1) / max(ch - 1, 1)

    target_depth = max(float(ref["depth"]), MIN_DEPTH)
    buffer_px = PLACEMENT_BUFFER * fx / target_depth

    bbox_width_px = max(1.0, (pbbox[2] - pbbox[0]) * sx)
    half_width = bbox_width_px / 2.0 + buffer_px
    half_height = bbox_width_px * FOOTPRINT_DEPTH_RATIO / 2.0 + buffer_px

    center_x = ((pbbox[0] + pbbox[2]) / 2.0) * sx
    base_y = pbbox[3] * sy

    fx1 = int(round(center_x - half_width))
    fy1 = int(round(base_y - half_height))
    fx2 = int(round(center_x + half_width))
    fy2 = int(round(base_y + half_height))

    rect_rgb = (
        int(fx1 / sx), int(fy1 / sy),
        int(fx2 / sx), int(fy2 / sy),
    )

    if fx1 < 0 or fy1 < 0 or fx2 > dw or fy2 > dh:
        return False, "place location is not on same surface", rect_rgb

    # The candidate footprint is excluded from plane estimation.
    placement_plane, plane_region = _estimate_placement_plane(
        fx1, fy1, fx2, fy2,
        depth, depth_scale, K,
        ref.get("dets", []), color_shape,
    )

    # If we cannot determine the placement surface depth, explicitly reject it.
    if placement_plane is None:
        return False, "place location is not on same surface", rect_rgb

    # NEW: compare original object's support point with the candidate plane.
    same_surface, surface_message = _check_original_and_placement_surface(
        ref, placement_plane
    )

    if not same_surface:
        return False, surface_message, rect_rgb

    pad = int(max(
        (fx2 - fx1) * ADAPTIVE_CONTEXT_RATIO,
        ADAPTIVE_CONTEXT_MIN_PX,
    ))

    context = (
        max(0, fx1 - pad), max(0, fy1 - pad),
        min(dw, fx2 + pad), min(dh, fy2 + pad),
    )

    ok, detail = _scan_local_surface(
        fx1, fy1, fx2, fy2,
        depth, depth_scale, K,
        placement_plane, context,
    )

    if isinstance(detail, dict) and plane_region is not None:
        detail["support_plane_region"] = [int(v) for v in plane_region]

    if not ok:
        if isinstance(detail, dict):
            reason = detail.get("reason", "place location is not on same surface")
            if "adaptive_tolerance" in detail:
                reason += (
                    f" | tol={detail['adaptive_tolerance']:.2f}"
                    f"{MEASUREMENT_UNIT}"
                )
            if "surface_ratio" in detail:
                reason += f" | planar={detail['surface_ratio']:.1%}"
            if "raised_ratio" in detail:
                reason += f" | raised={detail['raised_ratio']:.1%}"
            if "p95_error" in detail:
                reason += (
                    f" | P95={detail['p95_error']:.2f}"
                    f"{MEASUREMENT_UNIT}"
                )
            return False, reason, rect_rgb
        return False, str(detail), rect_rgb

    return True, (
        f"free and planar | {surface_message} | "
        f"planar={detail['surface_ratio']:.1%} | "
        f"tol={detail['adaptive_tolerance']:.2f}{MEASUREMENT_UNIT} | "
        f"P95={detail['p95_error']:.2f}{MEASUREMENT_UNIT}"
    ), rect_rgb

def _evaluate_slot(slot_xyz, ref, dets, selected_index, color, depth, depth_scale, K):

    """Validate a candidate placement location.

    ref = {"bbox_wh": (w, h) px, "base": 3D base point or None, "depth": target depth,

           "plane": support plane or None}.

    reason: None (ok) | "frame" | "surface" | "obstacle" | "space".

    "all_obstacles" lists EVERY detected obstacle hit by the slot (blocking or safe)."""

    pixel = _point_3d_to_rgb_pixel(slot_xyz, color, depth, K)

    res = {"ok": False, "reason": "frame", "obstacle": None, "bbox": None,

           "pixel": pixel, "all_obstacles": [], "footprint": None, "space_detail": None}

    if pixel is None:

        return res

    bbox = _placement_bbox(pixel, *ref["bbox_wh"])

    res["bbox"] = bbox



    pcx, pcy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2

    best, best_score = None, float("inf")

    for det in dets:

        if det["index"] == selected_index:

            continue

        x1, y1, x2, y2 = det["bbox"]

        if ENABLE_BBOX_OVERLAP:

            hit = _bbox_overlap(bbox, det["bbox"])

        else:

            hit = x1 <= pixel[0] <= x2 and y1 <= pixel[1] <= y2

        if not hit:

            continue

        oz = _median_bbox_depth(det["bbox"], color, depth, depth_scale)

        diff = abs(oz - ref["depth"]) if oz is not None else None

        # Unknown obstacle depth is treated conservatively as blocking.

        safe = (ENABLE_OBSTACLE_DEPTH_AVOIDANCE and diff is not None

                and diff > OBSTACLE_DEPTH_SAFETY_THRESHOLD)

        ob = dict(det)

        ob.update(depth=oz, depth_diff=diff,

                  blocking=bool(ENABLE_OBSTACLE_CHECK and not safe))

        res["all_obstacles"].append(ob)

        if ob["blocking"]:

            score = ((x1 + x2) / 2 - pcx) ** 2 + ((y1 + y2) / 2 - pcy) ** 2

            if score < best_score:

                best_score, best = score, ob



    # decision: legacy surface check, detected obstacles, then depth-based free space

    if ENABLE_SURFACE_VALIDATION and not _same_surface(

            ref.get("base"), bbox, color, depth, depth_scale):

        res["reason"] = "surface"

        return res

    if best is not None:

        res["reason"], res["obstacle"] = "obstacle", best

        return res

    if ENABLE_FREE_SPACE_CHECK:

        ok, detail, footprint = _check_free_space(bbox, ref, color.shape, depth,

                                                  depth_scale, K)

        res["footprint"], res["space_detail"] = footprint, detail

        if not ok:

            res["reason"] = "space"

            return res

    res.update(ok=True, reason=None)

    return res





def _block_text(ev, tag, first):

    if ev["reason"] == "obstacle":

        n = ev["obstacle"].get("class_name", "object")

        return f"There is {n} at {tag}" if first else f"{n} at {tag}"

    if ev["reason"] == "space":

        return f"{tag} has no free surface space ({ev['space_detail']})"

    return f"{tag} is not on the same surface"





def _choose_placement(target_xyz, right_xyz, ref, dets, selected_index,

                      color, depth, depth_scale, K):

    """+20 cm right of object, then +40 cm, else fall back to original XYZ

    (offsets are RIGHT_OFFSET, in MEASUREMENT_UNIT)."""

    p20_xyz = (right_xyz[0] + RIGHT_OFFSET, right_xyz[1], right_xyz[2])

    p40_xyz = (right_xyz[0] + 2 * RIGHT_OFFSET, right_xyz[1], right_xyz[2])

    p20 = _point_3d_to_rgb_pixel(p20_xyz, color, depth, K)

    p40 = _point_3d_to_rgb_pixel(p40_xyz, color, depth, K)

    h, w = color.shape[:2]

    p20_in, p40_in = _in_frame(p20, w, h), _in_frame(p40, w, h)



    ev20 = (_evaluate_slot(p20_xyz, ref, dets, selected_index, color, depth, depth_scale, K)

            if p20_in else None)

    ev40 = None

    if ev20 is not None and not ev20["ok"] and p40_in:

        ev40 = _evaluate_slot(p40_xyz, ref, dets, selected_index, color, depth, depth_scale, K)



    base = {"p20_pixel": p20, "p40_pixel": p40, "p40_in": p40_in,

            "obstacle20": ev20["obstacle"] if ev20 else None,

            "obstacle40": ev40["obstacle"] if ev40 else None,

            "bbox20": ev20["bbox"] if ev20 else None,

            "bbox40": ev40["bbox"] if ev40 else None, "bbox": None,

            "footprint20": ev20["footprint"] if ev20 else None,

            "footprint40": ev40["footprint"] if ev40 else None,

            "all_obstacles": ([dict(o, slot="+20") for o in ev20["all_obstacles"]] if ev20 else [])

                             + ([dict(o, slot="+40") for o in ev40["all_obstacles"]] if ev40 else [])}



    if not p20_in:

        return {**base, "xyz": target_xyz, "pixel": None,

                "message": "20cm is out of frame, using original object position"}

    if ev20["ok"]:

        return {**base, "xyz": p20_xyz, "pixel": p20, "bbox": ev20["bbox"],

                "message": "Free to keep object at +20cm"}



    d20 = _block_text(ev20, "+20cm", True)

    if not p40_in:

        return {**base, "xyz": target_xyz, "pixel": None,

                "message": f"{d20}, +40cm is out of frame, blocked"}

    if ev40["ok"]:

        return {**base, "xyz": p40_xyz, "pixel": p40, "bbox": ev40["bbox"],

                "message": f"{d20}, +40cm is free to keep object"}



    d40 = _block_text(ev40, "+40cm", False)

    return {**base, "xyz": target_xyz, "pixel": None,

            "message": f"{d20} and {d40}, no place to keep object"}





# ---------------------------- ENGINE ---------------------------------

class PerceptionEngine:

    """Holds YOLO model + tracking/grab state. One instance per camera."""



    def __init__(self, model_path=MODEL_PATH, conf=CONFIDENCE_THRESHOLD,

                 iou=IOU_THRESHOLD, tracker=TRACKER, imgsz=YOLO_IMAGE_SIZE,

                 device=DEVICE):

        self.warnings = []             # non-fatal issues, de-duplicated, for the caller

        self.model = YOLO(model_path)

        self.excluded = {str(n).strip().lower() for n in EXCLUDED_CLASSES if str(n).strip()}

        self.allowed_class_ids = self._build_allowed_class_ids()

        self.conf, self.iou, self.tracker = float(conf), float(iou), tracker

        self.imgsz, self.device = int(imgsz), device

        self.target = None

        self.grab_state = 0

        self.locked_id = None

        self.missing = 0

        self.original_pos = None

        self.original_right = None     # frozen right-edge XYZ at grab

        self.original_ref = None       # frozen bbox size / base point / depth / plane at grab

        self.target_pos_fixed = None   # frozen placement XYZ at grab

        self.target_is_slot = False    # True if fixed target is a real +20/+40 slot

        self.latest_color = None

        self.latest_depth = None

        self.depth_scale = 1.0

        self.intrinsics = None

        self._k_loaded = False



    def _warn(self, msg):

        if msg not in self.warnings:

            self.warnings.append(msg)



    def _build_allowed_class_ids(self):

        """Class ids YOLO may return (None = no restriction)."""

        if not self.excluded:

            return None

        names = getattr(self.model, "names", None)

        if not isinstance(names, dict) or not names:

            return None

        known = {str(n).lower() for n in names.values()}

        unknown = sorted(self.excluded - known)

        if unknown:

            self._warn(f"EXCLUDED_CLASSES not in model classes (ignored): {unknown}")

        allowed = [int(i) for i, n in names.items() if str(n).lower() not in self.excluded]

        return allowed or None



    # -- raw frame ingestion --

    def ingest(self, frames, depth_profile=None):

        if depth_profile is not None and not self._k_loaded:

            self.intrinsics = load_intrinsics(depth_profile)

            self._k_loaded = True

            if self.intrinsics is None:

                self._warn("camera intrinsics unavailable; using approximate intrinsics")

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

            self._warn(f"frame error: {type(e).__name__}: {e}")



    def process(self, frames, target_object=None, grab_state=None, depth_profile=None):

        """Raw frames in -> (payload_dict, details_dict)."""

        self.ingest(frames, depth_profile)

        payload, details = self.process_arrays(self.latest_color, self.latest_depth,

                                               target_object, grab_state,

                                               self.depth_scale, self.intrinsics)

        details.update({"color": self.latest_color, "payload": payload,

                        "target": self.target, "grab": self.grab_state})

        return payload, details



    # -- state helpers --

    def set_target(self, name):

        name = name.strip().lower() if name else None

        if name == self.target:

            return

        self.target = name

        if name and name in self.excluded:

            self._warn(f"target '{name}' is in EXCLUDED_CLASSES and will never be detected")

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

        self.original_ref = None

        self.target_pos_fixed = None

        self.target_is_slot = False



    def _grab_placement(self, original, ref_right, ref, dets, idx, color, depth,

                        depth_scale, K):

        """After grab: keep the fixed target while it stays free; if it becomes

        obstructed (or none was fixed yet), run the base +20/+40/original logic

        from the frozen original position and fix the new result."""

        h, w = color.shape[:2]

        replanned = False

        if self.target_pos_fixed is not None and self.target_is_slot:

            pix = _point_3d_to_rgb_pixel(self.target_pos_fixed, color, depth, K)

            ev = (_evaluate_slot(self.target_pos_fixed, ref, dets, idx, color, depth,

                                 depth_scale, K) if _in_frame(pix, w, h) else None)

            if ev is None or ev["ok"]:

                return {"xyz": self.target_pos_fixed, "pixel": pix,

                        "p20_pixel": None, "p40_pixel": None, "p40_in": True,

                        "obstacle20": None, "obstacle40": None,

                        "bbox": ev["bbox"] if ev else None,

                        "footprint": ev["footprint"] if ev else None,

                        "all_obstacles": ([dict(o, slot="fixed") for o in ev["all_obstacles"]]

                                          if ev else []),

                        "message": "Fixed target position is free"}, "fixed"

            replanned = True



        pl = _choose_placement(original, ref_right, ref, dets, idx, color, depth,

                               depth_scale, K)

        self.target_pos_fixed = pl["xyz"]

        self.target_is_slot = pl["xyz"] != original

        return pl, ("replanned" if replanned else "planned")



    # -- main entry --

    def process_arrays(self, color, depth, target_object=None, grab_state=None,

                       depth_scale=1.0, intrinsics=None):

        """Already-converted arrays in -> (payload_dict, details_dict)."""

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

                                  classes=self.allowed_class_ids,

                                  verbose=False)[0]



        dets, targets = [], []

        boxes = result.boxes

        if boxes is not None and len(boxes):

            for i in range(len(boxes)):

                name = str(result.names[int(boxes.cls[i].item())])

                if name.lower() in self.excluded:

                    continue

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



        # base point = bottom-centre of bbox, with its own robust depth (None if invalid)

        base_z = _sample_point_depth(cx_px, y2, color.shape, depth, depth_scale)

        base_pt = _pixel_to_3d(cx_px, y2, base_z, color, depth, intrinsics)

        plane = (_estimate_support_plane(bbox, dets, color.shape, depth, depth_scale,

                                         intrinsics)

                 if ENABLE_FREE_SPACE_CHECK else None)

        ref = {"bbox_wh": (x2 - x1, y2 - y1), "base": base_pt,
               "depth": median_depth, "plane": plane, "dets": dets}



        # ---- grab / original_pos / target_pos ----

        if self.grab_state == 0:

            original = tuple(center)

            placement = _choose_placement(original, right, ref, dets, selected["index"],

                                          color, depth, depth_scale, intrinsics)

        else:

            if self.original_pos is None:      # freeze at first valid grab frame

                self.original_pos = tuple(float(v) for v in center)

                self.original_right = tuple(float(v) for v in right)

                self.original_ref = ref

            original = self.original_pos

            placement, _ = self._grab_placement(original, self.original_right,

                                                self.original_ref, dets,

                                                selected["index"], color, depth,

                                                depth_scale, intrinsics)

        target_xyz = placement["xyz"]

        desc = placement.get("message", "Object detected; position is updating dynamically")



        dbg["selected"] = {

            "bbox": bbox, "center_px": (cx_px, cy_px), "confidence": conf,

            "track_id": track_id, "center": center, "left": left, "right": right,

            "base": base_pt, "target": target_xyz, "placement": placement,

        }

        payload = build_nested_json(True, self.target, track_id, conf, center, left,

                                    right, original, target_xyz, desc)

        return payload, dbg





# ------------------------- PUBLIC ENTRY POINT -------------------------

_engine = None





def main(frames, target_object=None, grab_state=0, depth_profile=None,

         return_details=False, **engine_kwargs):

    """

    Raw camera frames in -> JSON string out.



    engine_kwargs (first call only): model_path, conf, iou, tracker, imgsz, device

    return_details=True -> returns (json_str, details_dict) for callers that render/log.

    """

    global _engine

    if _engine is None:

        _engine = PerceptionEngine(**engine_kwargs)

    payload, details = _engine.process(frames, target_object, grab_state, depth_profile)

    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)

    return (encoded, details) if return_details else encoded





def get_engine():

    return _engine