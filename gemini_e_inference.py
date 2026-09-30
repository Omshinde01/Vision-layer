"""
Orbbec Gemini E - RGB + Depth + 3D Position
--------------------------------------------

Features:
    - Live RGB camera
    - Left-click any RGB pixel
    - Finds corresponding depth pixel
    - Reads depth
    - Calculates approximate 3D position X, Y, Z
      relative to the camera
    - Displays the result on screen

SDK:
    PyOrbbecSDK / Orbbec SDK v1.x

Controls:
    Left click : Select point
    Q / ESC    : Quit
"""

import cv2
import numpy as np

from pyorbbecsdk import Pipeline, Config, OBSensorType


WINDOW = "Gemini E - RGB + Depth + 3D"

selected_x = None
selected_y = None

latest_depth = None
latest_color = None

depth_scale = 1.0


# ============================================================
# DEPTH
# ============================================================

def depth_to_numpy(frame):
    global depth_scale

    if frame is None:
        return None

    # Depth scale supplied by camera
    depth_scale = float(frame.get_depth_scale())

    width = frame.get_width()
    height = frame.get_height()

    data = np.asarray(frame.get_data())

    # 16-bit depth stored as bytes
    if data.dtype == np.uint8:

        expected = width * height * 2

        if data.size != expected:
            raise RuntimeError(
                f"Unexpected depth byte count: {data.size}, "
                f"expected {expected}"
            )

        depth = data.view(np.uint16).reshape(
            height,
            width
        )

    else:

        depth = data.astype(
            np.uint16,
            copy=False
        ).reshape(
            height,
            width
        )

    return np.ascontiguousarray(depth)


# ============================================================
# RGB
# ============================================================

def color_to_bgr(frame):

    if frame is None:
        return None

    width = frame.get_width()
    height = frame.get_height()

    fmt = str(frame.get_format()).upper()

    data = np.asarray(frame.get_data())

    # --------------------------------------------------------
    # RGB888
    # --------------------------------------------------------

    if (
        data.dtype == np.uint8
        and data.size == width * height * 3
    ):

        image = data.reshape(
            height,
            width,
            3
        ).copy()

        if "RGB" in fmt and "BGR" not in fmt:

            image = cv2.cvtColor(
                image,
                cv2.COLOR_RGB2BGR
            )

        return np.ascontiguousarray(image)

    # --------------------------------------------------------
    # YUYV
    # --------------------------------------------------------

    if (
        data.dtype == np.uint8
        and data.size == width * height * 2
    ):

        yuyv = data.reshape(
            height,
            width,
            2
        )

        return cv2.cvtColor(
            yuyv,
            cv2.COLOR_YUV2BGR_YUY2
        )

    # --------------------------------------------------------
    # MJPEG
    # --------------------------------------------------------

    if data.dtype == np.uint8 and data.size > 0:

        decoded = cv2.imdecode(
            data.reshape(-1, 1),
            cv2.IMREAD_COLOR
        )

        if decoded is not None:
            return decoded

    return None


# ============================================================
# RGB PIXEL -> DEPTH PIXEL
# ============================================================

def rgb_to_depth_pixel(x, y):

    if latest_depth is None:
        return None

    dh, dw = latest_depth.shape

    if latest_color is not None:

        ch, cw = latest_color.shape[:2]

    else:

        cw = dw
        ch = dh

    # --------------------------------------------------------
    # Resolution-based mapping
    #
    # This is intentionally kept from your working script.
    # It is NOT a calibrated RGB-depth transformation.
    # --------------------------------------------------------

    dx = int(
        round(
            x * (dw - 1) /
            max(cw - 1, 1)
        )
    )

    dy = int(
        round(
            y * (dh - 1) /
            max(ch - 1, 1)
        )
    )

    dx = max(
        0,
        min(dw - 1, dx)
    )

    dy = max(
        0,
        min(dh - 1, dy)
    )

    return dx, dy


# ============================================================
# DEPTH AT RGB PIXEL
# ============================================================

def get_depth_at_rgb_pixel(x, y):

    result = rgb_to_depth_pixel(
        x,
        y
    )

    if result is None:
        return None

    dx, dy = result

    raw_depth = int(
        latest_depth[dy, dx]
    )

    if raw_depth == 0:

        return {
            "rgb_x": x,
            "rgb_y": y,
            "depth_x": dx,
            "depth_y": dy,
            "raw": 0,
            "depth_m": None
        }

    # Camera depth scale
    depth_m = raw_depth * depth_scale / 1000.0

    return {
        "rgb_x": x,
        "rgb_y": y,
        "depth_x": dx,
        "depth_y": dy,
        "raw": raw_depth,
        "depth_m": depth_m
    }


# ============================================================
# 3D POSITION
# ============================================================

def calculate_3d_position(
    depth_x,
    depth_y,
    depth_m,
    depth_width,
    depth_height
):
    """
    Calculate approximate camera-coordinate XYZ.

    IMPORTANT:
    This requires camera intrinsics for accurate results.

    For now we use approximate focal lengths and principal
    point based on the depth image size.

    Coordinate convention:

        X = right
        Y = down
        Z = forward

    Units:
        meters
    """

    # --------------------------------------------------------
    # APPROXIMATE INTRINSICS
    #
    # These are placeholders until we read the actual
    # Gemini E calibration parameters.
    # --------------------------------------------------------

    fx = depth_width * 0.9
    fy = depth_height * 0.9

    cx = depth_width / 2.0
    cy = depth_height / 2.0

    # --------------------------------------------------------
    # Pinhole camera model
    # --------------------------------------------------------

    X = (
        (depth_x - cx)
        * depth_m
        / fx
    )

    Y = (
        (depth_y - cy)
        * depth_m
        / fy
    )

    Z = depth_m

    return X, Y, Z


# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global selected_x
    global selected_y

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    selected_x = x
    selected_y = y

    result = get_depth_at_rgb_pixel(
        x,
        y
    )

    if result is None:

        print(
            f"RGB ({x}, {y}) -> "
            f"No depth frame"
        )

        return

    dx = result["depth_x"]
    dy = result["depth_y"]

    raw = result["raw"]

    depth_m = result["depth_m"]

    if depth_m is None:

        print(
            f"RGB ({x}, {y}) -> "
            f"Depth ({dx}, {dy}) -> "
            f"INVALID"
        )

        return

    # --------------------------------------------------------
    # Calculate XYZ
    # --------------------------------------------------------

    dh, dw = latest_depth.shape

    X, Y, Z = calculate_3d_position(
        dx,
        dy,
        depth_m,
        dw,
        dh
    )

    print("\n----------------------------------------")

    print(
        f"RGB pixel    : ({x}, {y})"
    )

    print(
        f"Depth pixel  : ({dx}, {dy})"
    )

    print(
        f"Raw depth    : {raw}"
    )

    print(
        f"Distance     : {depth_m:.3f} m"
    )

    print(
        f"Camera XYZ   : "
        f"X={X:.3f} m, "
        f"Y={Y:.3f} m, "
        f"Z={Z:.3f} m"
    )

    print("----------------------------------------")


# ============================================================
# MAIN
# ============================================================

def main():

    global latest_depth
    global latest_color

    pipeline = Pipeline()
    config = Config()

    try:

        # ====================================================
        # GET DEPTH PROFILES
        # ====================================================

        depth_profiles = (
            pipeline.get_stream_profile_list(
                OBSensorType.DEPTH_SENSOR
            )
        )

        if (
            depth_profiles is None
            or depth_profiles.get_count() == 0
        ):

            raise RuntimeError(
                "No depth profiles found."
            )

        # ====================================================
        # GET COLOR PROFILES
        # ====================================================

        color_profiles = (
            pipeline.get_stream_profile_list(
                OBSensorType.COLOR_SENSOR
            )
        )

        if (
            color_profiles is None
            or color_profiles.get_count() == 0
        ):

            raise RuntimeError(
                "No color profiles found."
            )

        # ====================================================
        # USE DEFAULT PROFILES
        #
        # This is the same approach as your working script.
        # ====================================================

        depth_profile = (
            depth_profiles
            .get_default_video_stream_profile()
        )

        color_profile = (
            color_profiles
            .get_default_video_stream_profile()
        )

        print("\n========================================")
        print("Gemini E RGB + Depth")
        print("========================================")

        print("\nDepth profile:")
        print(depth_profile)

        print("\nColor profile:")
        print(color_profile)

        # ====================================================
        # ENABLE STREAMS
        # ====================================================

        config.enable_stream(
            depth_profile
        )

        config.enable_stream(
            color_profile
        )

        # Gemini E does not support hardware frame sync.
        # Therefore we DO NOT enable frame synchronization.

        print(
            "\nStarting RGB + Depth..."
        )

        pipeline.start(config)

        print(
            "Camera started successfully."
        )

        # ====================================================
        # OPENCV WINDOW
        # ====================================================

        cv2.namedWindow(
            WINDOW,
            cv2.WINDOW_NORMAL
        )

        cv2.resizeWindow(
            WINDOW,
            1280,
            720
        )

        cv2.setMouseCallback(
            WINDOW,
            mouse_callback
        )

        print("\n========================================")
        print(" CAMERA RUNNING")
        print("========================================")
        print("LEFT CLICK : Get 3D position")
        print("Q / ESC    : Quit")
        print("========================================\n")

        # ====================================================
        # LOOP
        # ====================================================

        while True:

            frames = pipeline.wait_for_frames(
                1000
            )

            if frames is None:
                continue

            # ------------------------------------------------
            # DEPTH
            # ------------------------------------------------

            depth_frame = (
                frames.get_depth_frame()
            )

            if depth_frame is not None:

                latest_depth = (
                    depth_to_numpy(
                        depth_frame
                    )
                )

            # ------------------------------------------------
            # RGB
            # ------------------------------------------------

            color_frame = (
                frames.get_color_frame()
            )

            if color_frame is not None:

                color = color_to_bgr(
                    color_frame
                )

                if color is not None:

                    latest_color = color

            # ------------------------------------------------
            # Nothing to display yet
            # ------------------------------------------------

            if latest_color is None:
                continue

            # ------------------------------------------------
            # DISPLAY
            # ------------------------------------------------

            display = latest_color.copy()

            # ------------------------------------------------
            # Selected point
            # ------------------------------------------------

            if (
                selected_x is not None
                and selected_y is not None
            ):

                x = selected_x
                y = selected_y

                # Draw cross
                cv2.drawMarker(
                    display,
                    (x, y),
                    (0, 255, 255),
                    cv2.MARKER_CROSS,
                    30,
                    2
                )

                result = (
                    get_depth_at_rgb_pixel(
                        x,
                        y
                    )
                )

                if result is None:

                    text = (
                        f"RGB ({x},{y}) | "
                        f"No depth"
                    )

                else:

                    dx = result["depth_x"]
                    dy = result["depth_y"]

                    depth_m = result["depth_m"]

                    if depth_m is None:

                        text = (
                            f"RGB ({x},{y}) | "
                            f"Depth ({dx},{dy}) | "
                            f"INVALID"
                        )

                    else:

                        dh, dw = (
                            latest_depth.shape
                        )

                        X, Y, Z = (
                            calculate_3d_position(
                                dx,
                                dy,
                                depth_m,
                                dw,
                                dh
                            )
                        )

                        text1 = (
                            f"RGB: ({x},{y})  "
                            f"Depth: ({dx},{dy})"
                        )

                        text2 = (
                            f"Distance: "
                            f"{depth_m:.3f} m"
                        )

                        text3 = (
                            f"XYZ: "
                            f"X={X:.3f} "
                            f"Y={Y:.3f} "
                            f"Z={Z:.3f} m"
                        )

                        # Background
                        cv2.rectangle(
                            display,
                            (5, 5),
                            (620, 105),
                            (0, 0, 0),
                            -1
                        )

                        cv2.putText(
                            display,
                            text1,
                            (12, 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

                        cv2.putText(
                            display,
                            text2,
                            (12, 58),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (255, 255, 255),
                            1,
                            cv2.LINE_AA
                        )

                        cv2.putText(
                            display,
                            text3,
                            (12, 86),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.55,
                            (0, 255, 255),
                            2,
                            cv2.LINE_AA
                        )

                        # Skip generic text below
                        text = None

                if text is not None:

                    cv2.rectangle(
                        display,
                        (5, 5),
                        (620, 50),
                        (0, 0, 0),
                        -1
                    )

                    cv2.putText(
                        display,
                        text,
                        (12, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                        cv2.LINE_AA
                    )

            # ------------------------------------------------
            # SHOW
            # ------------------------------------------------

            cv2.imshow(
                WINDOW,
                display
            )

            key = cv2.waitKey(1) & 0xFF

            if key in (
                ord("q"),
                27
            ):
                break

    except KeyboardInterrupt:

        print(
            "\nStopped by user."
        )

    except Exception as e:

        print("\n========================================")
        print("ERROR")
        print("========================================")
        print(
            type(e).__name__,
            ":",
            str(e)
        )

        import traceback

        traceback.print_exc()

    finally:

        cv2.destroyAllWindows()

        try:
            pipeline.stop()
        except Exception:
            pass

        print(
            "\nCamera stopped."
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()