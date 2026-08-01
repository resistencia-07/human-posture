import os

import cv2

from config import CAMERA_INDEX, FRAME_HEIGHT, FRAME_WIDTH, FULLSCREEN, WINDOW_NAME


def _get_device_name(index: int) -> str:
    """Read the Linux V4L device name for a camera index if available."""
    try:
        with open(f"/sys/class/video4linux/video{index}/name", "r", encoding="utf-8") as handle:
            return handle.read().strip().lower()
    except OSError:
        return ""


def _is_usb_device(index: int) -> bool:
    """Check whether a Linux video device appears to be attached over USB."""
    try:
        device_path = os.path.realpath(f"/sys/class/video4linux/video{index}/device")
        return "usb" in device_path.lower()
    except OSError:
        return False


def _camera_priority(index: int) -> tuple[int, int, int]:
    """Return a ranking tuple to prefer external USB cameras over built-in ones."""
    device_name = _get_device_name(index)
    is_usb = _is_usb_device(index)
    is_builtin = "hp hd camera" in device_name or "built-in" in device_name
    return (0 if is_usb else 1, 0 if not is_builtin else 1, index)


def _find_camera_index():
    """Probe common camera indices and prefer a connected USB camera when available."""
    candidate_indices = list(range(0, 10))
    if CAMERA_INDEX is not None:
        candidate_indices = [int(CAMERA_INDEX)] + [idx for idx in candidate_indices if idx != int(CAMERA_INDEX)]

    ordered_indices = sorted(candidate_indices, key=_camera_priority)

    for index in ordered_indices:
        for api in (cv2.CAP_ANY, cv2.CAP_V4L2):
            capture = cv2.VideoCapture(index, api)
            if capture.isOpened():
                ok, frame = capture.read()
                if ok and frame is not None and frame.size > 0:
                    device_name = _get_device_name(index) or "unknown"
                    print(f"Selecting camera index {index}: {device_name}")
                    return capture, index
                capture.release()

    raise RuntimeError("Unable to open the webcam. Please connect a USB camera and try again.")


def open_camera():
    """Open the webcam and configure it for full-resolution capture."""
    capture, _ = _find_camera_index()

    capture.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return capture


def configure_window() -> None:
    """Create the display window and enable full-screen mode when possible."""
    try:
        cv2.namedWindow(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN)
        if FULLSCREEN:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
    except cv2.error:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)