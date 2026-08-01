import sys

import cv2

from camera import configure_window, open_camera
from config import WINDOW_NAME
from drawing import draw_posture_dashboard, draw_pose_landmarks
from metrics import evaluate_posture
from pose_detector import PoseDetector


def main() -> int:
    """Run the posture analysis loop until the user closes the window."""
    detector = PoseDetector()

    try:
        capture = open_camera()
    except RuntimeError as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1

    configure_window()

    frame_count = 0
    start_time = cv2.getTickCount()

    while True:
        success, frame = capture.read()
        if not success or frame is None:
            print("Unable to read frames from the camera.", file=sys.stderr)
            break

        frame_count += 1
        results = detector.detect(frame)
        analysis = evaluate_posture(detector.extract_landmarks(results))

        draw_pose_landmarks(frame, results)
        draw_posture_dashboard(frame, analysis, results, detector.fps(frame_count, start_time))

        try:
            cv2.imshow(WINDOW_NAME, frame)
        except cv2.error:
            print("Display is not available, so the application will exit gracefully.")
            break

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())