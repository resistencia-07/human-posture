import os

import cv2
from mediapipe.tasks import python
from mediapipe.tasks.python.vision.core import image as mp_image
from mediapipe.tasks.python import vision

from config import MODEL_COMPLEXITY, MIN_DETECTION_CONFIDENCE, MIN_TRACKING_CONFIDENCE, SMOOTH_LANDMARKS


class PoseDetector:
    """Wrap a MediaPipe pose landmarker so the main application can use it cleanly."""

    def __init__(self) -> None:
        """Create the MediaPipe pose landmarker model with tuned defaults."""
        model_path = os.path.join(os.path.dirname(__file__), "models", "pose_landmarker_heavy.task")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Pose model not found at {model_path}")
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            output_segmentation_masks=True,
            num_poses=1,
            min_pose_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_pose_presence_confidence=MIN_TRACKING_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        )
        self.landmarker = vision.PoseLandmarker.create_from_options(options)

    def detect(self, frame):
        """Run pose estimation on a BGR OpenCV frame and return the results."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = mp_image.Image(image_format=mp_image.ImageFormat.SRGB, data=rgb_frame)
        return self.landmarker.detect_for_video(image, int(cv2.getTickCount()))

    def extract_landmarks(self, results):
        """Return normalized landmarks for the single detected person, or None."""
        if not results or not results.pose_landmarks:
            return None
        return results.pose_landmarks[0]

    def close(self) -> None:
        """Release the underlying MediaPipe landmarker resources safely."""
        if hasattr(self, "landmarker") and self.landmarker is not None:
            try:
                self.landmarker.close()
            except Exception:
                pass

    def __del__(self) -> None:
        """Ensure the landmarker is closed when the detector is garbage-collected."""
        self.close()

    def fps(self, frame_count: int, start_time: int) -> float:
        """Return the current frames-per-second estimate for the stream."""
        elapsed_seconds = max((cv2.getTickCount() - start_time) / cv2.getTickFrequency(), 1e-6)
        return frame_count / elapsed_seconds