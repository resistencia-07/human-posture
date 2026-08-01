import math
from typing import Optional, Sequence, Tuple


Point = Tuple[float, float, float]


def _get_landmark(landmarks: Sequence[object], index: int) -> Optional[Point]:
    """Return a normalized 3D landmark point for the requested index."""
    if landmarks is None or index >= len(landmarks):
        return None
    landmark = landmarks[index]
    return (landmark.x, landmark.y, landmark.z)


def _midpoint(point_a: Optional[Point], point_b: Optional[Point]) -> Optional[Point]:
    """Return the midpoint between two landmarks when both points are present."""
    if point_a is None or point_b is None:
        return None
    return ((point_a[0] + point_b[0]) / 2.0, (point_a[1] + point_b[1]) / 2.0, (point_a[2] + point_b[2]) / 2.0)


def calculate_angle(point_a: Point, point_b: Point, point_c: Point) -> float:
    """Compute the interior angle in degrees for three landmarks."""
    vector_ba = (point_a[0] - point_b[0], point_a[1] - point_b[1], point_a[2] - point_b[2])
    vector_bc = (point_c[0] - point_b[0], point_c[1] - point_b[1], point_c[2] - point_b[2])

    magnitude_ba = math.sqrt(vector_ba[0] ** 2 + vector_ba[1] ** 2 + vector_ba[2] ** 2)
    magnitude_bc = math.sqrt(vector_bc[0] ** 2 + vector_bc[1] ** 2 + vector_bc[2] ** 2)

    if magnitude_ba < 1e-6 or magnitude_bc < 1e-6:
        return 0.0

    cosine = (vector_ba[0] * vector_bc[0] + vector_ba[1] * vector_bc[1] + vector_ba[2] * vector_bc[2]) / (magnitude_ba * magnitude_bc)
    cosine = max(-1.0, min(1.0, cosine))
    return abs(math.degrees(math.acos(cosine)))


def _angle_to_vertical(point_a: Optional[Point], point_b: Optional[Point]) -> float:
    """Measure how far a line deviates from the vertical axis in degrees."""
    if point_a is None or point_b is None:
        return 0.0
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return abs(math.degrees(math.atan2(abs(dx), abs(dy))))


def _angle_to_horizontal(point_a: Optional[Point], point_b: Optional[Point]) -> float:
    """Measure how far a line deviates from the horizontal axis in degrees."""
    if point_a is None or point_b is None:
        return 0.0
    dx = point_b[0] - point_a[0]
    dy = point_b[1] - point_a[1]
    if abs(dx) < 1e-6 and abs(dy) < 1e-6:
        return 0.0
    return abs(math.degrees(math.atan2(abs(dy), abs(dx))))


def analyze_posture(landmarks: Sequence[object]) -> dict:
    """Return front-view, side-view, and overall posture metrics for the detected person."""
    left_shoulder = _get_landmark(landmarks, 11)
    right_shoulder = _get_landmark(landmarks, 12)
    left_hip = _get_landmark(landmarks, 23)
    right_hip = _get_landmark(landmarks, 24)
    left_knee = _get_landmark(landmarks, 25)
    right_knee = _get_landmark(landmarks, 26)
    left_ear = _get_landmark(landmarks, 7)
    right_ear = _get_landmark(landmarks, 8)
    nose = _get_landmark(landmarks, 0)

    if None in {left_shoulder, right_shoulder, left_hip, right_hip, left_knee, right_knee, left_ear, right_ear, nose}:
        return {}

    shoulder_mid = _midpoint(left_shoulder, right_shoulder)
    hip_mid = _midpoint(left_hip, right_hip)
    ear_mid = _midpoint(left_ear, right_ear)

    front_metrics = {
        "Shoulder Alignment": _angle_to_horizontal(left_shoulder, right_shoulder),
        "Neck Angle": calculate_angle(left_shoulder, shoulder_mid, ear_mid),
        "Head Tilt": _angle_to_vertical(nose, ear_mid),
        "Spine Inclination": _angle_to_vertical(hip_mid, shoulder_mid),
        "Pelvic Alignment": _angle_to_horizontal(left_hip, right_hip),
        "Knee Alignment": _angle_to_horizontal(left_knee, right_knee),
        "Hip Shift": _angle_to_vertical(hip_mid, shoulder_mid),
    }

    side_metrics = {
        "Shoulder Alignment": _angle_to_horizontal((left_shoulder[0], left_shoulder[2], left_shoulder[1]), (right_shoulder[0], right_shoulder[2], right_shoulder[1])),
        "Neck Angle": calculate_angle((left_shoulder[0], left_shoulder[2], left_shoulder[1]), (shoulder_mid[0], shoulder_mid[2], shoulder_mid[1]), (ear_mid[0], ear_mid[2], ear_mid[1])),
        "Head Tilt": _angle_to_vertical((nose[0], nose[2], nose[1]), (ear_mid[0], ear_mid[2], ear_mid[1])),
        "Spine Inclination": _angle_to_vertical((hip_mid[0], hip_mid[2], hip_mid[1]), (shoulder_mid[0], shoulder_mid[2], shoulder_mid[1])),
        "Pelvic Alignment": _angle_to_horizontal((left_hip[0], left_hip[2], left_hip[1]), (right_hip[0], right_hip[2], right_hip[1])),
        "Knee Alignment": _angle_to_horizontal((left_knee[0], left_knee[2], left_knee[1]), (right_knee[0], right_knee[2], right_knee[1])),
        "Hip Shift": _angle_to_vertical((hip_mid[0], hip_mid[2], hip_mid[1]), (shoulder_mid[0], shoulder_mid[2], shoulder_mid[1])),
    }

    overall_score = 100.0
    for key, value in front_metrics.items():
        if key in {"Shoulder Alignment", "Pelvic Alignment", "Knee Alignment", "Hip Shift", "Spine Inclination", "Head Tilt"}:
            overall_score -= min(18.0, value * 0.8)
        elif key == "Neck Angle":
            overall_score -= min(15.0, value * 0.5)

    for key, value in side_metrics.items():
        if key in {"Shoulder Alignment", "Pelvic Alignment", "Knee Alignment", "Hip Shift", "Spine Inclination", "Head Tilt"}:
            overall_score -= min(10.0, value * 0.4)
        elif key == "Neck Angle":
            overall_score -= min(9.0, value * 0.3)

    overall_score = max(0.0, min(100.0, overall_score))
    return {
        "front_metrics": front_metrics,
        "side_metrics": side_metrics,
        "overall_score": overall_score,
    }