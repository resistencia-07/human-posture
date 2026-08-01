import cv2
import numpy as np

from config import WINDOW_NAME


def _draw_rounded_box(frame, x1, y1, x2, y2, color, thickness=1, radius=12) -> None:
    """Draw a rounded rectangle on the frame for a polished dashboard panel."""
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(frame, (x1 + radius, y1), (x2 - radius, y2), color, thickness)
    cv2.rectangle(frame, (x1, y1 + radius), (x2, y2 - radius), color, thickness)
    cv2.ellipse(frame, (x1 + radius, y1 + radius), (radius, radius), 180, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - radius, y1 + radius), (radius, radius), 270, 0, 90, color, thickness)
    cv2.ellipse(frame, (x1 + radius, y2 - radius), (radius, radius), 90, 0, 90, color, thickness)
    cv2.ellipse(frame, (x2 - radius, y2 - radius), (radius, radius), 0, 0, 90, color, thickness)


def _line_color(deviation: float):
    """Map a posture deviation angle to a visualization color."""
    if deviation < 6.0:
        return (0, 255, 0)
    if deviation < 12.0:
        return (0, 255, 255)
    return (0, 0, 255)


def _get_landmark_pixel(landmarks, index, frame_width, frame_height):
    """Convert a MediaPipe landmark to image coordinates."""
    if landmarks is None or index >= len(landmarks):
        return None
    landmark = landmarks[index]
    return (int(landmark.x * frame_width), int(landmark.y * frame_height))


def draw_pose_landmarks(frame, results) -> None:
    """Draw the MediaPipe skeleton and alignment guides over the live camera frame."""
    if not results or not results.pose_landmarks:
        return

    landmarks = results.pose_landmarks[0]
    height, width = frame.shape[:2]
    points = {}
    for idx in range(len(landmarks)):
        pixel = _get_landmark_pixel(landmarks, idx, width, height)
        if pixel is not None:
            points[idx] = pixel

    connections = [
        (11, 12), (11, 13), (13, 15), (15, 17),
        (12, 14), (14, 16), (16, 18),
        (11, 23), (12, 24), (23, 24),
        (23, 25), (24, 26), (25, 27), (26, 28),
        (27, 31), (28, 32), (31, 32),
    ]
    for start, end in connections:
        if start in points and end in points:
            cv2.line(frame, points[start], points[end], (255, 255, 255), 2, cv2.LINE_AA)

    for idx, point in points.items():
        color = (255, 0, 0) if idx in {0, 11, 12, 23, 24, 25, 26, 27, 28} else (255, 255, 255)
        cv2.circle(frame, point, 3, color, -1, cv2.LINE_AA)

    if 11 in points and 12 in points:
        shoulder_mid = ((points[11][0] + points[12][0]) // 2, (points[11][1] + points[12][1]) // 2)
        cv2.line(frame, points[11], points[12], (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, shoulder_mid, 4, (255, 255, 255), -1, cv2.LINE_AA)

    if 23 in points and 24 in points:
        hip_mid = ((points[23][0] + points[24][0]) // 2, (points[23][1] + points[24][1]) // 2)
        cv2.line(frame, points[23], points[24], (255, 255, 255), 2, cv2.LINE_AA)
        cv2.circle(frame, hip_mid, 4, (255, 255, 255), -1, cv2.LINE_AA)

    if 11 in points and 12 in points and 23 in points and 24 in points:
        shoulder_mid = ((points[11][0] + points[12][0]) // 2, (points[11][1] + points[12][1]) // 2)
        hip_mid = ((points[23][0] + points[24][0]) // 2, (points[23][1] + points[24][1]) // 2)
        deviation = abs(shoulder_mid[0] - hip_mid[0]) / 10.0
        cv2.line(frame, shoulder_mid, hip_mid, _line_color(deviation), 3, cv2.LINE_AA)

    for left_idx, right_idx in [(11, 12), (23, 24), (25, 26), (27, 28)]:
        if left_idx in points and right_idx in points:
            deviation = abs(points[left_idx][0] - points[right_idx][0]) / 10.0
            cv2.line(frame, points[left_idx], points[right_idx], _line_color(deviation), 3, cv2.LINE_AA)


def draw_posture_dashboard(frame, analysis, results, fps: float) -> None:
    """Render a polished posture dashboard on the right side of the camera frame."""
    if frame is None:
        return

    height, width = frame.shape[:2]
    dashboard_width = max(320, int(width * 0.25))
    dashboard_height = height - 24
    x0 = width - dashboard_width - 12
    y0 = 12
    x1 = width - 12
    y1 = height - 12

    panel = np.zeros((dashboard_height, dashboard_width, 3), dtype=np.uint8)
    panel[:] = (34, 34, 34)
    _draw_rounded_box(panel, 2, 2, dashboard_width - 2, dashboard_height - 2, (80, 80, 80), thickness=2, radius=16)

    title_color = (255, 255, 255)
    cv2.putText(panel, "AI Posture Analysis", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.75, title_color, 2, cv2.LINE_AA)
    cv2.putText(panel, f"FPS {fps:.1f}", (18, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)

    if analysis is None:
        cv2.putText(panel, "No pose detected", (18, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)
    else:
        view_label = "Front View"
        status_color = analysis.color
        cv2.putText(panel, f"View: {view_label}", (18, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(panel, f"Status: {analysis.status}", (18, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2, cv2.LINE_AA)

        score = max(0.0, min(100.0, analysis.overall_score))
        bar_x, bar_y, bar_w, bar_h = 18, 142, dashboard_width - 36, 16
        _draw_rounded_box(panel, bar_x, bar_y, bar_x + bar_w, bar_y + bar_h, (100, 100, 100), thickness=1, radius=8)
        fill_w = int((score / 100.0) * (bar_w - 4))
        fill_color = (0, 255, 0) if score >= 85 else (0, 255, 255) if score >= 60 else (0, 0, 255)
        cv2.rectangle(panel, (bar_x + 2, bar_y + 2), (bar_x + 2 + fill_w, bar_y + bar_h - 2), fill_color, -1)
        cv2.putText(panel, f"{score:.1f}%", (bar_x + bar_w + 8, bar_y + 13), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

        metrics = [
            ("Shoulder", analysis.front_metrics.get("Shoulder Alignment", 0.0)),
            ("Neck", analysis.front_metrics.get("Neck Angle", 0.0)),
            ("Spine", analysis.front_metrics.get("Spine Inclination", 0.0)),
            ("Pelvis", analysis.front_metrics.get("Pelvic Alignment", 0.0)),
            ("Knee", analysis.front_metrics.get("Knee Alignment", 0.0)),
        ]
        row_y = 176
        for name, value in metrics:
            metric_color = (0, 255, 0) if value < 8 else (0, 255, 255) if value < 18 else (0, 0, 255)
            cv2.putText(panel, name, (18, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
            cv2.putText(panel, f"{value:.1f}°", (dashboard_width - 90, row_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, metric_color, 1, cv2.LINE_AA)
            row_y += 24

        cv2.putText(panel, "Overall Score", (18, dashboard_height - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(panel, f"{analysis.status}", (dashboard_width - 120, dashboard_height - 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

    frame[y0:y1, x0:x1] = cv2.addWeighted(frame[y0:y1, x0:x1], 0.72, panel, 0.28, 0)
    cv2.putText(frame, "Press q to quit", (20, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, f"Camera: {WINDOW_NAME}", (width - 220, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
