from dataclasses import dataclass
from typing import Dict, Tuple

from geometry import analyze_posture


@dataclass
class PostureAnalysis:
    """Store the computed posture assessment and display metadata."""

    front_metrics: Dict[str, float]
    side_metrics: Dict[str, float]
    overall_score: float
    status: str
    color: Tuple[int, int, int]


def _status_for_score(score: float) -> Tuple[str, Tuple[int, int, int]]:
    """Map a posture score to a status label and BGR color."""
    if score >= 85.0:
        return "Excellent", (0, 255, 0)
    if score >= 70.0:
        return "Good", (0, 255, 0)
    if score >= 50.0:
        return "Moderate", (0, 255, 255)
    return "Poor", (0, 0, 255)


def evaluate_posture(landmarks) -> PostureAnalysis:
    """Convert raw geometry values into a readable posture assessment object."""
    result = analyze_posture(landmarks)
    if not result:
        return PostureAnalysis({}, {}, 0.0, "Poor", (0, 0, 255))

    score = result.get("overall_score", 0.0)
    status, color = _status_for_score(score)
    return PostureAnalysis(
        front_metrics=result.get("front_metrics", {}),
        side_metrics=result.get("side_metrics", {}),
        overall_score=score,
        status=status,
        color=color,
    )
