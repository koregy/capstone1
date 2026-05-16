"""
Draws bbox + track ID + trail + TTC + FPS onto a frame.

Uses only cv2. Colors are determined automatically by TTC danger level.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np


# BGR colors (cv2 default channel order)
_COLOR_BY_LEVEL = {
    "critical": (0, 0, 255),     # red
    "warning":  (0, 128, 255),   # orange
    "caution":  (0, 255, 255),   # yellow
    "safe":     (0, 255, 0),     # green
    "none":     (200, 200, 200), # gray
}


class TrackTrails:
    """Stores centroid trail per track, bounded to a fixed length."""

    def __init__(self, max_len: int = 30) -> None:
        self.max_len = max_len
        self.trails: Dict[int, Deque[Tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=self.max_len),
        )

    def add(self, track_id: int, center: Tuple[int, int]) -> None:
        self.trails[track_id].append(center)

    def get(self, track_id: int) -> Deque[Tuple[int, int]]:
        return self.trails[track_id]

    def prune(self, active_ids: Iterable[int]) -> None:
        active = set(active_ids)
        dead = [tid for tid in self.trails if tid not in active]
        for tid in dead:
            del self.trails[tid]


def draw_box_with_ttc(
    frame: np.ndarray,
    xyxy: Tuple[float, float, float, float],
    label: str,
    track_id: Optional[int],
    ttc: Optional[float],
    level: str,
) -> None:
    """Draws a single object box, label, and TTC text."""
    color = _COLOR_BY_LEVEL.get(level, _COLOR_BY_LEVEL["none"])
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    thickness = 3 if level in ("critical", "warning") else 2
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)

    # build label text
    parts = []
    if track_id is not None:
        parts.append(f"#{track_id}")
    parts.append(label)
    if ttc is not None:
        parts.append(f"TTC={ttc:.2f}s")
    text = " ".join(parts)

    # background box + text
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    y_text = max(y1 - 6, th + 4)
    cv2.rectangle(
        frame,
        (x1, y_text - th - 4),
        (x1 + tw + 4, y_text + baseline - 2),
        color,
        -1,
    )
    cv2.putText(
        frame, text, (x1 + 2, y_text - 2),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
    )


def draw_trail(
    frame: np.ndarray,
    points: Iterable[Tuple[int, int]],
    color: Tuple[int, int, int] = (255, 255, 0),
) -> None:
    pts = list(points)
    if len(pts) < 2:
        return
    for i in range(1, len(pts)):
        cv2.line(frame, pts[i - 1], pts[i], color, 2)


def draw_hud(
    frame: np.ndarray,
    fps: float,
    backend: str,
    extra: Optional[str] = None,
) -> None:
    """Top-left HUD: FPS, backend name, and optional extra info."""
    line1 = f"{backend} | {fps:.1f} FPS"
    cv2.putText(
        frame, line1, (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA,
    )
    if extra:
        cv2.putText(
            frame, extra, (10, 48),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1, cv2.LINE_AA,
        )


def draw_critical_banner(frame: np.ndarray, text: str = "COLLISION IMMINENT") -> None:
    """Red warning banner at the top of the frame (can be made to blink)."""
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 40), (0, 0, 255), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(
        frame, text, ((w - tw) // 2, 28),
        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA,
    )
