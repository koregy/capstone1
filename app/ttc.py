"""
TTC (Time-To-Collision) estimation — v2.

Five improvements over v1:
  1) Scale outlier rejection — ignores single-frame spikes
  2) Stronger smoothing — EMA alpha defaults to 0.15, EMA applied to scale itself
  3) Minimum observation count — TTC is None when n_updates < min_updates
  4) Danger level hysteresis — holds critical for N frames after entry
  5) Persistence requirement — classified as critical only when critical K times within window

API: estimator.update(tid, xyxy) → TTC or None.
Classification via estimator.classify(tid) (replaces legacy classify_ttc).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional


# if growth_ema is at or below this value, object is not approaching — TTC undefined
# v1's 1e-3 was too permissive, producing wild TTC near-zero oscillations
_MIN_GROWTH_PER_SEC = 2.0  # scale[px] / sec

# Outlier rejection: treat as spike if single-frame scale change exceeds ratio OR pixel threshold
# (ratio alone lets small jitter on large objects through — pixel threshold catches those)
_MAX_SCALE_JUMP_RATIO = 0.30
_MAX_SCALE_JUMP_PX = 15.0

# clamp inst_growth before feeding into growth_ema.
# prevents small jitter from becoming enormous ds/dt at high frame rates (e.g. 60 FPS)
_INST_GROWTH_CLAMP = 200.0  # |scale[px] / sec|

# EMA applied to scale itself
_SCALE_EMA_ALPHA = 0.4

_DEFAULT_MIN_UPDATES = 5
_DEFAULT_CRITICAL_HOLD = 5
_DEFAULT_CRITICAL_PERSIST = 3
_DEFAULT_CRITICAL_WINDOW = 5


@dataclass
class TrackState:
    last_scale_raw: float
    last_scale: float       # post-EMA
    last_time: float

    growth_ema: float = 0.0
    alpha: float = 0.15

    ttc: Optional[float] = None
    n_updates: int = 0

    level_history: Deque[str] = field(default_factory=lambda: deque(maxlen=5))
    critical_hold: int = 0


@dataclass
class TTCEstimator:
    states: Dict[int, TrackState] = field(default_factory=dict)
    alpha: float = 0.15
    stale_after_s: float = 1.0
    min_updates: int = _DEFAULT_MIN_UPDATES
    critical_persist: int = _DEFAULT_CRITICAL_PERSIST
    critical_window: int = _DEFAULT_CRITICAL_WINDOW
    critical_hold: int = _DEFAULT_CRITICAL_HOLD

    @staticmethod
    def _scale_from_xyxy(xyxy: tuple[float, float, float, float]) -> float:
        x1, y1, x2, y2 = xyxy
        w = max(0.0, x2 - x1)
        h = max(0.0, y2 - y1)
        return math.sqrt(max(w * h, 1e-6))

    def update(
        self,
        track_id: int,
        xyxy: tuple[float, float, float, float],
        now: Optional[float] = None,
    ) -> Optional[float]:
        t = now if now is not None else time.perf_counter()
        s_raw = self._scale_from_xyxy(xyxy)

        st = self.states.get(track_id)
        if st is None:
            self.states[track_id] = TrackState(
                last_scale_raw=s_raw,
                last_scale=s_raw,
                last_time=t,
                alpha=self.alpha,
                level_history=deque(maxlen=self.critical_window),
            )
            return None

        dt = t - st.last_time
        if dt <= 0:
            return st.ttc

        # ── 1) Outlier rejection: spike if ratio OR pixel jump exceeds threshold
        abs_jump = abs(s_raw - st.last_scale_raw)
        ratio = abs_jump / max(st.last_scale_raw, 1.0)
        if st.n_updates >= 2 and (ratio > _MAX_SCALE_JUMP_RATIO or
                                   abs_jump > _MAX_SCALE_JUMP_PX):
            st.last_scale_raw = s_raw
            st.last_time = t
            if st.level_history:
                st.level_history.append(st.level_history[-1])
            return st.ttc  # retain previous TTC

        # ── 2) EMA on scale
        if st.n_updates == 0:
            s_smooth = s_raw
        else:
            s_smooth = _SCALE_EMA_ALPHA * s_raw + (1 - _SCALE_EMA_ALPHA) * st.last_scale

        # ── 3) ds/dt and its EMA
        # clamp to prevent small jitter from producing huge ds/dt at high frame rates
        inst_growth = (s_smooth - st.last_scale) / dt
        inst_growth = max(-_INST_GROWTH_CLAMP, min(_INST_GROWTH_CLAMP, inst_growth))
        if st.n_updates == 0:
            st.growth_ema = inst_growth
        else:
            st.growth_ema = st.alpha * inst_growth + (1 - st.alpha) * st.growth_ema

        st.last_scale_raw = s_raw
        st.last_scale = s_smooth
        st.last_time = t
        st.n_updates += 1

        # ── 4) insufficient observations
        if st.n_updates < self.min_updates:
            st.ttc = None
            return None

        # ── 5) compute TTC
        if st.growth_ema <= _MIN_GROWTH_PER_SEC:
            st.ttc = None
        else:
            st.ttc = s_smooth / st.growth_ema

        return st.ttc

    def classify(self, track_id: int) -> str:
        """Stable classification: applies hysteresis and persistence."""
        st = self.states.get(track_id)
        if st is None:
            return "none"
        instant = _instant_level(st.ttc)
        st.level_history.append(instant)

        # within critical hold window
        if st.critical_hold > 0:
            st.critical_hold -= 1
            if instant in ("safe", "caution", "none"):
                return "warning"
            return instant

        # critical persistence check
        if instant == "critical":
            n_crit = sum(1 for L in st.level_history if L == "critical")
            if n_crit >= self.critical_persist:
                st.critical_hold = self.critical_hold
                return "critical"
            return "warning"

        return instant

    def prune(self, active_ids: Optional[set[int]] = None,
              now: Optional[float] = None) -> None:
        t = now if now is not None else time.perf_counter()
        dead = []
        for tid, st in self.states.items():
            if active_ids is not None and tid not in active_ids:
                if t - st.last_time > self.stale_after_s:
                    dead.append(tid)
            elif t - st.last_time > self.stale_after_s:
                dead.append(tid)
        for tid in dead:
            del self.states[tid]

    def get(self, track_id: int) -> Optional[float]:
        st = self.states.get(track_id)
        return st.ttc if st is not None else None

    def debug_state(self, track_id: int) -> Optional[dict]:
        st = self.states.get(track_id)
        if st is None:
            return None
        return {
            "scale": st.last_scale,
            "growth_ema": st.growth_ema,
            "ttc": st.ttc if st.ttc is not None else float("nan"),
            "n_updates": st.n_updates,
        }


def _instant_level(ttc: Optional[float]) -> str:
    if ttc is None:
        return "none"
    if ttc < 1.0:
        return "critical"
    if ttc < 2.5:
        return "warning"
    if ttc < 5.0:
        return "caution"
    return "safe"


def classify_ttc(ttc: Optional[float]) -> str:
    """[compatibility] Instantaneous TTC classification. For runtime use TTCEstimator.classify()."""
    return _instant_level(ttc)
