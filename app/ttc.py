"""
TTC (Time-To-Collision) 계산 — v2.

v1 대비 다섯 가지 개선:
  1) Scale outlier rejection — 한 프레임 spike 무시
  2) 스무딩 강화 — EMA alpha 기본 0.15, scale 자체에도 EMA
  3) 최소 관측 수 — n_updates < min_updates 이면 TTC None
  4) 위험 레벨 히스테리시스 — critical 진입 후 N 프레임 유지
  5) 지속성 요구 — 윈도우 안에 K 번 critical 일 때만 critical 로 분류

API 호환: estimator.update(tid, xyxy) → TTC or None.
화면 분류는 estimator.classify(tid) 로 (이전 classify_ttc 함수 대신).
"""

from __future__ import annotations

import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Optional


# growth_ema 가 이 값 이하면 "수렴 중 아님" → TTC 미정의
# v1 의 1e-3 은 너무 관대해 0 근처 진동에서 미친 TTC 가 나옴
_MIN_GROWTH_PER_SEC = 2.0  # scale[px] / sec

# Outlier rejection: 한 프레임 scale 변화가 이 비율 또는 이 픽셀값을 넘으면 spike
# (ratio 만으로는 큰 객체의 작은 jitter가 새어나가서 픽셀 임계값도 함께 본다)
_MAX_SCALE_JUMP_RATIO = 0.30
_MAX_SCALE_JUMP_PX = 15.0

# growth_ema 에 들어가기 전 inst_growth 를 이 범위로 clamp.
# 60 FPS 영상처럼 dt 가 작을 때 작은 jitter 가 거대한 ds/dt 가 되는 것을 방지.
_INST_GROWTH_CLAMP = 200.0  # |scale[px] / sec|

# scale 자체에도 EMA 적용
_SCALE_EMA_ALPHA = 0.4

_DEFAULT_MIN_UPDATES = 5
_DEFAULT_CRITICAL_HOLD = 5
_DEFAULT_CRITICAL_PERSIST = 3
_DEFAULT_CRITICAL_WINDOW = 5


@dataclass
class TrackState:
    last_scale_raw: float
    last_scale: float       # EMA 통과 후
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

        # ── 1) Outlier rejection: ratio OR 픽셀값 둘 중 하나라도 임계 초과면 spike
        abs_jump = abs(s_raw - st.last_scale_raw)
        ratio = abs_jump / max(st.last_scale_raw, 1.0)
        if st.n_updates >= 2 and (ratio > _MAX_SCALE_JUMP_RATIO or
                                   abs_jump > _MAX_SCALE_JUMP_PX):
            st.last_scale_raw = s_raw
            st.last_time = t
            if st.level_history:
                st.level_history.append(st.level_history[-1])
            return st.ttc  # 직전 TTC 그대로 유지

        # ── 2) scale 자체 EMA
        if st.n_updates == 0:
            s_smooth = s_raw
        else:
            s_smooth = _SCALE_EMA_ALPHA * s_raw + (1 - _SCALE_EMA_ALPHA) * st.last_scale

        # ── 3) ds/dt 와 그것의 EMA
        # 작은 dt(고프레임률)에서 작은 jitter가 거대한 ds/dt가 되는 것을 막기 위해 clamp.
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

        # ── 4) 최소 관측 수 미달
        if st.n_updates < self.min_updates:
            st.ttc = None
            return None

        # ── 5) TTC 계산
        if st.growth_ema <= _MIN_GROWTH_PER_SEC:
            st.ttc = None
        else:
            st.ttc = s_smooth / st.growth_ema

        return st.ttc

    def classify(self, track_id: int) -> str:
        """안정 분류: 히스테리시스 + 지속성 적용."""
        st = self.states.get(track_id)
        if st is None:
            return "none"
        instant = _instant_level(st.ttc)
        st.level_history.append(instant)

        # critical hold 중
        if st.critical_hold > 0:
            st.critical_hold -= 1
            if instant in ("safe", "caution", "none"):
                return "warning"
            return instant

        # critical 지속성 판정
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
    """[호환용] 순간 TTC 분류. 런타임 분류는 TTCEstimator.classify()."""
    return _instant_level(ttc)
