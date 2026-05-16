"""
plot_ttc_debug.py — `--ttc-debug` 로 저장된 CSV 를 시각화.

트랙별로 scale, growth_ema, ttc 의 시계열을 보여줘서
"어디서 TTC 가 이상하게 점프했는지" 를 한눈에 보게 함.

사용 예:
  python experiments/plot_ttc_debug.py \
      --csv results/ttc_debug.csv \
      --out  results/ttc_debug.png
  # 또는 특정 트랙만:
  python experiments/plot_ttc_debug.py --csv ... --tracks 3 7
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="ttc_debug.csv 경로")
    p.add_argument("--out", default=None,
                   help="png 출력 경로. 없으면 화면에 표시.")
    p.add_argument("--tracks", nargs="*", type=int, default=None,
                   help="필터링할 track_id 목록. 없으면 가장 오래 보인 6개.")
    p.add_argument("--min-frames", type=int, default=15,
                   help="이 프레임 수 미만으로 보인 트랙은 무시")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    # CSV 읽기
    rows_per_track: dict[int, list[dict]] = defaultdict(list)
    with open(args.csv) as f:
        reader = csv.DictReader(f)
        for row in reader:
            tid = int(row["track_id"])
            rows_per_track[tid].append({
                "frame": int(row["frame"]),
                "t": float(row["t_sec"]),
                "class": row["class"],
                "scale": float(row["scale"]),
                "growth": float(row["growth_ema"]),
                "ttc": float(row["ttc"]),  # nan 가능
                "level": row["level"],
                "n_updates": int(row["n_updates"]),
            })

    if not rows_per_track:
        print("[plot] no rows in csv", file=sys.stderr)
        return 1

    # 트랙 선택: 명시값 > 가장 길게 보인 top-N
    if args.tracks:
        selected = [t for t in args.tracks if t in rows_per_track]
    else:
        sorted_tids = sorted(rows_per_track,
                             key=lambda t: -len(rows_per_track[t]))
        selected = [t for t in sorted_tids
                    if len(rows_per_track[t]) >= args.min_frames][:6]

    if not selected:
        print(f"[plot] no track with >= {args.min_frames} frames", file=sys.stderr)
        return 1
    print(f"[plot] tracks: {selected}")

    # ── plot
    import matplotlib  # noqa: WPS433
    if args.out:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: WPS433

    fig, (ax_scale, ax_growth, ax_ttc) = plt.subplots(
        3, 1, figsize=(12, 9), sharex=True,
    )

    colors = plt.cm.tab10.colors

    for i, tid in enumerate(selected):
        rows = rows_per_track[tid]
        ts = [r["t"] for r in rows]
        scales = [r["scale"] for r in rows]
        growths = [r["growth"] for r in rows]
        ttcs = [r["ttc"] if r["ttc"] == r["ttc"] else None for r in rows]
        cls = rows[0]["class"]
        c = colors[i % len(colors)]
        label = f"#{tid} {cls}"

        ax_scale.plot(ts, scales, marker=".", ms=3, color=c, label=label)
        ax_growth.plot(ts, growths, marker=".", ms=3, color=c, label=label)

        # TTC: nan 또는 None 은 끊김으로 표시
        clean_ts, clean_ttcs = [], []
        for t, v in zip(ts, ttcs):
            if v is None or v != v:  # nan check
                # 단절: 빈 자리는 그래프에서 끊김
                if clean_ts:
                    ax_ttc.plot(clean_ts, clean_ttcs, marker=".", ms=3,
                                color=c, label=label if not clean_ts else None)
                    clean_ts, clean_ttcs = [], []
            else:
                clean_ts.append(t)
                clean_ttcs.append(v)
        if clean_ts:
            ax_ttc.plot(clean_ts, clean_ttcs, marker=".", ms=3,
                        color=c, label=label)

    ax_scale.set_ylabel("scale = sqrt(area)\n[pixels]")
    ax_scale.set_title("Track scale over time")
    ax_scale.legend(loc="upper left", fontsize=8)
    ax_scale.grid(True, alpha=0.3)

    ax_growth.set_ylabel("growth_ema\n[scale / sec]")
    ax_growth.set_title("EMA-smoothed ds/dt — positive = approaching")
    ax_growth.axhline(0, color="k", lw=0.5)
    ax_growth.grid(True, alpha=0.3)

    ax_ttc.set_ylabel("TTC [sec]")
    ax_ttc.set_xlabel("time [sec]")
    ax_ttc.set_title("Estimated TTC — gaps = TTC undefined (not approaching)")
    # 위험 영역 음영
    ax_ttc.axhspan(0, 1.0, color="red", alpha=0.10, label="critical")
    ax_ttc.axhspan(1.0, 2.5, color="orange", alpha=0.10, label="warning")
    ax_ttc.axhspan(2.5, 5.0, color="yellow", alpha=0.10, label="caution")
    ax_ttc.set_ylim(0, 15)
    ax_ttc.grid(True, alpha=0.3)

    fig.tight_layout()

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(args.out, dpi=120)
        print(f"[plot] saved {args.out}")
    else:
        plt.show()

    return 0


if __name__ == "__main__":
    sys.exit(main())
