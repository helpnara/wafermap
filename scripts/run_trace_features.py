#!/usr/bin/env python3
"""M5.5-② 실행 — 요약통계가 놓치는 순간 이상을 시계열 피처로 잡는다.

첨부 도메인 문서 §12의 문제를 데이터로 확인한다.

    정상 500℃인데 30~31초만 540℃  →  60초 평균은 정상 범위 안
    요약통계(mean/std/min/max)만 보는 분석은 이 이상을 구조적으로 놓친다.

사용법:
    python scripts/run_trace_features.py
    python scripts/run_trace_features.py --show-trace   # 실제 시계열 비교
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

warnings.filterwarnings("ignore", message=".*binary classifier with TreeExplainer.*")

import numpy as np  # noqa: E402

from wafermap.analysis import validate  # noqa: E402
from wafermap.config import SPIKE_CAUSE_RULES, STEPS_BY_ID  # noqa: E402
from wafermap.data import loader  # noqa: E402


def show_trace(fdc, trace, gt, pattern: str) -> None:
    """스파이크 기인 웨이퍼와 정상 웨이퍼의 시계열을 나란히 그린다 (아스키)."""
    rule = SPIKE_CAUSE_RULES[pattern]
    param = rule.perturbations[0].param
    spec = STEPS_BY_ID[rule.step_id].param(param)

    sub = trace[(trace["step_id"] == rule.step_id) & (trace["param"] == param)]
    if sub.empty:
        print("  (저장된 시계열 표본에 해당 파라미터가 없습니다)")
        return

    spike_ids = set(gt.loc[gt["cause_mechanism"] == "spike", "wafer_id"])
    normal_ids = set(gt.loc[gt["pattern_label"] == "none", "wafer_id"])

    def pick(ids: set[str]) -> str | None:
        found = sub[sub["wafer_id"].isin(ids)]["wafer_id"]
        return str(found.iloc[0]) if len(found) else None

    for label, wid in (("스파이크 기인", pick(spike_ids)), ("정상", pick(normal_ids))):
        if wid is None:
            print(f"  {label}: 표본 없음")
            continue
        series = sub[sub["wafer_id"] == wid].sort_values("t_sec")
        values = series["value"].to_numpy()
        lo, hi = values.min(), values.max()
        span = max(hi - lo, 1e-9)
        print(f"\n  {label} ({wid})  {param} [{lo:.2f} ~ {hi:.2f} {spec.unit}]")
        for level in range(10, -1, -1):
            row = "".join(
                "█" if (v - lo) / span >= level / 10 else " " for v in values
            )
            print(f"    {lo + span * level / 10:7.2f} |{row}")
        print(f"            +{'-' * len(values)}  (0 ~ 60초)")

        row = fdc[(fdc["wafer_id"] == wid) & (fdc["step_id"] == rule.step_id)].iloc[0]
        print(
            f"      요약: mean {row[f'{param}_mean']:.3f}  std {row[f'{param}_std']:.3f}"
            f"  max {row[f'{param}_max']:.3f}"
        )
        print(
            f"      파생: time_above {row[f'{param}_time_above']:.1f}초"
            f"  peak_dev {row[f'{param}_peak_dev']:.2f}"
            f"  n_excursions {row[f'{param}_n_excursions']:.0f}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description="M5.5-② 시계열 파생 피처")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--show-trace", action="store_true", help="시계열을 그려 비교")
    args = parser.parse_args()

    if not loader.is_built(args.source):
        print(f"❌ 데이터가 없습니다. python scripts/build_dataset.py --source {args.source}",
              file=sys.stderr)
        return 1

    wm = loader.load_wafer_master(args.source)
    fdc = loader.load_fdc_summary(args.source)
    gt = loader.load_ground_truth(args.source)

    print("=" * 78)
    print(f" M5.5-② 시계열 파생 피처   (source={args.source})")
    print("=" * 78)
    print("  60초 중 3초만 튄 이상은 평균을 0.35σ밖에 못 움직인다.")
    print("  요약통계만 보는 분석은 이런 이상을 **구조적으로** 놓친다.")

    for pattern, rule in SPIKE_CAUSE_RULES.items():
        n = int((gt["cause_mechanism"] == "spike").sum())
        print(f"\n{'=' * 78}\n [1] 심어 둔 스파이크 — {pattern} ({n}장)\n{'=' * 78}")
        print(f"  스텝: {rule.step_id} {STEPS_BY_ID[rule.step_id].name_ko}")
        print(f"  기전: {rule.mechanism}")
        pert = rule.perturbations[0]
        print(
            f"  주입: {pert.param} +{pert.spike_sigma:.0f}σ × {pert.spike_seconds:.0f}초"
            f"  → 60초 평균 이동 {pert.spike_sigma * pert.spike_seconds / 60:.2f}σ"
        )

    if args.show_trace:
        print(f"\n{'=' * 78}\n [2] 시계열 비교\n{'=' * 78}")
        trace = loader.load_fdc_trace(args.source)
        for pattern in SPIKE_CAUSE_RULES:
            show_trace(fdc, trace, gt, pattern)

    print(f"\n{'=' * 78}\n [3] 채점 — 요약통계만 vs 시계열 피처 추가\n{'=' * 78}")
    report = validate.score_trace_features(wm, fdc, gt)
    for score in report.scores:
        print("  " + score.describe())
    print()
    print("  " + report.summary().replace("\n", "\n  "))

    print("  ⚠️ 한계: 여기서는 스파이크를 직접 심었으므로 검출이 깨끗하게 된다.")
    print("     실측 FDC는 샘플링 주기가 성기고(수 초~수십 초) 결측·센서 드리프트가")
    print("     섞여 있어, 짧은 스파이크는 애초에 기록되지 않았을 수 있다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
