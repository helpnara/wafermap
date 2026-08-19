#!/usr/bin/env python3
"""M3 실행 — SPC 이상 탐지 + 커미널리티 분석 + 정답 대비 채점.

사용법:
    python scripts/run_commonality.py                     # 전체 파이프라인
    python scripts/run_commonality.py --pattern Edge-Ring # 한 패턴만 상세히
    python scripts/run_commonality.py --unit wafer        # 표본 단위 비교용
    python scripts/run_commonality.py --compare-units     # lot vs wafer 비교
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from wafermap.analysis import commonality, spc, validate  # noqa: E402
from wafermap.config import PROCESS_STEPS  # noqa: E402
from wafermap.data import loader  # noqa: E402
from wafermap.data.fdc_simulator import EDS_DELAY_HOURS, STEP_INTERVAL_HOURS  # noqa: E402

CYCLE_TIME = pd.Timedelta(hours=STEP_INTERVAL_HOURS * len(PROCESS_STEPS) + EDS_DELAY_HOURS)


def run_spc(wm: pd.DataFrame, exc: pd.DataFrame, pattern: str) -> None:
    """SPC 이상 탐지를 돌리고 채점한다."""
    print(f"\n{'=' * 78}\n [1] SPC 이상 구간 탐지 — {pattern}\n{'=' * 78}")

    data = spc.aggregate(wm, metric=f"{pattern}_rate", freq="D")
    chart = spc.control_chart(data, metric=f"{pattern}_rate")

    print(f"  관리도: p-chart · 중심선 {chart.center:.2f}% · 과분산계수 {chart.overdispersion:.2f}")
    if chart.overdispersion > 1.5:
        print(f"     └ 이항분포보다 변동이 {chart.overdispersion:.1f}배 크다 "
              f"— lot 단위로 함께 움직이기 때문 (Laney 보정 적용됨)")
    print(f"  검출된 이상 구간: {len(chart.alarms)}건")

    for alarm in chart.alarms[:6]:
        cause_start, cause_end = spc.to_cause_window(alarm, CYCLE_TIME)
        print(
            f"    {alarm.alarm_id} [{alarm.method:<5}] "
            f"EDS {alarm.start:%m-%d}~{alarm.end:%m-%d} "
            f"→ 원인 추정 {cause_start:%m-%d}~{cause_end:%m-%d} "
            f"({alarm.peak_deviation:.1f}σ)"
        )
    if len(chart.alarms) > 6:
        print(f"    … 외 {len(chart.alarms) - 6}건")

    score = validate.score_spc_detection(chart.alarms, exc, pattern, CYCLE_TIME)
    print(
        f"\n  채점: 실제 이상 사건 {score['n_truth']}건 중 "
        f"{score['n_detected']}건 검출 ({score['detection_rate']:.0%}) · "
        f"위양성 {score['n_false_positive']}건"
    )
    print("     └ ⚠️ 사이클 타임 66시간을 보정한 채점이다. 보정하지 않으면 "
          "'불량 발견 시점'과 '원인 발생 시점'이 어긋나 검출률이 낮게 나온다.")


def run_commonality(
    wm: pd.DataFrame, fdc: pd.DataFrame, gt: pd.DataFrame, pattern: str, unit: str
) -> None:
    """커미널리티 분석을 돌리고 결과를 해석과 함께 출력한다."""
    print(f"\n{'=' * 78}\n [2] 커미널리티 분석 — {pattern} (단위: {unit})\n{'=' * 78}")

    split = spc.split_by_pattern(wm, pattern)
    ranking = commonality.analyze(
        fdc, split.case_ids, split.control_ids, wafer_master=wm, unit=unit
    )
    if ranking.empty:
        print("  검정할 챔버가 없습니다.")
        return

    true_step, true_chambers = validate.true_causes(gt, pattern)
    n_case = ranking.iloc[0]["case_total"]
    n_control = ranking.iloc[0]["control_total"]

    print(f"  표본: 이상군 {n_case} · 정상군 {n_control} ({unit} 단위)")
    print(f"  검정한 챔버 {len(ranking)}개 · 유의 판정 {int(ranking['significant'].sum())}개")
    print(f"  정답: 스텝 {true_step}, 챔버 {len(true_chambers)}개\n")

    print(f"  {'':<3}{'챔버':<14}{'스텝':<7}{'이상군':>10}{'정상군':>10}"
          f"{'OR':>8}{'p_adj':>10}  유의")
    print("  " + "-" * 74)
    for i, row in ranking.head(5).iterrows():
        mark = "✅" if row["chamber_id"] in true_chambers else "  "
        sig = "★" if row["significant"] else " "
        print(
            f"  {mark} {row['chamber_id']:<14}{row['step_id']:<7}"
            f"{row['case_through']:>4}/{row['case_total']:<5}"
            f"{row['control_through']:>4}/{row['control_total']:<5}"
            f"{row['odds_ratio']:>8.1f}{row['p_adj']:>10.1e}   {sig}"
        )

    print("\n  📖 읽는 법: 세 숫자를 함께 봐야 한다.")
    print("     · 통과율 차이 — 직관적 크기 (표본이 적으면 우연일 수 있다)")
    print("     · OR — 효과 크기 (신뢰구간이 넓으면 불확실)")
    print("     · p_adj — 우연일 확률 (**보정된 값**을 볼 것, 원래 p는 보지 말 것)")

    if int(ranking["significant"].sum()) == 0:
        print("\n  ⚠️ 유의 판정이 0개다. 이것이 '원인이 없다'는 뜻은 아니다.")
        print(f"     이상군이 {n_case}{unit}뿐이라 검정력이 부족한 것일 수 있다.")
        print("     순위 자체는 정보를 갖는다 — 1순위가 정답을 맞히고 있는지 확인할 것.")


def run_stratification(
    wm: pd.DataFrame, fdc: pd.DataFrame, gt: pd.DataFrame, pattern: str, unit: str
) -> None:
    """교락 판별 (CMH 층화 검정)."""
    print(f"\n{'=' * 78}\n [3] 교락 판별 — CMH 층화 검정\n{'=' * 78}")

    split = spc.split_by_pattern(wm, pattern)
    ranking = commonality.analyze(
        fdc, split.case_ids, split.control_ids, wafer_master=wm, unit=unit
    )
    # 유의 판정이 없으면 상위 후보로 대신한다 (검정력 부족 상황 대응)
    if not ranking["significant"].any():
        ranking = ranking.assign(significant=ranking.index < 4)

    screen = commonality.screen_confounders(
        fdc, split.case_ids, split.control_ids, ranking, top_n=4
    )
    if screen.empty:
        print("  층화할 후보가 부족합니다 (유의 판정 2개 미만).")
        return

    _, true_chambers = validate.true_causes(gt, pattern)
    print(f"  {'':<3}{'챔버':<14}{'원래OR':>8}{'최소조정OR':>11}{'유지/검정':>10}  판정")
    print("  " + "-" * 60)
    for _, row in screen.iterrows():
        mark = "✅" if row["chamber_id"] in true_chambers else "  "
        print(
            f"  {mark} {row['chamber_id']:<14}{row['crude_or']:>8.1f}"
            f"{row['min_adjusted_or']:>11.1f}"
            f"{row['n_survived']:>6}/{row['n_tests']:<4}{row['verdict']}"
        )

    print("\n  📖 읽는 법: 다른 설비를 고정해도 효과가 남으면 '원인 유지',")
    print("     고정하는 순간 사라지면 '교락 의심'(같이 흘렀을 뿐)이다.")


def main() -> int:
    parser = argparse.ArgumentParser(description="M3 — SPC + 커미널리티 분석")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--pattern", default="Edge-Ring", help="상세 분석할 패턴")
    parser.add_argument("--unit", choices=["lot", "wafer"], default="lot")
    parser.add_argument("--compare-units", action="store_true",
                        help="lot vs wafer 표본 단위 비교")
    args = parser.parse_args()

    if not loader.is_built(args.source):
        print(f"❌ 데이터가 없습니다. python scripts/build_dataset.py --source {args.source}",
              file=sys.stderr)
        return 1

    wm = loader.load_wafer_master(args.source)
    fdc = loader.load_fdc_summary(args.source)
    gt = loader.load_ground_truth(args.source)
    exc = loader.load_excursions(args.source)

    print(f"{'=' * 78}")
    print(f" M3 — 이상 공정 탐지 및 원인 설비 특정   (source={args.source})")
    print(f"{'=' * 78}")
    print(f"  웨이퍼 {len(wm):,}장 · lot {wm['lot_id'].nunique():,}개 · "
          f"챔버 {fdc['chamber_id'].nunique()}개")
    print(f"  사이클 타임 (투입→EDS): {CYCLE_TIME.total_seconds() / 3600:.0f}시간")

    run_spc(wm, exc, args.pattern)
    run_commonality(wm, fdc, gt, args.pattern, args.unit)
    run_stratification(wm, fdc, gt, args.pattern, args.unit)

    # ── 전체 패턴 채점 ────────────────────────────────────────────────
    print(f"\n{'=' * 78}\n [4] 전체 채점 — 정답지 대비\n{'=' * 78}")
    report = validate.score_all(wm, fdc, gt, unit=args.unit)
    for score in report.scores:
        print(f"  {score.describe()}")
    print()
    print("  " + report.summary().replace("\n", "\n  "))

    if args.compare_units:
        print(f"\n{'=' * 78}\n [5] 표본 단위 비교 — 왜 lot이어야 하는가\n{'=' * 78}")
        for unit in ("wafer", "lot"):
            rep = validate.score_all(wm, fdc, gt, unit=unit)
            frame = rep.to_frame()
            print(f"\n  [{unit} 단위]  Top-1 챔버 적중 {rep.top1_chamber_rate:.0%} · "
                  f"Top-1 스텝 적중 {rep.top1_step_rate:.0%}")
            print(f"    OR 범위 {frame['top1_odds_ratio'].min():.1f} ~ "
                  f"{frame['top1_odds_ratio'].max():.1f} · "
                  f"유의 판정 평균 {frame['n_significant'].mean():.1f}개")
        print("\n  📖 웨이퍼 단위는 같은 lot의 25장을 독립 관측으로 세어 표본을 25배 부풀린다.")
        print("     그 결과 OR이 수백까지 치솟고 사소한 차이도 압도적으로 유의해 보인다.")
        print("     lot 단위가 통계적으로 올바른 선택이다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
