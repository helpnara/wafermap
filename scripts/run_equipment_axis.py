#!/usr/bin/env python3
"""M5.5-① 실행 — 불량이 공정에서 왔나, 검사에서 왔나.

프로브 카드 니들이 마모되면 엣지 die부터 접촉이 끊겨 **Edge-Ring과 똑같이 생긴 맵**이
나온다. 그런데 이건 식각 챔버를 아무리 손봐도 없어지지 않는다. 이 스크립트는 두 축을
나란히 놓고 비교해, 분석이 공정만 쳐다보지 않도록 만든다.

사용법:
    python scripts/run_equipment_axis.py
    python scripts/run_equipment_axis.py --pattern Edge-Ring   # 껍질 벗기기까지
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafermap.analysis import commonality, spc, validate  # noqa: E402
from wafermap.config import TEST_CAUSE_RULES  # noqa: E402
from wafermap.data import loader  # noqa: E402


def show_pattern(wm, fdc, gt, pattern: str) -> None:
    """한 패턴을 축별로 상세히 본다."""
    split = spc.split_by_pattern(wm, pattern)
    sub = gt[gt["pattern_label"] == pattern]
    n_test = int(sub["is_test_induced"].sum())

    print(f"\n{'=' * 78}\n [1] {pattern} — 두 축 비교\n{'=' * 78}")
    print(f"  정답지: 공정 기인 {len(sub) - n_test}장 · 검사 기인 {n_test}장")
    if pattern in TEST_CAUSE_RULES:
        print(f"  검사 기전: {TEST_CAUSE_RULES[pattern].mechanism}")

    axes = commonality.by_axis(fdc, split.case_ids, split.control_ids, wafer_master=wm)
    print(f"\n  {'축':<10}{'설비':<16}{'OR':>7}{'통합순위':>9}")
    print("  " + "-" * 44)
    for _, row in axes.iterrows():
        print(
            f"  {row['axis_ko']:<10}{row['chamber_id']:<16}"
            f"{row['odds_ratio']:>7.2f}{row['rank_overall']:>8}위"
        )

    print(f"\n{'=' * 78}\n [2] 껍질 벗기기 — 원인이 둘 이상일 때\n{'=' * 78}")
    layers = commonality.peel(fdc, split.case_ids, split.control_ids, wafer_master=wm)
    for layer in layers:
        print("  " + layer.describe())
    if not layers:
        print("  (벗겨 낼 층이 없다)")
    print("\n  ⚠️ 벗기기는 '그 설비를 지난 웨이퍼'를 통째로 제외한다. 모든 웨이퍼가")
    print("     모든 스텝을 지나므로 무관한 웨이퍼까지 빠지며, 챔버가 적은 스텝일수록")
    print("     심하다. 축 비교([1])가 더 안전한 판단 근거다.")


def main() -> int:
    parser = argparse.ArgumentParser(description="M5.5-① 공정 vs 검사 설비 축")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--pattern", default=None, help="상세히 볼 패턴")
    args = parser.parse_args()

    if not loader.is_built(args.source):
        print(f"❌ 데이터가 없습니다. python scripts/build_dataset.py --source {args.source}",
              file=sys.stderr)
        return 1

    wm = loader.load_wafer_master(args.source)
    fdc = loader.load_fdc_summary(args.source)
    gt = loader.load_ground_truth(args.source)

    print("=" * 78)
    print(f" M5.5-① 공정 기인 vs 검사 기인   (source={args.source})")
    print("=" * 78)
    print("  웨이퍼 맵의 불량이 공정에서 왔다는 보장은 없다. 프로브 카드가 마모돼도")
    print("  똑같이 생긴 맵이 나오고, 그건 챔버를 손봐도 없어지지 않는다.")

    if args.pattern:
        show_pattern(wm, fdc, gt, args.pattern)

    print(f"\n{'=' * 78}\n [3] 전체 채점 — 정답지 대비\n{'=' * 78}")
    report = validate.score_equipment_axis(wm, fdc, gt)
    for score in report.scores:
        print("  " + score.describe())
    print()
    print("  " + report.summary().replace("\n", "\n  "))

    print("\n  📖 조치가 다르다는 점이 핵심이다. 프로브 카드 세정은 수 시간,")
    print("     공정 챔버 PM은 수 일이다. 오진하면 멀쩡한 설비를 세우고도")
    print("     불량이 계속된다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
