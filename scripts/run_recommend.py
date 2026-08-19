#!/usr/bin/env python3
"""M5 실행 — 개선안 도출과 기대효과·ROI 추정.

사용법:
    python scripts/run_recommend.py                     # 전 패턴 요약
    python scripts/run_recommend.py --pattern Donut     # 한 패턴 상세
    python scripts/run_recommend.py --wafer-value 300   # 가정치 조정 (만원)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

warnings.filterwarnings("ignore", message=".*binary classifier with TreeExplainer.*")

from wafermap.analysis import attribution, commonality, recommend, spc  # noqa: E402
from wafermap.config import CAUSE_RULES, STEPS_BY_ID  # noqa: E402
from wafermap.data import loader  # noqa: E402
from wafermap.models import cause_model  # noqa: E402


def analyze(wm, fdc, pattern: str):
    """M3 → M4 → M5를 한 패턴에 대해 이어서 돌린다."""
    rule = CAUSE_RULES.get(pattern)
    if rule is None or rule.step_id is None:
        return None

    split = spc.split_by_pattern(wm, pattern)
    ranking = commonality.analyze(
        fdc, split.case_ids, split.control_ids, wafer_master=wm, unit="lot"
    )
    same_step = ranking[ranking["step_id"] == rule.step_id]
    chamber = str(same_step.iloc[0]["chamber_id"]) if not same_step.empty else None

    result = cause_model.fit(
        fdc, rule.step_id, pattern, split.case_ids, split.control_ids, chamber_id=chamber
    )
    evidence = attribution.build_evidence(
        result, fdc, split.case_ids, split.control_ids, chamber_id=chamber
    )
    actions = recommend.build_actions(evidence, result, chamber_id=chamber)
    cf = recommend.counterfactual_from_actions(result, actions)
    return result, chamber, actions, cf


def main() -> int:
    parser = argparse.ArgumentParser(description="M5 — 개선안 및 기대효과")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--pattern", default=None, help="상세 분석할 패턴")
    parser.add_argument("--wafer-value", type=float, default=500.0,
                        help="웨이퍼 1장당 가치 (만원, 기본 500)")
    parser.add_argument("--monthly-wafers", type=int, default=3000)
    parser.add_argument("--adoption", type=float, default=0.8, help="조치 적용률 (0~1)")
    args = parser.parse_args()

    if not loader.is_built(args.source):
        print(f"❌ 데이터가 없습니다. python scripts/build_dataset.py --source {args.source}",
              file=sys.stderr)
        return 1

    wm = loader.load_wafer_master(args.source)
    fdc = loader.load_fdc_summary(args.source)
    assumptions = recommend.RoiAssumptions(
        wafer_value_krw=args.wafer_value * 10_000,
        monthly_wafers=args.monthly_wafers,
        adoption_rate=args.adoption,
    )

    print("=" * 78)
    print(f" M5 — 개선방안 및 기대효과   (source={args.source})")
    print("=" * 78)
    print("  M4가 '무엇이 원인인가'까지 왔다. 이제 '무엇을 할 것인가'와 '얼마를 버는가'다.")

    patterns = [args.pattern] if args.pattern else sorted(
        p for p, r in CAUSE_RULES.items() if r.step_id
    )
    total_gain = 0.0

    for pattern in patterns:
        out = analyze(wm, fdc, pattern)
        if out is None:
            print(f"\n❌ '{pattern}'은 원인 스텝이 정의되지 않았습니다.", file=sys.stderr)
            continue
        result, chamber, actions, cf = out
        step = STEPS_BY_ID[result.step_id]

        print(f"\n{'=' * 78}")
        print(f" {pattern} — {result.step_id} {step.name_ko} · {chamber} (AUC {result.auc:.3f})")
        print("=" * 78)
        print(recommend.summarize(actions, cf, None))

        if cf is not None:
            roi = recommend.estimate_roi(cf.yield_gain_pp, assumptions)
            total_gain += cf.yield_gain_pp
            print(f"\n  ROI: 연간 순효익 {roi.format_krw(roi.net_saving_krw)}"
                  f" · 추가 양품 {roi.recovered_wafers_year:,.0f}장", end="")
            print(f" · 회수 {roi.payback_months:.1f}개월" if roi.payback_months
                  else " · ⚠️ 회수 불가")

    # ── 전체 합산 ────────────────────────────────────────────────────────
    if len(patterns) > 1 and total_gain > 0:
        roi = recommend.estimate_roi(total_gain, assumptions)
        print(f"\n{'=' * 78}\n 전체 합산\n{'=' * 78}")
        print(f"  수율 향상분 합계     : +{total_gain:.2f}%p (적용률 반영 +{roi.effective_gain_pp:.2f}%p)")
        print(f"  연간 추가 양품       : {roi.recovered_wafers_year:,.0f}장")
        print(f"  연간 총 효익         : {roi.format_krw(roi.gross_saving_krw)}")
        print(f"  연간 순효익          : {roi.format_krw(roi.net_saving_krw)}")
        if roi.payback_months:
            print(f"  투자 회수 기간       : {roi.payback_months:.1f}개월")

        print("\n  ⚠️ 합산은 **패턴별 효과가 겹치지 않는다는 가정** 위에 있다.")
        print("     Center와 Scratch는 같은 P050 스텝이라 실제로는 일부 중복될 수 있다.")
        print("     그리고 이 금액의 자릿수는 전부 가정치(웨이퍼 단가·투입량)가 결정한다.")
        print("     민감도를 함께 보지 않은 ROI 단일 수치는 인용하지 말 것.")

        print(f"\n  민감도 — 웨이퍼 단가를 흔들면 (현재 {assumptions.wafer_value_krw / 10_000:,.0f}만원)")
        table = recommend.roi_sensitivity(total_gain, assumptions)
        print(f"    {'배수':>6}{'단가(만원)':>12}{'연간 순효익':>16}{'회수(개월)':>12}")
        for _, row in table.iterrows():
            pb = f"{row['payback_months']:.1f}" if row["payback_months"] else "회수불가"
            print(f"    {row['factor']:>6.2f}{row['value'] / 10_000:>12,.0f}"
                  f"{roi.format_krw(row['net_saving_krw']):>16}{pb:>12}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
