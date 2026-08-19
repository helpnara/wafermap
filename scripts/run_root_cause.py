#!/usr/bin/env python3
"""M4 실행 — FDC 파라미터 원인 규명 (SHAP) + 정답 대비 채점.

사용법:
    python scripts/run_root_cause.py                      # 전체 채점
    python scripts/run_root_cause.py --pattern Donut      # 한 패턴 상세
    python scripts/run_root_cause.py --no-actionable      # 필터 효과 비교
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# SHAP이 LightGBM 이진 분류에서 내는 출력 형식 경고 — 코드에서 이미 두 형식을 모두 처리한다
warnings.filterwarnings("ignore", message=".*binary classifier with TreeExplainer.*")

from wafermap.analysis import attribution, commonality, spc, validate  # noqa: E402
from wafermap.config import CAUSE_RULES, STEPS_BY_ID  # noqa: E402
from wafermap.data import loader  # noqa: E402
from wafermap.models import cause_model  # noqa: E402


def run_pattern(wm, fdc, pattern: str, *, stratify: bool = True) -> None:
    """패턴 하나를 상세 분석하고 해석과 함께 출력한다."""
    rule = CAUSE_RULES.get(pattern)
    if rule is None or rule.step_id is None:
        print(f"❌ '{pattern}'은 원인 스텝이 정의되지 않은 패턴입니다.", file=sys.stderr)
        return

    step = STEPS_BY_ID[rule.step_id]
    split = spc.split_by_pattern(wm, pattern)

    print(f"\n{'=' * 78}\n [1] 분석 대상 — {pattern}\n{'=' * 78}")
    print(f"  원인 스텝: {rule.step_id} {step.name_ko} ({step.name_en})")
    print(f"  물리적 기전: {rule.mechanism}")
    print(f"  표본: 불량 {len(split.case_ids)}장 · 정상 {len(split.control_ids)}장")

    # ── M3로 진범 챔버 찾기 (층화용) ────────────────────────────────────
    chamber = None
    if stratify:
        ranking = commonality.analyze(
            fdc, split.case_ids, split.control_ids, wafer_master=wm, unit="lot"
        )
        same_step = ranking[ranking["step_id"] == rule.step_id]
        if not same_step.empty:
            chamber = str(same_step.iloc[0]["chamber_id"])
            print(f"  층화 챔버: {chamber} (M3 커미널리티 1위)")
            print("     └ 챔버를 고정하면 챔버 간 baseline 차이가 제거되어 신호가 선명해진다")

    # ── 모델 학습 + SHAP ────────────────────────────────────────────────
    print(f"\n{'=' * 78}\n [2] 원인 파라미터 규명 (LightGBM + SHAP)\n{'=' * 78}")
    try:
        result = cause_model.fit(
            fdc, rule.step_id, pattern, split.case_ids, split.control_ids,
            chamber_id=chamber,
        )
    except ValueError as exc:
        print(f"  ❌ {exc}", file=sys.stderr)
        return

    print(f"  모델 AUC: {result.auc:.3f}")
    if result.auc < 0.7:
        print("     ⚠️ AUC가 0.7 미만이다. 모델이 신호를 제대로 잡지 못했으므로")
        print("        아래 SHAP 결과는 노이즈를 분해한 것일 수 있다. 해석에 주의할 것.")
    else:
        print("     └ 모델이 불량/정상을 구분하는 신호를 잡았다. SHAP 해석이 의미를 갖는다.")

    evidence = attribution.build_evidence(
        result, fdc, split.case_ids, split.control_ids, chamber_id=chamber
    )
    print()
    print(attribution.summarize(evidence, top_n=8))

    # ── 조치 가능한 것만 ────────────────────────────────────────────────
    actionable = attribution.actionable_ranking(evidence)
    if evidence and not evidence[0].is_controllable:
        print(f"\n  ⚠️ SHAP 1위 '{evidence[0].column}'는 **계측값**이라 조작할 수 없다.")
        print("     계측값은 공정 이상의 '결과'다. 조치하려면 상류 파라미터를 봐야 한다.")
        if actionable:
            print(f"     → 조치 대상 1위: {actionable[0].column}")

    # ── 개선안 근거 ─────────────────────────────────────────────────────
    if actionable:
        target = actionable[0]
        print(f"\n{'=' * 78}\n [3] 개선안 근거 — {target.column}\n{'=' * 78}")

        curve = cause_model.dependence_curve(result, target.column)
        if not curve.empty:
            print(f"  {'구간':<24}{'평균 SHAP':>12}{'표본':>7}{'불량률':>9}")
            print("  " + "-" * 54)
            for _, row in curve.iterrows():
                bar = "█" * max(0, int(row["mean_shap"] * 8))
                print(
                    f"  {row['value_lo']:>9.2f} ~ {row['value_hi']:<11.2f}"
                    f"{row['mean_shap']:>12.3f}{row['n_samples']:>7}"
                    f"{row['case_rate']:>8.0%}  {bar}"
                )

        spec = cause_model.recommend_spec(result, target.column)
        if spec:
            print(f"\n  권고 운전 구간: {spec['low']:.2f} ~ {spec['high']:.2f}")
            print(f"  현재 관측 범위: {spec['current_min']:.2f} ~ {spec['current_max']:.2f}")
            print(f"  권고 구간이 덮는 표본: {spec['coverage']:.0%}")
            print("\n  ⚠️ 한계: 이 권고는 **관측된 범위 안에서만** 유효하다.")
            print("     한 번도 본 적 없는 값에 대해서는 모델이 근거 없이 외삽한다.")
            print("     그리고 상관 기반이므로 '이 구간으로 옮기면 좋아진다'는 인과 보장이 아니다.")

    # ── 개별 웨이퍼 설명 ────────────────────────────────────────────────
    case_rows = [i for i, y in enumerate(result.y) if y == 1]
    if case_rows:
        print(f"\n{'=' * 78}\n [4] 개별 웨이퍼 판정 근거 (local SHAP)\n{'=' * 78}")
        frame = cause_model.explain_wafer(result, case_rows[0], top_n=4)
        print(f"  {'파라미터':<30}{'값':>12}{'기여도':>10}  방향")
        print("  " + "-" * 62)
        for _, row in frame.iterrows():
            print(
                f"  {row['parameter']:<30}{row['value']:>12.2f}"
                f"{row['shap']:>+10.3f}  {row['direction']}"
            )
        print("\n  📖 현업에서 엔지니어에게 설명할 때 가장 설득력 있는 형태다.")


def main() -> int:
    parser = argparse.ArgumentParser(description="M4 — FDC 파라미터 원인 규명")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--pattern", default=None, help="상세 분석할 패턴 (생략 시 전체 채점만)")
    parser.add_argument("--no-stratify", action="store_true", help="챔버 층화 비활성화")
    parser.add_argument(
        "--no-actionable", action="store_true",
        help="조치 가능 필터 없이 채점 (필터 효과 비교용)",
    )
    args = parser.parse_args()

    if not loader.is_built(args.source):
        print(f"❌ 데이터가 없습니다. python scripts/build_dataset.py --source {args.source}",
              file=sys.stderr)
        return 1

    wm = loader.load_wafer_master(args.source)
    fdc = loader.load_fdc_summary(args.source)

    print("=" * 78)
    print(f" M4 — FDC 파라미터 원인 규명   (source={args.source})")
    print("=" * 78)
    print("  M3가 '어느 챔버'까지 좁혔다. 이제 '그 챔버의 무엇이 문제인가'를 찾는다.")

    if args.pattern:
        run_pattern(wm, fdc, args.pattern, stratify=not args.no_stratify)

    # ── 전체 채점 ────────────────────────────────────────────────────────
    print(f"\n{'=' * 78}\n [5] 전체 채점 — 정답지 대비\n{'=' * 78}")
    report = validate.score_parameters(
        wm, fdc,
        stratify_by_chamber=not args.no_stratify,
        actionable_only=not args.no_actionable,
    )
    for score in report.scores:
        print(f"  {score.describe()}")
    print()
    print("  " + report.summary().replace("\n", "\n  "))

    if args.no_actionable:
        print("\n  (조치 가능 필터 없이 채점한 결과다. 필터를 켜면 계측값이 제외된다.)")

    n_measurement = sum(s.top1_is_measurement for s in report.scores)
    if n_measurement:
        print(f"\n  📖 {n_measurement}개 패턴에서 SHAP 1위가 **계측값**이었다.")
        print("     계측값은 공정 이상의 '결과'라 조작할 수 없다. 조치 대상이 되려면")
        print("     조작 가능한 파라미터여야 하므로, 필터가 순위를 바로잡았다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
