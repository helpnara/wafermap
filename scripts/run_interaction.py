#!/usr/bin/env python3
"""M5.5-③ 실행 — 각각은 규격 안인데 조합이 문제인 불량을 찾는다.

첨부 도메인 문서 §14·15·16·37의 주제다.

    증착이 두께 상한 쪽  +  CMP 제거량이 하한 쪽  →  중심부 잔막
    (규격 안)              (규격 안)                  (불량)

주효과가 0이므로 단변량 SPC도, 규격 위반 검사도, 파라미터별 평균 비교도 조용하다.

사용법:
    python scripts/run_interaction.py
    python scripts/run_interaction.py --shap      # SHAP interaction까지 (느림)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

warnings.filterwarnings("ignore", message=".*binary classifier with TreeExplainer.*")

from wafermap.analysis import interaction, validate  # noqa: E402
from wafermap.config import INTERACTION_CAUSE_RULES, STEPS_BY_ID  # noqa: E402
from wafermap.data import loader  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="M5.5-③ 공정 간 교호작용")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--shap", action="store_true", help="SHAP interaction도 계산")
    args = parser.parse_args()

    if not loader.is_built(args.source):
        print(f"❌ 데이터가 없습니다. python scripts/build_dataset.py --source {args.source}",
              file=sys.stderr)
        return 1

    fdc = loader.load_fdc_summary(args.source)
    gt = loader.load_ground_truth(args.source)

    print("=" * 78)
    print(f" M5.5-③ 공정 간 교호작용   (source={args.source})")
    print("=" * 78)
    print("  지금까지의 분석은 전부 '어느 하나가 이탈했나'를 물었다.")
    print("  각각은 규격 안인데 **조합**이 문제인 경우는 그 질문으로 못 찾는다.")

    control = list(gt.loc[gt["pattern_label"] == "none", "wafer_id"])
    all_ids = list(gt["wafer_id"])

    for pattern, rule in INTERACTION_CAUSE_RULES.items():
        sub = gt[gt["pattern_label"] == pattern]
        case = list(sub.loc[sub["cause_mechanism"] == "interaction", "wafer_id"])

        print(f"\n{'=' * 78}\n [1] 심어 둔 교호작용 — {pattern} ({len(case)}장)\n{'=' * 78}")
        print(f"  기전: {rule.mechanism}")
        for step, param in ((rule.step_a, rule.param_a), (rule.step_b, rule.param_b)):
            spec = STEPS_BY_ID[step].param(param)
            headroom = min(
                (spec.nominal - spec.spec_lo) / spec.sigma,
                (spec.spec_hi - spec.nominal) / spec.sigma,
            )
            print(f"    {step}.{param:<20} 규격 폭 ±{headroom:.1f}σ · "
                  f"주입 {rule.shift_range[0]}~{rule.shift_range[1]}σ → 대부분 규격 안")
        print(f"  설비 조합: {rule.equip_pair[0]} → {rule.equip_pair[1]}")

        ids = case + control
        matrix = interaction.cross_step_matrix(
            fdc, ids, steps=(rule.step_a, rule.step_b)
        )
        matrix = matrix.loc[[w for w in ids if w in matrix.index]]
        is_case = matrix.index.isin(set(case))

        print(f"\n{'=' * 78}\n [2] 파라미터 쌍 탐색 (2×2 분할표)\n{'=' * 78}")
        pairs = interaction.screen_pairs(matrix, is_case, top_n=100)
        hidden = [p for p in pairs if p.hidden_from_main_effects]
        print(f"  후보 {len(pairs)}쌍 → 주효과로 안 보이는 쌍 ★{len(hidden)}개\n")
        print(interaction.summarize(hidden, top_n=3))

        target = next(
            (p for p in pairs
             if {p.param_a, p.param_b}
             == {f"{rule.step_a}.{rule.param_a}", f"{rule.step_b}.{rule.param_b}"}),
            None,
        )
        if target is not None:
            print(f"\n  정답 쌍의 2×2 표 (칸 안은 불량률):\n")
            print("  " + target.table().replace("\n", "\n  "))
            print(
                f"\n  대각선 두 칸만 높다 → **같은 방향으로 치우칠 때만** 문제라는 뜻이다.\n"
                f"  한 줄만 높았다면 그건 교호작용이 아니라 그 파라미터의 주효과다."
            )

        print(f"\n{'=' * 78}\n [3] 설비 조합 (앞 챔버 → 뒤 챔버)\n{'=' * 78}")
        equips = interaction.equipment_pairs(
            fdc, case, all_ids, step_a=rule.step_a, step_b=rule.step_b, top_n=3
        )
        for equip in equips:
            mark = " ← 정답" if (equip.chamber_a, equip.chamber_b) == rule.equip_pair else ""
            print("  " + equip.describe() + mark)
        print('\n  📖 "B가 문제다"가 아니라 "**A → B 조합**에서 증가한다"이다.')
        print("     조치도 달라진다 — B를 세우는 대신 그 조합만 피하도록")
        print("     디스패치 규칙을 바꾸면 된다. 처리량 손실이 훨씬 작다.")

        if args.shap:
            print(f"\n{'=' * 78}\n [4] SHAP interaction values\n{'=' * 78}")
            print(interaction.shap_pairs(matrix, is_case, top_n=5).describe())
            print("\n  ⚠️ 계산량이 피처 수의 제곱에 비례한다. 후보를 좁히는 용도이고,")
            print("     확인은 2×2 표로 하는 편이 설명하기 쉽다.")

    print(f"\n{'=' * 78}\n [5] 채점 — 정답지 대비\n{'=' * 78}")
    report = validate.score_interaction(fdc, gt)
    for score in report.scores:
        print("  " + score.describe().replace("\n", "\n  "))
    print()
    print("  " + report.summary().replace("\n", "\n  "))

    print("\n  ⚠️ 한계: 쌍의 수는 파라미터 수의 제곱으로 늘어난다. 다중검정을 보정해도")
    print("     우연히 유의한 쌍은 반드시 나온다. **물리적으로 말이 되는지**를")
    print("     공정 엔지니어가 반드시 확인해야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
