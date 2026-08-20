#!/usr/bin/env python3
"""M2 합성 ↔ 실측 대조 — "합성 성능은 상한선"이라는 주장을 숫자로 확인한다.

무엇을: 같은 코드로 학습한 두 모델(`models/synthetic`, `models/real`)의 교차검증
        결과를 나란히 놓고, 클래스별로 얼마나 떨어졌는지 보여 준다.

왜 필요한가 ★: 이 프로젝트는 화면과 문서 곳곳에 "합성 데이터라 실측보다 신호가
        깨끗하다, 이 점수는 상한선이다"라고 적어 두었다. 그건 지금까지 **주장**일
        뿐이었다. 상한선이라고 말하려면 실제로 얼마나 내려가는지를 재야 한다.
        떨어지는 폭 자체가 결과이고, 어떤 클래스가 특히 무너지는지가 더 중요하다.

사용법:
    # 1) 실측 데이터셋 생성 (LSWMD.pkl 필요 — docs/06_local_validation.md 참고)
    python scripts/download_wm811k.py
    python scripts/build_dataset.py  --source real --max-wafers 30000
    python scripts/build_features.py --source real
    python scripts/train_pattern_model.py --source real

    # 2) 대조
    python scripts/compare_sources.py
    python scripts/compare_sources.py --json docs/m2_real_vs_synthetic.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from wafermap.config import MODELS_DIR, PATTERN_LABELS  # noqa: E402
from wafermap.models import pattern_lgbm  # noqa: E402

#: 이 밑으로 떨어지면 "실측에서는 못 쓴다"고 봐야 하는 F1.
USABLE_F1 = 0.60


def _load_report(source: str) -> dict | None:
    """저장된 모델 메타에서 교차검증 결과를 읽는다."""
    meta_path = MODELS_DIR / source / pattern_lgbm.META_FILE
    if not meta_path.exists():
        return None
    return json.loads(meta_path.read_text(encoding="utf-8"))


def _table(syn: dict, real: dict) -> pd.DataFrame:
    """클래스별 F1을 나란히 놓고 낙폭을 낸다."""
    s_f1 = syn["report"]["per_class_f1"]
    r_f1 = real["report"]["per_class_f1"]
    s_n = syn["report"]["per_class_support"]
    r_n = real["report"]["per_class_support"]

    rows = []
    for label in PATTERN_LABELS:
        a, b = s_f1.get(label), r_f1.get(label)
        if a is None or b is None:
            continue
        rows.append({
            "패턴": label,
            "합성 F1": a,
            "실측 F1": b,
            "낙폭": b - a,
            "합성 표본": s_n.get(label, 0),
            "실측 표본": r_n.get(label, 0),
        })
    return pd.DataFrame(rows).sort_values("낙폭")


def _interpret(table: pd.DataFrame, syn_macro: float, real_macro: float) -> list[str]:
    """숫자를 읽는 법을 함께 낸다 — 표만 두면 보는 사람이 제 마음대로 읽는다."""
    lines: list[str] = []
    drop = real_macro - syn_macro
    lines.append(
        f"macro-F1 {syn_macro:.4f} → {real_macro:.4f} ({drop:+.4f})"
    )
    if drop < -0.05:
        lines.append(
            "  → 상한선이라는 주장이 확인됐다. 합성 성능을 그대로 인용하면 안 된다."
        )
    elif drop > -0.01:
        lines.append(
            "  → 거의 안 떨어졌다. 두 가지 중 하나다 — 피처가 정말 강건하거나, "
            "실측 표본이 너무 적어 쉬운 맵만 뽑혔거나. --max-wafers 를 늘려 다시 볼 것."
        )
    else:
        lines.append("  → 예상 범위의 하락이다.")

    worst = table.iloc[0]
    lines.append(
        f"가장 많이 무너진 패턴: {worst['패턴']} "
        f"({worst['합성 F1']:.3f} → {worst['실측 F1']:.3f}, 실측 표본 {worst['실측 표본']:,}장)"
    )
    unusable = table[table["실측 F1"] < USABLE_F1]
    if len(unusable):
        names = ", ".join(f"{r['패턴']}({r['실측 F1']:.2f})" for _, r in unusable.iterrows())
        lines.append(
            f"실측 F1 {USABLE_F1} 미만 — 이 클래스는 실무에 못 쓴다: {names}"
        )
        lines.append(
            "  → 표본이 적어서인지 형상이 정말 애매해서인지는 표본 수 열로 가른다."
        )
    else:
        lines.append(f"모든 클래스가 실측 F1 {USABLE_F1} 이상이다.")

    small = table[table["실측 표본"] < 100]
    if len(small):
        names = ", ".join(f"{r['패턴']}({r['실측 표본']:,})" for _, r in small.iterrows())
        lines.append(f"⚠️ 실측 표본 100장 미만이라 수치를 신뢰하기 어려운 클래스: {names}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description="M2 합성 ↔ 실측 성능 대조")
    parser.add_argument("--json", type=Path, default=None, help="결과를 JSON으로도 저장")
    args = parser.parse_args()

    syn, real = _load_report("synthetic"), _load_report("real")

    if syn is None:
        print("❌ 합성 모델이 없습니다.\n"
              "   먼저 실행: python scripts/train_pattern_model.py", file=sys.stderr)
        return 1
    if real is None:
        print(
            "❌ 실측 모델이 없습니다. 아래를 차례로 실행하세요:\n"
            "     python scripts/download_wm811k.py\n"
            "     python scripts/build_dataset.py  --source real --max-wafers 30000\n"
            "     python scripts/build_features.py --source real\n"
            "     python scripts/train_pattern_model.py --source real\n"
            "   자세한 절차는 docs/06_local_validation.md 를 보세요.",
            file=sys.stderr,
        )
        return 1

    # 피처 목록이 다르면 두 수치를 비교하는 것 자체가 성립하지 않는다.
    s_cols, r_cols = set(syn["feature_columns"]), set(real["feature_columns"])
    if s_cols != r_cols:
        only_s, only_r = sorted(s_cols - r_cols), sorted(r_cols - s_cols)
        print(
            "❌ 두 모델의 피처가 다릅니다 — 비교가 성립하지 않습니다.\n"
            f"   합성에만: {only_s[:5]}\n   실측에만: {only_r[:5]}\n"
            "   피처 코드를 고친 뒤 한쪽만 다시 만든 경우입니다. "
            "양쪽 모두 build_features.py 부터 다시 돌리세요.",
            file=sys.stderr,
        )
        return 1

    table = _table(syn, real)
    syn_macro = syn["report"]["macro_f1"]
    real_macro = real["report"]["macro_f1"]

    print("=" * 74)
    print(" M2 패턴 분류 — 합성 ↔ 실측(WM-811K) 대조")
    print("=" * 74)
    print(f"\n피처 {len(s_cols)}종 · 두 모델 모두 같은 코드 경로\n")

    show = table.copy()
    for col in ("합성 F1", "실측 F1", "낙폭"):
        show[col] = show[col].map(lambda v: f"{v:+.4f}" if col == "낙폭" else f"{v:.4f}")
    for col in ("합성 표본", "실측 표본"):
        show[col] = show[col].map(lambda v: f"{v:,}")
    print(show.to_string(index=False))

    print("\n" + "-" * 74)
    for line in _interpret(table, syn_macro, real_macro):
        print(line)
    print("-" * 74)

    print(
        "\n읽는 법: 낙폭이 큰 순서로 정렬돼 있다. 합성 데이터는 규칙으로 그린 맵이라\n"
        "경계가 또렷하지만, 실측에는 사람이 봐도 Edge-Loc인지 Edge-Ring인지 애매한\n"
        "맵이 많다. 그 애매함이 어디에 몰리는지가 이 표의 내용이다."
    )

    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "macro_f1": {"synthetic": syn_macro, "real": real_macro,
                                 "drop": real_macro - syn_macro},
                    "n_features": len(s_cols),
                    "per_class": table.to_dict(orient="records"),
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"\n💾 저장 → {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
