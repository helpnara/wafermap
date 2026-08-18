#!/usr/bin/env python3
"""패턴 분류 모델 학습 — 교차검증 후 최종 모델을 저장한다.

사용법:
    python scripts/train_pattern_model.py
    python scripts/train_pattern_model.py --no-augment      # 증강 효과 비교용
    python scripts/train_pattern_model.py --source real
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from wafermap.config import PATTERN_LABELS  # noqa: E402
from wafermap.data import loader  # noqa: E402
from wafermap.features import build as feature_build  # noqa: E402
from wafermap.models import pattern_lgbm  # noqa: E402


def print_confusion(report: pattern_lgbm.CVReport) -> None:
    """혼동행렬을 읽기 쉽게 출력한다 (행=실제, 열=예측)."""
    labels = report.labels
    cm = np.array(report.confusion)
    short = [lb[:5] for lb in labels]

    print(f"\n  혼동행렬 (행=실제, 열=예측)")
    print("  " + " " * 12 + "".join(f"{s:>7}" for s in short))
    for i, label in enumerate(labels):
        row = "".join(f"{v:>7,}" if v else f"{'·':>7}" for v in cm[i])
        print(f"  {label:<12}{row}")


def main() -> int:
    parser = argparse.ArgumentParser(description="wafer map 패턴 분류 모델 학습")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rounds", type=int, default=400)
    parser.add_argument(
        "--augment-target", type=int, default=0,
        help=(
            "클래스당 최소 목표 장수 (회전·미러 증강 기준). 0이면 증강 없음(기본값).\n"
            "합성 데이터에서는 효과가 측정되지 않아 기본을 끔 — 자세한 근거는 "
            "docs/04_results.md 참고. 실측 데이터에서는 재평가할 것."
        ),
    )
    parser.add_argument("--no-augment", action="store_true", help="증강 완전 비활성화")
    args = parser.parse_args()

    augment_target = 0 if args.no_augment else args.augment_target

    print(f"{'=' * 70}\n 패턴 분류 모델 학습 — source={args.source}\n{'=' * 70}")

    if not feature_build.is_built(args.source):
        print(
            f"❌ 피처가 없습니다.\n"
            f"   먼저 실행: python scripts/build_features.py --source {args.source}",
            file=sys.stderr,
        )
        return 1

    features = feature_build.load(args.source)
    maps = None if augment_target == 0 else loader.load_die_maps(args.source)

    print(f"  웨이퍼 {len(features):,}장 · 피처 {len(feature_build.feature_columns(features))}개")
    print(f"  증강: {'없음' if augment_target == 0 else f'클래스당 최소 {augment_target}장'}")
    counts = features["pattern_label"].value_counts()
    print("  라벨 분포: " + ", ".join(f"{lb}={counts.get(lb, 0):,}" for lb in PATTERN_LABELS))

    print(f"\n🔁 {args.n_splits}-fold 교차검증")
    report = pattern_lgbm.cross_validate(
        features,
        maps,
        n_splits=args.n_splits,
        seed=args.seed,
        num_boost_round=args.rounds,
        augment_target=augment_target,
    )

    print(f"\n{'=' * 70}\n 교차검증 결과 ({report.train_seconds:.1f}초)\n{'=' * 70}")
    print(report.summary())
    print_confusion(report)

    print("\n🏗️  최종 모델 학습 (전체 데이터)")
    model, feature_cols = pattern_lgbm.fit_final(
        features,
        maps,
        seed=args.seed,
        num_boost_round=args.rounds,
        augment_target=augment_target,
    )
    out_dir = pattern_lgbm.save(model, feature_cols, report, args.source)

    print(f"\n  💾 저장 → {out_dir}")
    size_kb = (out_dir / pattern_lgbm.MODEL_FILE).stat().st_size / 1024
    print(f"     모델 크기: {size_kb:,.0f} KB")

    print("\n  피처 중요도 상위 15개:")
    for _, row in pattern_lgbm.feature_importance(model, feature_cols, 15).iterrows():
        print(f"    {row['feature']:<34} {row['gain']:>12,.0f}")

    target = 0.80
    status = "✅ 달성" if report.macro_f1 >= target else "⚠️ 미달"
    print(f"\n  성공 기준 macro-F1 ≥ {target}: {status} ({report.macro_f1:.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
