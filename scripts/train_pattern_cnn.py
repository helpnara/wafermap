#!/usr/bin/env python3
"""경량 CNN 학습 — LightGBM과 같은 방식으로 교차검증하고 결과를 비교한다.

사용법:
    python scripts/train_pattern_cnn.py
    python scripts/train_pattern_cnn.py --epochs 40 --n-splits 5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafermap.config import PATTERN_LABELS  # noqa: E402
from wafermap.data import loader  # noqa: E402
from wafermap.models import pattern_cnn, pattern_lgbm  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="wafer map 패턴 분류 CNN 학습")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"{'=' * 70}\n 패턴 분류 CNN 학습 — source={args.source}\n{'=' * 70}")

    if not loader.is_built(args.source):
        print(
            f"❌ 데이터셋이 없습니다.\n"
            f"   먼저 실행: python scripts/build_dataset.py --source {args.source}",
            file=sys.stderr,
        )
        return 1

    maps = loader.load_die_maps(args.source)
    master = loader.load_wafer_master(args.source)
    labels = master.set_index("wafer_id")["pattern_label"]

    model_probe = pattern_cnn.make_model()
    n_params = sum(p.numel() for p in model_probe.parameters())
    print(f"  웨이퍼 {len(maps):,}장 · 입력 {pattern_cnn.IMG_SIZE}×{pattern_cnn.IMG_SIZE}×2채널")
    print(f"  파라미터 {n_params:,}개 · epoch {args.epochs}")

    print(f"\n🔁 {args.n_splits}-fold 교차검증 (CPU 학습이라 시간이 걸립니다)")
    report = pattern_cnn.cross_validate(
        maps, labels, n_splits=args.n_splits, seed=args.seed, epochs=args.epochs
    )

    print(f"\n{'=' * 70}\n CNN 교차검증 결과 ({report.train_seconds:.1f}초)\n{'=' * 70}")
    print(report.summary())

    print("\n🏗️  최종 모델 학습 (전체 데이터)")
    X, y = pattern_cnn.build_dataset(maps, labels)
    model = pattern_cnn.train_one(X, y, epochs=args.epochs, seed=args.seed, verbose=True)
    out_dir = pattern_cnn.save(model, report, args.source)

    size_mb = (out_dir / pattern_cnn.MODEL_FILE).stat().st_size / 1024**2
    print(f"\n  💾 저장 → {out_dir}  ({size_mb:.2f} MB)")

    # ── LightGBM과 비교 ──────────────────────────────────────────────────
    try:
        _, lgbm_meta = pattern_lgbm.load(args.source)
        lgbm = lgbm_meta["report"]
    except FileNotFoundError:
        print("\n  ℹ️ LightGBM 모델이 없어 비교를 건너뜁니다.")
        return 0

    print(f"\n{'=' * 70}\n 모델 비교\n{'=' * 70}")
    print(f"  {'':<12}{'LightGBM':>12}{'CNN':>12}{'차이':>10}")
    print("  " + "-" * 46)
    print(
        f"  {'macro-F1':<12}{lgbm['macro_f1']:>12.4f}{report.macro_f1:>12.4f}"
        f"{report.macro_f1 - lgbm['macro_f1']:>+10.4f}"
    )
    print(
        f"  {'학습 시간':<11}{lgbm['train_seconds']:>11.1f}초{report.train_seconds:>11.1f}초"
    )
    print(f"\n  {'클래스':<12}{'LGBM F1':>10}{'CNN F1':>10}{'차이':>10}")
    print("  " + "-" * 44)
    for label in PATTERN_LABELS:
        a = lgbm["per_class_f1"][label]
        b = report.per_class_f1[label]
        print(f"  {label:<12}{a:>10.3f}{b:>10.3f}{b - a:>+10.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
