#!/usr/bin/env python3
"""피처 추출 — die_map에서 피처 행렬(features.parquet)을 만든다.

사용법:
    python scripts/build_features.py                  # 합성 데이터
    python scripts/build_features.py --source real    # 실측 데이터
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafermap.data import loader  # noqa: E402
from wafermap.features import build  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="웨이퍼 맵 피처 추출")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    parser.add_argument(
        "--n-jobs", type=int, default=1,
        help="병렬 프로세스 수 (-1=CPU 전체). 실측 데이터처럼 대량일 때 올린다.",
    )
    args = parser.parse_args()

    print(f"{'=' * 70}\n 피처 추출 — source={args.source}\n{'=' * 70}")

    if not loader.is_built(args.source):
        print(
            f"❌ 데이터셋이 없습니다.\n"
            f"   먼저 실행하세요: python scripts/build_dataset.py --source {args.source}",
            file=sys.stderr,
        )
        return 1

    t0 = time.time()
    print("📂 맵 로드")
    maps = loader.load_die_maps(args.source)
    master = loader.load_wafer_master(args.source)
    labels = master.set_index("wafer_id")["pattern_label"]

    print(f"🔧 피처 추출 ({len(maps):,}장)")
    df = build.build(maps, labels, n_jobs=args.n_jobs)

    out = build.save(df, args.source)
    feature_cols = build.feature_columns(df)

    print(f"\n{'=' * 70}\n 완료 ({time.time() - t0:.1f}초) → {out}\n{'=' * 70}")
    print(f"  웨이퍼   : {len(df):,}장")
    print(f"  피처     : {len(feature_cols)}개")
    print(f"  파일 크기: {out.stat().st_size / 1024**2:.2f} MB")

    # 피처 그룹별 개수 — 설계서 §M1의 그룹 구성과 대조하기 쉽게
    groups = {
        "반경(ring)": sum(c.startswith("ring") or "radial" in c or "ratio" in c for c in feature_cols),
        "각도(sector)": sum(c.startswith("sector") or "angular" in c for c in feature_cols),
        "13-zone": sum(c.startswith("zone") for c in feature_cols),
        "모멘트(hu)": sum(c.startswith("hu") or "fail_" in c for c in feature_cols),
        "연결성분": sum("cluster" in c for c in feature_cols),
        "Radon": sum(c.startswith("radon") for c in feature_cols),
    }
    print("\n  피처 그룹:")
    for name, cnt in groups.items():
        print(f"    {name:<14} {cnt:>3}개")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
