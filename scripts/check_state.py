#!/usr/bin/env python3
"""로컬 산출물이 지금 코드와 맞는지 한 번에 확인한다.

왜 필요한가 ★: 이 저장소는 `models/` 는 커밋하지만 `data/processed/` 는 커밋하지
    않는다(용량 때문). 그래서 `git pull` 을 하면 **새 모델 옆에 옛 데이터**가 남는
    조합이 아주 쉽게 생긴다. 그 상태로 앱을 켜면 화면마다 다른 이유로 실패하거나,
    더 나쁘게는 옛 수치를 보여 준다.

    "무엇을 다시 만들어야 하나"를 사람이 기억하게 두지 말고 도구가 답하게 한다.

사용법:
    python scripts/check_state.py
    python scripts/check_state.py --source real
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafermap import config as C  # noqa: E402

OK, BAD, WARN = "✅", "❌", "⚠️"


class Check:
    """확인 항목 하나. 실패하면 '다음에 칠 명령'을 함께 들고 있는다."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.status = OK
        self.detail = ""
        self.fix = ""

    def fail(self, detail: str, fix: str) -> "Check":
        self.status, self.detail, self.fix = BAD, detail, fix
        return self

    def warn(self, detail: str, fix: str = "") -> "Check":
        self.status, self.detail, self.fix = WARN, detail, fix
        return self

    def passed(self, detail: str) -> "Check":
        self.detail = detail
        return self


def check_dataset(source: str) -> Check:
    from wafermap.data import loader

    c = Check("데이터셋")
    if not loader.is_built(source):
        return c.fail(
            "없음",
            f"python scripts/build_dataset.py --source {source}",
        )
    master = loader.load_wafer_master(source)
    return c.passed(f"웨이퍼 {len(master):,}장 · lot {master['lot_id'].nunique():,}개")


def check_features(source: str) -> Check:
    from wafermap.features import build

    c = Check("피처 파일")
    if not build.is_built(source):
        return c.fail("없음", f"python scripts/build_features.py --source {source}")

    df = build.load(source)
    have = set(build.feature_columns(df))

    # 지금 코드가 실제로 내보내는 피처와 맞춰 본다 — 개수가 아니라 이름으로 본다.
    from wafermap.data import synth_wafer as sw
    import numpy as np

    geom = sw.build_geometry()
    expected = set(build.extract_one(
        sw.generate_wafer_map("Center", np.random.default_rng(0), geom, 0.8)
    ))

    missing, extra = expected - have, have - expected
    if missing or extra:
        bits = []
        if missing:
            bits.append(f"없는 것 {len(missing)}개 (예: {sorted(missing)[:3]})")
        if extra:
            bits.append(f"옛 피처가 남음 {len(extra)}개 (예: {sorted(extra)[:3]})")
        return c.fail(
            "코드와 어긋남 — " + " · ".join(bits),
            f"python scripts/build_features.py --source {source}",
        )
    return c.passed(f"{len(have)}종 · 코드와 일치")


def check_pattern_model(source: str) -> Check:
    from wafermap.features import build
    from wafermap.models import pattern_lgbm

    c = Check("패턴 분류 모델")
    meta_path = C.MODELS_DIR / source / pattern_lgbm.META_FILE
    if not meta_path.exists():
        return c.fail("없음", f"python scripts/train_pattern_model.py --source {source}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    wanted = meta["feature_columns"]

    if not build.is_built(source):
        return c.warn("피처 파일이 없어 대조 불가", "위 항목을 먼저 해결할 것")

    have = set(build.load(source).columns)
    missing = [col for col in wanted if col not in have]
    if missing:
        return c.fail(
            f"모델이 요구하는 피처 {len(missing)}개가 피처 파일에 없음 "
            f"(예: {missing[:3]})",
            f"python scripts/build_features.py --source {source} && "
            f"python scripts/train_pattern_model.py --source {source}",
        )
    f1 = meta["report"]["macro_f1"]
    return c.passed(f"피처 {len(wanted)}종 · macro-F1 {f1:.4f}")


def check_artifact(source: str, name: str, filename: str, script: str) -> Check:
    c = Check(name)
    path = C.processed_dir(source) / filename
    if not path.exists():
        return c.fail("없음", f"python scripts/{script} --source {source}")
    size_kb = path.stat().st_size / 1024
    return c.passed(f"{size_kb:,.0f} KB")


def check_generated_doc() -> Check:
    """생성 문서가 코드와 맞는지."""
    import subprocess

    c = Check("데이터 사전 문서")
    result = subprocess.run(
        [sys.executable, str(C.PROJECT_ROOT / "scripts" / "build_data_dictionary.py"),
         "--check"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        return c.fail("코드와 어긋남", "python scripts/build_data_dictionary.py")
    return c.passed("최신")


def main() -> int:
    parser = argparse.ArgumentParser(description="로컬 산출물 상태 점검")
    parser.add_argument("--source", choices=["synthetic", "real"], default="synthetic")
    args = parser.parse_args()
    src = args.source

    print("=" * 70)
    print(f" 로컬 상태 점검 — source={src}")
    print("=" * 70)

    checks = [
        check_dataset(src),
        check_features(src),
        check_pattern_model(src),
        check_artifact(src, "원인 분석 아티팩트", "rootcause.json", "build_rootcause.py"),
        check_artifact(src, "개선안 아티팩트", "recommendations.json",
                       "build_recommendations.py"),
        check_generated_doc(),
    ]

    for c in checks:
        print(f"\n{c.status} {c.name}")
        print(f"     {c.detail}")
        if c.fix:
            print(f"     → {c.fix}")

    broken = [c for c in checks if c.status == BAD]
    print("\n" + "=" * 70)
    if not broken:
        print(" 모두 최신입니다. `streamlit run app.py` 로 확인하세요.")
        return 0

    print(f" 다시 만들어야 할 것 {len(broken)}건. 아래를 순서대로 실행하세요:\n")
    # 파이프라인 순서를 지켜야 한다 — 피처가 바뀌면 그 아래가 전부 다시 필요하다.
    order = [
        f"python scripts/build_dataset.py --source {src}",
        f"python scripts/build_features.py --source {src}",
        f"python scripts/train_pattern_model.py --source {src}",
        f"python scripts/build_rootcause.py --source {src}",
        f"python scripts/build_recommendations.py --source {src}",
    ]
    first = min(order.index(c.fix.split(" && ")[0]) if c.fix.split(" && ")[0] in order
                else len(order) for c in broken)
    if first == len(order):
        for c in broken:
            print(f"   {c.fix}")
    else:
        print("   (앞 단계가 바뀌면 뒤 단계도 전부 다시 만들어야 합니다)")
        for cmd in order[first:]:
            print(f"   {cmd}")
    print()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
