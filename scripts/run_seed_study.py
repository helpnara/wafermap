#!/usr/bin/env python3
"""여러 시드로 같은 채점을 반복해 **성능의 분포**를 본다.

왜 필요한가 ★★: 합성 데이터의 채점 결과는 시드 하나에 크게 좌우된다. 이상 사건이
    어느 챔버에 몇 건 걸리느냐가 매번 달라지기 때문이다. 시드 하나에서 나온
    "적중률 100%"를 성능이라고 보고하면, 다른 시드로 돌린 사람이 67%를 보고
    **재현이 안 된다**고 판단하게 된다.

    단일 수치가 아니라 평균 ± 표준편차로 말해야 한다.

사용법:
    python scripts/run_seed_study.py                    # M3, 5개 시드
    python scripts/run_seed_study.py --milestone m4 --n-seeds 3
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

warnings.filterwarnings("ignore", message=".*binary classifier with TreeExplainer.*")

import numpy as np  # noqa: E402

from wafermap.analysis import validate  # noqa: E402
from wafermap.data.fdc_simulator import simulate  # noqa: E402

DEFAULT_SEEDS = (42, 7, 1234, 99, 2026, 31337, 555, 8)


def report(name: str, values: list[float]) -> None:
    arr = np.array(values, dtype=float)
    detail = " ".join(f"{v:.0%}" for v in arr)
    print(f"  {name:<26}{arr.mean():>6.0%} ± {arr.std():>3.0%}   [{detail}]")


def main() -> int:
    parser = argparse.ArgumentParser(description="시드 반복 채점")
    parser.add_argument("--milestone", choices=["m3", "m4"], default="m3")
    parser.add_argument("--n-seeds", type=int, default=5)
    parser.add_argument("--n-wafers", type=int, default=6_000)
    args = parser.parse_args()

    seeds = DEFAULT_SEEDS[: args.n_seeds]
    print("=" * 78)
    print(f" 시드 반복 채점 — {args.milestone.upper()} · 시드 {len(seeds)}개 · "
          f"웨이퍼 {args.n_wafers:,}장")
    print("=" * 78)
    print("  합성 데이터의 채점은 시드에 크게 흔들린다. 단일 수치가 아니라")
    print("  분포로 보고해야 다른 사람이 돌렸을 때 재현된다.\n")

    collected: dict[str, list[float]] = {}
    for seed in seeds:
        res = simulate(n_wafers=args.n_wafers, n_days=120, seed=seed)
        if args.milestone == "m3":
            rep = validate.score_all(res.wafer_master, res.fdc_summary, res.ground_truth)
            metrics = {
                "원인 스텝 Top-1": [s.top1_step_hit for s in rep.scores],
                "원인 챔버 Top-1": [s.top1_chamber_hit for s in rep.scores],
                "원인 챔버 Top-3": [s.top3_chamber_hit for s in rep.scores],
            }
        else:
            rep = validate.score_parameters(res.wafer_master, res.fdc_summary)
            metrics = {
                "파라미터 Top-1": [s.top1_hit for s in rep.scores],
                "파라미터 Top-3": [s.top3_hit for s in rep.scores],
                "파라미터 Top-5": [s.top5_hit for s in rep.scores],
            }
        for key, hits in metrics.items():
            collected.setdefault(key, []).append(float(np.mean(hits)))
        print(f"  시드 {seed:>6} 완료")

    print(f"\n{'=' * 78}\n 결과\n{'=' * 78}")
    for key, values in collected.items():
        report(key, values)

    spread = max(np.std(v) for v in collected.values())
    print(f"\n  📖 시드 간 표준편차가 최대 {spread:.0%}다.")
    print("     이보다 작은 차이는 **개선인지 우연인지 구분할 수 없다.**")
    print("     설정을 바꿔 비교할 때는 반드시 같은 시드 집합으로 나란히 재야 한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
