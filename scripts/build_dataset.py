#!/usr/bin/env python3
"""데이터셋 생성 — 합성 또는 실측 원본으로부터 processed 계층을 일괄 생성한다.

무엇을: 시뮬레이터/원본 로더를 돌려 wafer_master, die_map, fdc_summary,
        fdc_trace, ground_truth, excursions를 만들고 스키마 검증까지 마친다.

파이프라인:
    [합성]  FDC 시뮬레이터(타임라인·라우팅·이상사건·FDC) ─┐
                                                        ├─→ 맵 렌더링 → 검증 → 저장
    [실측]  WM-811K 로더(맵·라벨) + FDC 시뮬레이터(공정) ─┘

왜 실측에도 시뮬레이터를 쓰나: WM-811K에는 공정/설비 정보가 아예 없다(§2.1).
    실측 맵의 **패턴 라벨을 그대로 받아서**, 그 라벨에 맞는 공정 이력을 시뮬레이터가
    역으로 구성한다. 맵은 진짜, 공정은 합성이라는 설계(§2.2)가 여기서 구현된다.

사용법:
    python scripts/build_dataset.py                          # 합성 6,000장
    python scripts/build_dataset.py --n-wafers 20000
    python scripts/build_dataset.py --source real            # 실측 (로컬 전용)
    python scripts/build_dataset.py --source real --max-wafers 30000
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafermap.config import DEFAULT_SEED, processed_dir  # noqa: E402
from wafermap.data import loader, schema, wm811k  # noqa: E402
from wafermap.data.fdc_simulator import SimulationResult, simulate  # noqa: E402
from wafermap.data.synth_wafer import build_geometry, generate_wafer_map, map_stats  # noqa: E402


def _render_maps(
    master: pd.DataFrame, ground_truth: pd.DataFrame, seed: int
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    """패턴 라벨에 맞는 wafer map을 렌더링하고 수율 통계를 채운다.

    어떻게: ground_truth의 severity를 맵 심각도로 그대로 넘긴다.
    왜:    FDC 파라미터 이탈 정도와 맵 불량 정도가 **같은 severity를 공유해야**
           "파라미터가 심하게 튄 웨이퍼일수록 맵도 심하다"는 상관이 생긴다.
           이 연결이 없으면 원인 분석 모델이 학습할 신호 자체가 없다.
    """
    geom = build_geometry()
    rng = np.random.default_rng(seed + 1)

    sev_by_id = dict(zip(ground_truth["wafer_id"], ground_truth["severity"]))

    maps: dict[str, np.ndarray] = {}
    die_total, die_pass, yield_pct = [], [], []

    for wafer_id, pattern in zip(master["wafer_id"], master["pattern_label"]):
        sev = sev_by_id.get(wafer_id, np.nan)
        sev = 1.0 if (sev is None or (isinstance(sev, float) and np.isnan(sev))) else float(sev)

        wafer = generate_wafer_map(pattern, rng, geom, severity=sev)
        maps[wafer_id] = wafer

        dt, dp, y = map_stats(wafer)
        die_total.append(dt)
        die_pass.append(dp)
        yield_pct.append(y)

    master = master.copy()
    master["die_total"] = np.asarray(die_total, dtype="int64")
    master["die_pass"] = np.asarray(die_pass, dtype="int64")
    master["yield_pct"] = np.asarray(yield_pct, dtype="float64")
    return maps, master


def build_synthetic(
    n_wafers: int, n_days: int, seed: int, trace_sample_rate: float
) -> tuple[SimulationResult, dict[str, np.ndarray]]:
    """합성 데이터셋을 생성한다."""
    print(f"⚙️  FDC 시뮬레이션: {n_wafers:,}장 / {n_days}일")
    res = simulate(
        n_wafers=n_wafers, n_days=n_days, seed=seed, trace_sample_rate=trace_sample_rate
    )

    print("🗺️  wafer map 렌더링")
    maps, master = _render_maps(res.wafer_master, res.ground_truth, seed)
    res.wafer_master = master
    return res, maps


def build_real(
    max_wafers: int | None, n_days: int, seed: int, trace_sample_rate: float
) -> tuple[SimulationResult, dict[str, np.ndarray]]:
    """실측 WM-811K 맵 + 합성 공정 이력으로 데이터셋을 생성한다.

    어떻게: 원본에서 맵과 라벨을 읽은 뒤, **같은 라벨 구성**을 갖도록 시뮬레이터를
            돌리고 두 결과를 라벨별로 짝지어 붙인다.
    왜 짝짓기가 필요한가: 시뮬레이터는 자체 목표 분포로 라벨을 배정하므로, 실측
            라벨과 장수가 정확히 일치하지 않는다. 라벨별로 매칭해야 "Edge-Ring
            웨이퍼에는 Edge-Ring을 유발한 Etch 이력이 붙는다"는 대응이 성립한다.
    """
    print("📂 WM-811K 원본 로드")
    real_master, real_maps = wm811k.load(
        labeled_only=True, max_wafers=max_wafers, seed=seed
    )
    print(f"   {len(real_master):,}장 로드 완료")

    print("⚙️  공정 이력 시뮬레이션 (맵은 실측, 공정은 합성)")
    res = simulate(
        n_wafers=len(real_master),
        n_days=n_days,
        seed=seed,
        trace_sample_rate=trace_sample_rate,
    )

    # 라벨별로 실측 웨이퍼 ↔ 시뮬레이션 웨이퍼를 짝짓는다
    rng = np.random.default_rng(seed)
    sim_by_label: dict[str, list[str]] = {
        label: list(grp["wafer_id"])
        for label, grp in res.wafer_master.groupby("pattern_label")
    }
    for ids in sim_by_label.values():
        rng.shuffle(ids)

    mapping: dict[str, str] = {}  # 시뮬 wafer_id → 실측 wafer_id
    unmatched: list[str] = []
    for real_id, label in zip(real_master["wafer_id"], real_master["pattern_label"]):
        pool = sim_by_label.get(label)
        if pool:
            mapping[pool.pop()] = real_id
        else:
            unmatched.append(real_id)

    if unmatched:
        # 라벨이 남는 경우(시뮬 쪽 장수가 모자람) — 남은 시뮬 웨이퍼를 라벨 무시하고 배정.
        # 공정 이력이 라벨과 어긋나므로, 이 웨이퍼는 원인 분석 정답이 없는 것으로 표시된다.
        leftovers = [w for pool in sim_by_label.values() for w in pool]
        rng.shuffle(leftovers)
        for real_id, sim_id in zip(unmatched, leftovers):
            mapping[sim_id] = real_id
        print(f"   ⚠️ 라벨 매칭 실패 {len(unmatched):,}장 — 공정 이력을 임의 배정")

    # 시뮬 wafer_id를 실측 wafer_id로 치환
    for df_name in ("wafer_master", "fdc_summary", "fdc_trace", "ground_truth"):
        df = getattr(res, df_name)
        df = df[df["wafer_id"].isin(mapping)].copy()
        df["wafer_id"] = df["wafer_id"].map(mapping)
        setattr(res, df_name, df)

    # 실측 라벨·맵 통계로 마스터를 덮어쓴다 (맵이 진실의 원천)
    real_indexed = real_master.set_index("wafer_id")
    res.wafer_master = res.wafer_master.set_index("wafer_id")
    for col in ("pattern_label", "die_total", "die_pass", "yield_pct", "data_source", "is_labeled"):
        res.wafer_master[col] = real_indexed[col]
    res.wafer_master = res.wafer_master.reset_index()

    res.ground_truth = res.ground_truth.set_index("wafer_id")
    res.ground_truth["pattern_label"] = real_indexed["pattern_label"]
    res.ground_truth = res.ground_truth.reset_index()

    maps = {wid: real_maps[wid] for wid in res.wafer_master["wafer_id"]}
    return res, maps


def _excursions_frame(res: SimulationResult) -> pd.DataFrame:
    """이상 사건 목록을 저장 가능한 표로 변환한다."""
    if not res.excursions:
        return pd.DataFrame(
            columns=["excursion_id", "pattern", "step_id", "chamber_id",
                     "t_start", "t_end", "attack_rate"]
        )
    return pd.DataFrame(
        [
            {
                "excursion_id": e.excursion_id,
                "pattern": e.pattern,
                "step_id": e.step_id,
                "chamber_id": e.chamber_id,
                "t_start": e.t_start,
                "t_end": e.t_end,
                "attack_rate": e.attack_rate,
            }
            for e in res.excursions
        ]
    )


def validate_all(res: SimulationResult) -> None:
    """저장 전에 스키마와 테이블 간 정합성을 검사한다.

    왜 저장 전인가: 잘못된 데이터를 디스크에 남기면, 며칠 뒤 모델 성능이 이상할 때
        원인이 데이터인지 모델인지 되짚어야 한다. 생성 시점에 막는 게 가장 싸다(§schema).
    """
    print("🔍 스키마 검증")
    schema.validate(res.wafer_master, schema.WAFER_MASTER)
    schema.validate(res.fdc_summary, schema.FDC_SUMMARY)
    if len(res.fdc_trace):
        schema.validate(res.fdc_trace, schema.FDC_TRACE)
    schema.validate(res.ground_truth, schema.GROUND_TRUTH)
    schema.validate_consistency(res.wafer_master, res.fdc_summary, res.ground_truth)
    print("   ✅ 통과")


def save(res: SimulationResult, maps: dict[str, np.ndarray], source: str) -> Path:
    """processed 디렉터리에 저장한다."""
    out = processed_dir(source)
    out.mkdir(parents=True, exist_ok=True)

    res.wafer_master.to_parquet(out / loader.WAFER_MASTER_FILE, index=False)
    res.fdc_summary.to_parquet(out / loader.FDC_SUMMARY_FILE, index=False)
    res.fdc_trace.to_parquet(out / loader.FDC_TRACE_FILE, index=False)
    res.ground_truth.to_parquet(out / loader.GROUND_TRUTH_FILE, index=False)
    _excursions_frame(res).to_parquet(out / loader.EXCURSION_FILE, index=False)
    np.savez_compressed(out / loader.DIE_MAP_FILE, **maps)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="EDS 분석용 데이터셋 생성")
    parser.add_argument(
        "--source", choices=["synthetic", "real"], default="synthetic",
        help="synthetic=합성 맵(기본) / real=WM-811K 실측 맵 (로컬 전용)",
    )
    parser.add_argument("--n-wafers", type=int, default=6_000, help="합성 시 생성할 웨이퍼 수")
    parser.add_argument("--max-wafers", type=int, default=None, help="실측 시 상한 (None=전체)")
    parser.add_argument("--n-days", type=int, default=120, help="시뮬레이션 기간(일)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--trace-sample-rate", type=float, default=0.05)
    args = parser.parse_args()

    t0 = time.time()
    print(f"{'=' * 70}\n 데이터셋 생성 — source={args.source}\n{'=' * 70}")

    if args.source == "synthetic":
        res, maps = build_synthetic(
            args.n_wafers, args.n_days, args.seed, args.trace_sample_rate
        )
    else:
        if not wm811k.is_available():
            print(
                f"❌ 실측 원본이 없습니다: {wm811k.default_raw_path()}\n"
                f"   python scripts/download_wm811k.py --manual 을 참고하세요.",
                file=sys.stderr,
            )
            return 1
        res, maps = build_real(
            args.max_wafers, args.n_days, args.seed, args.trace_sample_rate
        )

    validate_all(res)
    out = save(res, maps, args.source)

    master = res.wafer_master
    print(f"\n{'=' * 70}\n 완료 ({time.time() - t0:.1f}초) → {out}\n{'=' * 70}")
    print(f"  웨이퍼      : {len(master):,}장 / {master['lot_id'].nunique():,} lot")
    print(f"  기간        : {master['eds_time'].min():%Y-%m-%d} ~ {master['eds_time'].max():%Y-%m-%d}")
    print(f"  평균 수율   : {master['yield_pct'].mean():.2f}%")
    print(f"  FDC         : {len(res.fdc_summary):,}행 × {res.fdc_summary.shape[1]}열")
    print(f"  trace       : {len(res.fdc_trace):,}행")
    print(f"  이상 사건   : {len(res.excursions)}건")
    print("\n  라벨 분포:")
    for label, cnt in master["pattern_label"].value_counts().items():
        print(f"    {label:<12} {cnt:>7,}  ({cnt / len(master) * 100:5.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
