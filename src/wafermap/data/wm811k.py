"""WM-811K 실측 데이터 로더 — `LSWMD.pkl`을 프로젝트 스키마로 변환한다.

무엇을: MIR Lab이 공개한 WM-811K 원본 pickle(811,457장)을 읽어, 합성 데이터와
        **동일한 wafer_master + die_map** 형태로 정규화한다.

어떻게: 원본은 wafer map 배열과 라벨이 섞인 DataFrame이며, 라벨 컬럼이
        중첩 배열([[...]])로 저장되어 있는 등 그대로 쓰기 어렵다. 이 모듈이
        그 지저분함을 흡수해 downstream이 소스를 구분하지 않아도 되게 만든다.

왜 이 계층이 필요한가: 앱의 데이터 소스 토글(§2.3)은 "코드 경로가 완전히 동일하다"는
        전제 위에 있다. 실측/합성 차이를 여기서 전부 흡수하지 못하면, 그 차이가
        피처·모델·화면 코드로 새어 나가 분기가 곳곳에 생긴다.

원본 스키마 (LSWMD.pkl):
    waferMap        : np.ndarray (2D, 0=die 없음 / 1=pass / 2=fail)  ← 우리 규약과 동일
    dieSize         : float
    lotName         : str      (예: 'lot1')
    waferIndex      : float    (lot 내 순번 = slot)
    trianTestLabel  : [['Training']] 등 (오타는 원본 그대로)
    failureType     : [['Edge-Ring']] / [] (미라벨)

참고: docs/00_design.md §2.1(데이터셋), §2.3(개발 순서)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wafermap.config import (
    PATTERN_LABELS,
    PRODUCT_NAME,
    RAW_DIR,
    TECH_NODE,
    WM811K_FILENAME,
)
from wafermap.data.synth_wafer import map_stats

#: 원본 failureType 표기 → 프로젝트 표준 라벨.
#: 원본에는 'Edge-Loc'/'Edge-Loc ' 처럼 공백이 붙거나 대소문자가 흔들리는 값이 섞여 있다.
_LABEL_ALIASES = {
    "center": "Center",
    "donut": "Donut",
    "edge-loc": "Edge-Loc",
    "edge-ring": "Edge-Ring",
    "loc": "Loc",
    "near-full": "Near-full",
    "random": "Random",
    "scratch": "Scratch",
    "none": "none",
}


def default_raw_path() -> Path:
    """원본 pickle의 기본 위치."""
    return RAW_DIR / WM811K_FILENAME


def is_available(path: Path | None = None) -> bool:
    """실측 원본이 준비되어 있는지 확인한다.

    왜: 앱이 '실측' 토글을 눌렀을 때 예외로 죽지 않고, 안내와 함께 합성으로
        폴백해야 한다(§4.6 에러/폴백 UX). 그 판정을 이 함수 하나로 통일한다.
    """
    return (path or default_raw_path()).exists()


def _normalize_label(raw: object) -> str | None:
    """원본 failureType 값을 표준 라벨로 정규화한다.

    원본은 ``[['Edge-Ring']]``, ``[]``, ``'none'`` 등 형태가 제각각이라 단계적으로 벗겨낸다.
    미라벨이면 None을 돌려준다(백로그 §11의 미라벨 확장에서 사용).
    """
    value = raw
    # 중첩 배열/리스트를 스칼라가 나올 때까지 벗긴다
    for _ in range(4):
        if isinstance(value, (list, tuple, np.ndarray)):
            if len(value) == 0:
                return None
            value = value[0]
        else:
            break

    if not isinstance(value, str):
        return None

    key = value.strip().lower()
    return _LABEL_ALIASES.get(key)


def load_raw(path: Path | None = None) -> pd.DataFrame:
    """`LSWMD.pkl`을 그대로 읽어 온다 (정규화 전).

    Raises:
        FileNotFoundError: 원본이 없을 때. 준비 방법을 메시지에 담는다.
    """
    path = path or default_raw_path()
    if not path.exists():
        raise FileNotFoundError(
            f"WM-811K 원본을 찾을 수 없습니다: {path}\n"
            f"  준비 방법: python scripts/download_wm811k.py  (또는 수동으로 {path}에 배치)\n"
            f"  자세한 절차는 docs/00_design.md §2.3 참고"
        )
    return pd.read_pickle(path)


def load(
    path: Path | None = None,
    *,
    labeled_only: bool = True,
    max_wafers: int | None = None,
    seed: int = 0,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    """WM-811K를 프로젝트 스키마로 변환해 돌려준다.

    Args:
        path: 원본 경로 (기본 data/raw/LSWMD.pkl)
        labeled_only: True면 라벨이 있는 172,950장만 사용한다(§M1 확정 범위).
                      False면 미라벨까지 포함 — 백로그 §11 확장용.
        max_wafers: 상한 (개발 중 부분 로드용). None이면 전체.
        seed: max_wafers로 샘플링할 때의 시드

    Returns:
        (wafer_master DataFrame, {wafer_id: 2D map 배열})

    Note:
        wafer_master의 fab_in_time/eds_time은 원본에 없다. 여기서는 비워 두고,
        build_dataset이 FDC 시뮬레이터의 타임라인과 결합할 때 채운다.
        원본에는 공정 이력이 없으므로 이는 불가피한 합성 구간이며, 앱에 명시된다.
    """
    raw = load_raw(path)

    labels = raw["failureType"].map(_normalize_label)
    df = raw.assign(_label=labels)

    if labeled_only:
        df = df[df["_label"].notna()]
    if max_wafers is not None and len(df) > max_wafers:
        df = df.sample(n=max_wafers, random_state=seed)

    df = df.reset_index(drop=True)

    records, maps = [], {}
    for i, row in df.iterrows():
        wafer = np.asarray(row["waferMap"], dtype=np.int8)
        lot = str(row.get("lotName", f"LOT{i:05d}")).strip()

        # 원본 waferIndex는 float(1.0~25.0)이며 결측이 있다
        try:
            slot = int(row.get("waferIndex", 0))
        except (TypeError, ValueError):
            slot = 0
        slot = slot if 1 <= slot <= 25 else (i % 25) + 1

        wafer_id = f"{lot}-W{slot:02d}-{i:06d}"  # 원본은 lot+slot이 유일하지 않아 순번을 덧붙인다
        die_total, die_pass, yield_pct = map_stats(wafer)

        label = row["_label"]
        records.append(
            {
                "wafer_id": wafer_id,
                "lot_id": lot,
                "slot_no": slot,
                "product": PRODUCT_NAME,
                "tech_node": TECH_NODE,
                "die_total": die_total,
                "die_pass": die_pass,
                "yield_pct": yield_pct,
                "pattern_label": label if label is not None else "none",
                "data_source": "real",
                "is_labeled": label is not None,
            }
        )
        maps[wafer_id] = wafer

    master = pd.DataFrame.from_records(records)

    unknown = set(master["pattern_label"]) - set(PATTERN_LABELS)
    if unknown:
        raise ValueError(f"정규화되지 않은 라벨이 있습니다: {sorted(unknown)}")

    return master, maps


def label_distribution(path: Path | None = None) -> pd.Series:
    """원본의 라벨 분포를 센다 (맵을 로드하지 않아 빠르다).

    왜: 실데이터 전환 시 첫 확인 항목이 라벨 분포다(§2.3 체크리스트). 맵 배열까지
        메모리에 올리지 않고 분포만 보려면 별도 경로가 필요하다.
    """
    raw = load_raw(path)
    return raw["failureType"].map(_normalize_label).value_counts(dropna=False)
