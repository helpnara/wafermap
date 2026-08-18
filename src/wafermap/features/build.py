"""피처 파이프라인 — 맵 전체에서 피처 행렬을 만든다.

무엇을: `die_map.npz`의 모든 웨이퍼에 대해 기하·연결성분·Radon 피처를 뽑아
        `features.parquet` 한 장으로 만든다.

왜 별도 단계인가: 피처 추출은 웨이퍼당 약 3ms지만 172,950장이면 10분 가까이 걸린다.
        모델을 재학습할 때마다 다시 뽑으면 실험 반복이 느려지고, 앱에서 실시간으로
        뽑으면 화면이 멈춘다. 한 번 뽑아 저장해 두고 계속 재사용한다.

참고: docs/00_design.md §M1(피처 목록), §4.2(앱 내 학습 금지)
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from wafermap.config import processed_dir
from wafermap.features import connectivity, geometry, radon_feat

FEATURES_FILE = "features.parquet"

#: 피처가 아닌 식별·라벨 컬럼 (모델 입력에서 제외해야 한다)
NON_FEATURE_COLUMNS = ("wafer_id", "pattern_label")


def _report(done: int, total: int, t0: float) -> None:
    """진행 상황을 남은 시간과 함께 출력한다."""
    elapsed = time.time() - t0
    rate = done / elapsed if elapsed else 0.0
    print(
        f"   {done:,}/{total:,} ({done / total * 100:5.1f}%) "
        f"· {rate:,.0f}장/초 · 남은 시간 {(total - done) / max(rate, 1e-9):,.0f}초"
    )


def extract_one(wafer: np.ndarray) -> dict[str, float]:
    """웨이퍼 1장에서 전체 피처를 뽑는다.

    Returns:
        피처명 → 값 딕셔너리 (약 115개)
    """
    out: dict[str, float] = {}
    out.update(geometry.extract(wafer))
    out.update(connectivity.extract(wafer))
    out.update(radon_feat.extract(wafer))
    return out


def _extract_pair(item: tuple[str, np.ndarray]) -> dict[str, object]:
    """(wafer_id, map) 한 쌍에서 피처를 뽑는다 — 병렬 워커용 최상위 함수.

    왜 최상위 함수여야 하나: multiprocessing은 워커에 넘길 함수를 pickle로 직렬화한다.
    지역 함수나 람다는 pickle이 되지 않아 모듈 최상위에 두어야 한다.
    """
    wafer_id, wafer = item
    record: dict[str, object] = {"wafer_id": wafer_id}
    record.update(extract_one(wafer))
    return record


def build(
    maps: dict[str, np.ndarray],
    labels: pd.Series | None = None,
    *,
    verbose: bool = True,
    n_jobs: int = 1,
) -> pd.DataFrame:
    """맵 딕셔너리에서 피처 행렬을 만든다.

    Args:
        maps: {wafer_id: 2D 배열}
        labels: wafer_id를 인덱스로 갖는 패턴 라벨 (없으면 라벨 컬럼 생략)
        verbose: 진행 상황 출력 여부
        n_jobs: 병렬 프로세스 수. 1이면 단일 프로세스, -1이면 CPU 수만큼.

    Returns:
        wafer_id + 피처들 (+ pattern_label) DataFrame

    Note:
        웨이퍼당 약 10ms가 걸리며 그중 70%가 Radon 변환이다. 실측 172,950장이면
        단일 프로세스로 약 30분이 걸리므로, 실데이터 추출 시에는 n_jobs를 올린다.
        피처 추출은 웨이퍼끼리 완전히 독립적이라 병렬화 효율이 좋다.
    """
    t0 = time.time()
    total = len(maps)
    items = list(maps.items())

    if n_jobs == 1:
        rows: list[dict[str, object]] = []
        for i, item in enumerate(items, start=1):
            rows.append(_extract_pair(item))
            if verbose and (i % 2_000 == 0 or i == total):
                _report(i, total, t0)
    else:
        import multiprocessing as mp

        workers = mp.cpu_count() if n_jobs < 0 else n_jobs
        chunk = max(1, total // (workers * 8))
        rows = []
        with mp.Pool(workers) as pool:
            for i, record in enumerate(
                pool.imap(_extract_pair, items, chunksize=chunk), start=1
            ):
                rows.append(record)
                if verbose and (i % 2_000 == 0 or i == total):
                    _report(i, total, t0)

    df = pd.DataFrame(rows)

    if labels is not None:
        df["pattern_label"] = df["wafer_id"].map(labels)

    # 수치 안정성: 무한대/결측을 0으로 정리한다.
    # 왜: 나눗셈 기반 비율 피처(edge_inner_ratio 등)는 분모가 0에 가까우면 폭주한다.
    #     LightGBM은 NaN을 다루지만 inf는 처리하지 못해 학습이 실패한다.
    feature_cols = [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
    df[feature_cols] = (
        df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype("float32")
    )
    return df


def feature_columns(df: pd.DataFrame) -> list[str]:
    """모델 입력으로 쓸 피처 컬럼 목록."""
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]


def save(df: pd.DataFrame, source: str) -> Path:
    out = processed_dir(source) / FEATURES_FILE
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    return out


def load(source: str = "synthetic") -> pd.DataFrame:
    """저장된 피처 행렬을 읽는다."""
    path = processed_dir(source) / FEATURES_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"피처가 없습니다: {path}\n"
            f"  먼저 생성하세요: python scripts/build_features.py --source {source}"
        )
    return pd.read_parquet(path)


def is_built(source: str = "synthetic") -> bool:
    return (processed_dir(source) / FEATURES_FILE).exists()
