"""processed 데이터 로더 — 앱과 분석 코드의 유일한 데이터 진입점.

무엇을: `build_dataset.py`가 만들어 둔 parquet/npz를 읽어 온다. 데이터 소스
        (합성/실측)는 인자 하나로 전환되며, 그 외 호출 방식은 완전히 동일하다.

왜 별도 모듈인가: 앱 화면 코드가 parquet 경로를 직접 알면, 나중에 저장 형식을
        바꾸거나 클라우드용 샘플로 갈아탈 때 화면 코드를 전부 고쳐야 한다.
        경로와 형식을 이 모듈에만 가둬 둔다.

Streamlit 캐시 주의: 이 모듈은 streamlit을 import하지 않는다. 캐싱은 앱 계층에서
        `@st.cache_data(...)`로 감싸며, 이때 **캐시 키에 source를 반드시 포함**해야
        소스 전환 시 옛 결과가 남지 않는다(§4.7).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from wafermap.config import processed_dir

WAFER_MASTER_FILE = "wafer_master.parquet"
FDC_SUMMARY_FILE = "fdc_summary.parquet"
FDC_TRACE_FILE = "fdc_trace.parquet"
GROUND_TRUTH_FILE = "ground_truth.parquet"
DIE_MAP_FILE = "die_map.npz"
EXCURSION_FILE = "excursions.parquet"


class DatasetNotBuiltError(FileNotFoundError):
    """아직 build_dataset을 돌리지 않은 상태. 다음에 할 일을 메시지에 담는다."""


def _require(path: Path, source: str) -> Path:
    if not path.exists():
        raise DatasetNotBuiltError(
            f"데이터가 없습니다: {path}\n"
            f"  먼저 생성하세요: python scripts/build_dataset.py --source {source}"
        )
    return path


def is_built(source: str = "synthetic") -> bool:
    """해당 소스의 데이터셋이 생성되어 있는지 확인한다."""
    return (processed_dir(source) / WAFER_MASTER_FILE).exists()


def available_sources() -> list[str]:
    """생성이 완료된 데이터 소스 목록.

    왜: 앱 사이드바의 소스 토글에 '실측'을 띄울지 말지를 이 함수로 판정한다.
        준비되지 않은 소스를 선택지로 보여 주고 나서 실패시키는 것보다 낫다.
    """
    return [s for s in ("synthetic", "real") if is_built(s)]


def load_wafer_master(source: str = "synthetic") -> pd.DataFrame:
    """웨이퍼 단위 마스터 테이블."""
    return pd.read_parquet(_require(processed_dir(source) / WAFER_MASTER_FILE, source))


def load_fdc_summary(source: str = "synthetic", step_id: str | None = None) -> pd.DataFrame:
    """FDC 요약 통계. step_id를 주면 해당 스텝만 읽는다.

    왜 스텝 필터를 로더에 두나: 전 스텝 FDC는 컬럼이 200개에 육박한다. 원인 분석
        화면은 한 번에 한 스텝만 보므로, 읽은 뒤 거르지 말고 읽을 때 걸러야
        모바일·클라우드 메모리 예산(§4A.7)을 지킬 수 있다.
    """
    path = _require(processed_dir(source) / FDC_SUMMARY_FILE, source)
    if step_id is None:
        return pd.read_parquet(path)
    return pd.read_parquet(path, filters=[("step_id", "==", step_id)])


def load_fdc_trace(
    source: str = "synthetic",
    wafer_ids: list[str] | None = None,
    step_id: str | None = None,
) -> pd.DataFrame:
    """파라미터 시계열. 표본만 저장되어 있으므로 없는 웨이퍼는 빈 결과가 나온다."""
    path = _require(processed_dir(source) / FDC_TRACE_FILE, source)
    filters = []
    if step_id is not None:
        filters.append(("step_id", "==", step_id))
    if wafer_ids:
        filters.append(("wafer_id", "in", list(wafer_ids)))
    return pd.read_parquet(path, filters=filters or None)


def load_ground_truth(source: str = "synthetic") -> pd.DataFrame:
    """정답지.

    ⚠️ 이 테이블은 **검증 전용**이다(§2.2 원칙 4). 피처·모델 입력에 절대 넣지 않는다.
       합성 소스에만 존재하며, 실측에는 원인 정답이 없다.
    """
    return pd.read_parquet(_require(processed_dir(source) / GROUND_TRUTH_FILE, source))


def load_excursions(source: str = "synthetic") -> pd.DataFrame:
    """주입된 이상 사건 목록 (검증 전용)."""
    return pd.read_parquet(_require(processed_dir(source) / EXCURSION_FILE, source))


def load_die_maps(source: str = "synthetic") -> dict[str, np.ndarray]:
    """전체 wafer map을 메모리로 읽는다.

    ⚠️ 대량 데이터에서는 무겁다. 화면에 몇 장만 띄울 때는 `load_die_map` 을 쓸 것.
    """
    path = _require(processed_dir(source) / DIE_MAP_FILE, source)
    with np.load(path) as npz:
        return {k: npz[k] for k in npz.files}


def load_die_map(wafer_id: str, source: str = "synthetic") -> np.ndarray:
    """웨이퍼 1장의 map만 읽는다.

    왜: npz는 지연 로딩이라 필요한 키만 꺼내면 전체를 메모리에 올리지 않는다.
        맵 뷰어(§4.5)처럼 한 장씩 보는 화면은 반드시 이 경로를 쓴다.
    """
    path = _require(processed_dir(source) / DIE_MAP_FILE, source)
    with np.load(path) as npz:
        if wafer_id not in npz.files:
            raise KeyError(f"map을 찾을 수 없습니다: {wafer_id}")
        return npz[wafer_id]


def summary(source: str = "synthetic") -> dict[str, object]:
    """데이터셋 개요 — 앱의 '데이터 개요' 페이지와 CLI 확인용."""
    master = load_wafer_master(source)
    return {
        "source": source,
        "n_wafers": len(master),
        "n_lots": master["lot_id"].nunique(),
        "period": (master["eds_time"].min(), master["eds_time"].max()),
        "mean_yield": float(master["yield_pct"].mean()),
        "label_counts": master["pattern_label"].value_counts().to_dict(),
    }
