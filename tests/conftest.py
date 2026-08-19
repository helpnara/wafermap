"""pytest 공통 설정 — src 레이아웃을 import 경로에 추가한다."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wafermap.data.fdc_simulator import simulate  # noqa: E402
from wafermap.data.synth_wafer import build_geometry  # noqa: E402


@pytest.fixture(scope="session")
def geom():
    """웨이퍼 기하 — 생성 비용이 있어 세션 단위로 재사용한다."""
    return build_geometry()


@pytest.fixture(scope="session")
def sim():
    """작은 시뮬레이션 결과 — 여러 테스트가 공유한다."""
    return simulate(n_wafers=1_500, n_days=60, seed=1234)


@pytest.fixture
def rng():
    return np.random.default_rng(0)


# ──────────────────────────────────────────────────────────────────────────
# 공용 인공 FDC 생성기
#
# 왜 conftest에 두나 ★: 처음에는 `tests/test_recommend.py`가
# `from tests.test_cause_model import _make_fdc` 로 다른 테스트 모듈을 직접
# 임포트했다. 그런데 `tests`는 패키지가 아니라서(`__init__.py` 없음) 이 임포트는
# **프로젝트 루트가 sys.path에 있을 때만** 통한다. `python -m pytest`는 현재
# 디렉터리를 sys.path에 넣어 주지만 `pytest`는 넣지 않는다. 즉 실행 방식에 따라
# 되기도 하고 안 되기도 하는 코드였다.
#
# conftest.py의 픽스처는 pytest가 **실행 방식과 무관하게** 자동으로 주입하므로
# 이런 경로 문제가 원천적으로 생기지 않는다. 테스트끼리 임포트로 얽지 않는 것이
# 옳은 방향이기도 하다.
# ──────────────────────────────────────────────────────────────────────────

#: 인공 FDC의 파라미터 규격 — (목표값, 표준편차)
_FDC_PARAMS: dict[str, tuple[float, float]] = {
    "chamber_pressure": (45.0, 0.8),
    "rf_power": (1500.0, 12.0),
    "o2_flow": (20.0, 0.6),
    "cf4_flow": (120.0, 1.5),
    "electrode_temp": (60.0, 0.7),
}


def build_fdc(
    n_case: int = 60,
    n_control: int = 300,
    culprit: str = "chamber_pressure",
    shift: float = 3.0,
    seed: int = 0,
) -> tuple[pd.DataFrame, list[str], list[str]]:
    """인공 FDC 데이터. `culprit` 파라미터만 불량군에서 이동시킨다.

    Returns:
        (fdc, 불량 웨이퍼 id, 정상 웨이퍼 id)
    """
    rng = np.random.default_rng(seed)
    n = n_case + n_control
    is_case = np.array([True] * n_case + [False] * n_control)

    data: dict[str, object] = {
        "wafer_id": [f"W{i:04d}" for i in range(n)],
        "step_id": "P020",
        "step_name": "Etch",
        "equip_id": "ETCH-A",
        "chamber_id": "ETCH-A/ch1",
        "recipe_id": "RCP-STD-01",
        "run_time": pd.Timestamp("2026-01-01"),
    }
    for name, (nominal, sigma) in _FDC_PARAMS.items():
        values = rng.normal(nominal, sigma, n)
        if name == culprit:
            values = values + is_case * shift * sigma
        data[f"{name}_mean"] = values
        data[f"{name}_std"] = np.abs(rng.normal(sigma * 0.3, sigma * 0.05, n))

    fdc = pd.DataFrame(data)
    case_ids = list(fdc.loc[is_case, "wafer_id"])
    control_ids = list(fdc.loc[~is_case, "wafer_id"])
    return fdc, case_ids, control_ids


@pytest.fixture(scope="session")
def make_fdc():
    """인공 FDC 생성기를 함수째로 넘겨준다 (테스트마다 인자가 달라서).

    세션 스코프인 이유: 생성기는 상태가 없는 함수라 공유해도 안전하고,
    **모듈 스코프 픽스처에서도 받을 수 있어야** 하기 때문이다.
    (좁은 스코프의 픽스처는 넓은 스코프에서 못 받는다.)
    """
    return build_fdc
